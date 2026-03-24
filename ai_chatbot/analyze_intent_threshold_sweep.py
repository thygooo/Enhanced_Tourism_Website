import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd


def parse_manual_queries(path: Path) -> pd.DataFrame:
    rows = []
    lines = [line.rstrip("\n\r") for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    for line in lines[1:]:
        parts = line.split(",", 2)
        if len(parts) != 3:
            continue
        rows.append(
            {
                "query_id": parts[0].strip(),
                "expected_intent": parts[1].strip(),
                "input_text": parts[2].strip(),
            }
        )
    return pd.DataFrame(rows)


def run_sweep_for_model(views, manual_df: pd.DataFrame, model_path: str, thresholds: list[float]):
    results = []
    per_threshold_rows = []

    os.environ["CHATBOT_INTENT_CNN_MODEL_PATH"] = model_path
    # Reset cache when switching model path.
    views._TEXT_CNN_MODEL_CACHE = None
    views._TEXT_CNN_MODEL_PATH_CACHE = None

    resolved_model_path, _ = views._resolve_intent_text_cnn_model_path()
    resolved_label_map = views._default_label_map_path_for_model(resolved_model_path)

    for thr in thresholds:
        os.environ["CHATBOT_INTENT_CNN_CONFIDENCE_THRESHOLD"] = str(thr)
        row_records = []
        for _, row in manual_df.iterrows():
            qid = str(row["query_id"])
            expected = str(row["expected_intent"])
            text = str(row["input_text"])

            raw, raw_err = views._predict_text_cnn_labels(
                text=text,
                model_path=resolved_model_path,
                label_map_path=resolved_label_map,
            )
            raw_pred = ""
            raw_conf = 0.0
            if raw and not raw_err:
                raw_pred = views._normalize_intent_label(raw.get("predicted_class"))
                raw_conf = float(raw.get("confidence", 0.0) or 0.0)

            gated = views._classify_intent_with_text_cnn(text)
            final = views._classify_intent_and_extract_params(text)

            row_records.append(
                {
                    "query_id": qid,
                    "expected_intent": expected,
                    "raw_pred_intent": raw_pred,
                    "raw_confidence": raw_conf,
                    "raw_is_correct": int(raw_pred == expected),
                    "gated_source": str(gated.get("source") or ""),
                    "gated_intent": str(gated.get("intent") or ""),
                    "final_source": str(final.get("source") or ""),
                    "final_intent": str(final.get("intent") or ""),
                    "final_is_correct": int(str(final.get("intent") or "") == expected),
                }
            )

        rows_df = pd.DataFrame(row_records)
        raw_acc = float(rows_df["raw_is_correct"].mean()) if not rows_df.empty else 0.0
        final_acc = float(rows_df["final_is_correct"].mean()) if not rows_df.empty else 0.0
        fallback_count = int((rows_df["final_source"] == "heuristic_intent_fallback").sum())
        low_conf_count = int((rows_df["gated_source"] == "text_cnn_low_confidence").sum())
        accepted_df = rows_df[rows_df["gated_intent"].astype(str).str.strip() != ""]
        gated_acc = float(accepted_df["final_is_correct"].mean()) if not accepted_df.empty else 0.0

        results.append(
            {
                "threshold": float(thr),
                "raw_cnn_accuracy": raw_acc,
                "fallback_usage_count": fallback_count,
                "cnn_low_confidence_count": low_conf_count,
                "cnn_accepted_count": int(len(accepted_df)),
                "cnn_accepted_accuracy": gated_acc,
                "final_routed_accuracy": final_acc,
            }
        )

        rows_df["threshold"] = float(thr)
        per_threshold_rows.append(rows_df)

    combined_rows = pd.concat(per_threshold_rows, ignore_index=True) if per_threshold_rows else pd.DataFrame()
    return results, combined_rows


def compare_old_new_rows(old_path: Path, new_path: Path) -> tuple[pd.DataFrame, dict]:
    old_df = pd.read_csv(old_path)
    new_df = pd.read_csv(new_path)
    merged = old_df.merge(
        new_df,
        on=["query_id", "expected_intent"],
        suffixes=("_old", "_new"),
        how="inner",
    )
    merged["status_change"] = "unchanged"
    merged.loc[(merged["is_final_correct_old"] == 1) & (merged["is_final_correct_new"] == 0), "status_change"] = "regressed"
    merged.loc[(merged["is_final_correct_old"] == 0) & (merged["is_final_correct_new"] == 1), "status_change"] = "improved"

    summary = {
        "rows_compared": int(len(merged)),
        "regressed_count": int((merged["status_change"] == "regressed").sum()),
        "improved_count": int((merged["status_change"] == "improved").sum()),
        "unchanged_count": int((merged["status_change"] == "unchanged").sum()),
        "regressed_by_intent": {
            str(k): int(v)
            for k, v in merged[merged["status_change"] == "regressed"]["expected_intent"].value_counts().to_dict().items()
        },
        "improved_by_intent": {
            str(k): int(v)
            for k, v in merged[merged["status_change"] == "improved"]["expected_intent"].value_counts().to_dict().items()
        },
    }
    return merged, summary


def pick_best_threshold(results: list[dict]) -> dict:
    if not results:
        return {}
    ranked = sorted(
        results,
        key=lambda r: (float(r["final_routed_accuracy"]), -int(r["fallback_usage_count"])),
        reverse=True,
    )
    return ranked[0]


def main():
    parser = argparse.ArgumentParser(description="Analyze CNN threshold/fallback behavior on manual realistic validation set.")
    parser.add_argument("--manual-query-csv", default="thesis_data_templates/text_cnn_intent_manual_validation_queries_v1.csv")
    parser.add_argument("--old-model-path", default="artifacts/text_cnn_intent/text_cnn_intent.h5")
    parser.add_argument("--new-model-path", default="artifacts/text_cnn_intent_robust_round1/text_cnn_intent.h5")
    parser.add_argument("--thresholds", default="0.30,0.40,0.50,0.60,0.70")
    parser.add_argument("--old-fallback-csv", default="artifacts/text_cnn_intent/manual_validation/fallback_routing_predictions_v1.csv")
    parser.add_argument("--new-fallback-csv", default="artifacts/text_cnn_intent/manual_validation/fallback_routing_predictions_round1.csv")
    parser.add_argument("--out-json", default="artifacts/text_cnn_intent/manual_validation/threshold_sweep_analysis_v1.json")
    parser.add_argument("--out-csv", default="artifacts/text_cnn_intent/manual_validation/threshold_sweep_analysis_v1.csv")
    parser.add_argument("--out-row-compare-csv", default="artifacts/text_cnn_intent/manual_validation/old_vs_new_row_compare_v1.csv")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "tourism_project.settings")
    import django

    django.setup()
    from ai_chatbot import views

    manual_df = parse_manual_queries(Path(args.manual_query_csv))
    thresholds = [float(x.strip()) for x in str(args.thresholds).split(",") if str(x).strip()]

    old_results, old_rows = run_sweep_for_model(
        views=views,
        manual_df=manual_df,
        model_path=str(args.old_model_path),
        thresholds=thresholds,
    )
    new_results, new_rows = run_sweep_for_model(
        views=views,
        manual_df=manual_df,
        model_path=str(args.new_model_path),
        thresholds=thresholds,
    )

    row_compare_df, row_compare_summary = compare_old_new_rows(
        old_path=Path(args.old_fallback_csv),
        new_path=Path(args.new_fallback_csv),
    )
    Path(args.out_row_compare_csv).parent.mkdir(parents=True, exist_ok=True)
    row_compare_df.to_csv(args.out_row_compare_csv, index=False, encoding="utf-8")

    best_old = pick_best_threshold(old_results)
    best_new = pick_best_threshold(new_results)

    analysis = {
        "thresholds_tested": thresholds,
        "old_model_path": str(args.old_model_path),
        "new_model_path": str(args.new_model_path),
        "old_model_sweep": old_results,
        "new_model_sweep": new_results,
        "best_threshold_old_model": best_old,
        "best_threshold_new_model": best_new,
        "row_by_row_old_vs_new_summary": row_compare_summary,
        "inference": {
            "raw_accuracy_change_at_0_60": (
                next((r["raw_cnn_accuracy"] for r in new_results if abs(r["threshold"] - 0.60) < 1e-9), 0.0)
                - next((r["raw_cnn_accuracy"] for r in old_results if abs(r["threshold"] - 0.60) < 1e-9), 0.0)
            ),
            "final_accuracy_change_at_0_60": (
                next((r["final_routed_accuracy"] for r in new_results if abs(r["threshold"] - 0.60) < 1e-9), 0.0)
                - next((r["final_routed_accuracy"] for r in old_results if abs(r["threshold"] - 0.60) < 1e-9), 0.0)
            ),
            "fallback_usage_change_at_0_60": (
                next((r["fallback_usage_count"] for r in new_results if abs(r["threshold"] - 0.60) < 1e-9), 0)
                - next((r["fallback_usage_count"] for r in old_results if abs(r["threshold"] - 0.60) < 1e-9), 0)
            ),
        },
        "outputs": {
            "analysis_json": str(args.out_json),
            "analysis_csv": str(args.out_csv),
            "row_compare_csv": str(args.out_row_compare_csv),
        },
    }

    combined_table = []
    for r in old_results:
        combined_table.append({"model": "old", **r})
    for r in new_results:
        combined_table.append({"model": "new", **r})
    pd.DataFrame(combined_table).to_csv(args.out_csv, index=False, encoding="utf-8")

    Path(args.out_json).write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()
