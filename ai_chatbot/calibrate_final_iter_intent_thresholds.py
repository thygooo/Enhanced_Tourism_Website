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


def run_threshold(views, df: pd.DataFrame, threshold: float) -> tuple[dict, pd.DataFrame]:
    os.environ["CHATBOT_INTENT_CNN_CONFIDENCE_THRESHOLD"] = str(threshold)
    rows = []
    for _, row in df.iterrows():
        qid = str(row["query_id"])
        expected = str(row["expected_intent"])
        text = str(row["input_text"])

        raw, _err = views._predict_text_cnn_labels(
            text=text,
            model_path=views._default_intent_text_cnn_model_path(),
            label_map_path=views._default_label_map_path_for_model(views._default_intent_text_cnn_model_path()),
        )
        raw_pred = views._normalize_intent_label(raw.get("predicted_class")) if isinstance(raw, dict) else ""
        raw_conf = float(raw.get("confidence", 0.0) or 0.0) if isinstance(raw, dict) else 0.0

        gated = views._classify_intent_with_text_cnn(text)
        final = views._classify_intent_and_extract_params(text)

        rows.append(
            {
                "threshold": float(threshold),
                "query_id": qid,
                "expected_intent": expected,
                "raw_pred_intent": raw_pred,
                "raw_confidence": raw_conf,
                "raw_is_correct": int(raw_pred == expected),
                "gated_intent": str(gated.get("intent") or ""),
                "gated_source": str(gated.get("source") or ""),
                "final_intent": str(final.get("intent") or ""),
                "final_source": str(final.get("source") or ""),
                "final_is_correct": int(str(final.get("intent") or "") == expected),
            }
        )

    out_df = pd.DataFrame(rows)
    summary = {
        "threshold": float(threshold),
        "raw_cnn_accuracy": float(out_df["raw_is_correct"].mean()) if not out_df.empty else 0.0,
        "fallback_usage_count": int((out_df["final_source"] == "heuristic_intent_fallback").sum()),
        "final_routed_accuracy": float(out_df["final_is_correct"].mean()) if not out_df.empty else 0.0,
    }
    return summary, out_df


def analyze_intent_safety(all_rows: pd.DataFrame, thresholds: list[float]) -> dict:
    # Intent-level direct-routing safety at lower thresholds.
    per_intent = {}
    for thr in thresholds:
        tdf = all_rows[all_rows["threshold"] == float(thr)].copy()
        direct_df = tdf[tdf["gated_source"] == "text_cnn_intent"].copy()
        intent_stats = {}
        for intent, g in direct_df.groupby("gated_intent"):
            intent_stats[intent] = {
                "accepted_count": int(len(g)),
                "accepted_accuracy": float(g["final_is_correct"].mean()) if len(g) else 0.0,
            }
        per_intent[str(thr)] = intent_stats

    # Aggregate across thresholds <= 0.40 to detect safer intents under lower confidence gates.
    lower_df = all_rows[all_rows["threshold"].isin([0.10, 0.20, 0.30, 0.40])].copy()
    lower_direct = lower_df[lower_df["gated_source"] == "text_cnn_intent"].copy()
    agg_stats = {}
    for intent, g in lower_direct.groupby("gated_intent"):
        agg_stats[intent] = {
            "accepted_count": int(len(g)),
            "accepted_accuracy": float(g["final_is_correct"].mean()) if len(g) else 0.0,
        }

    safest = [
        {"intent": k, **v}
        for k, v in agg_stats.items()
        if v["accepted_count"] >= 5 and v["accepted_accuracy"] >= 0.70
    ]
    riskiest = [
        {"intent": k, **v}
        for k, v in agg_stats.items()
        if v["accepted_count"] >= 5 and v["accepted_accuracy"] <= 0.40
    ]
    return {
        "per_threshold_direct_intent_stats": per_intent,
        "lower_threshold_aggregate_direct_intent_stats": agg_stats,
        "safest_intents_for_direct_routing": safest,
        "riskiest_intents_for_direct_routing": riskiest,
    }


def main():
    parser = argparse.ArgumentParser(description="Calibration-only threshold sweep for final iteration CNN model.")
    parser.add_argument("--manual-query-csv", default="thesis_data_templates/text_cnn_intent_manual_validation_queries_v1.csv")
    parser.add_argument("--model-path", default="artifacts/text_cnn_intent_final_iter_v1/text_cnn_intent.h5")
    parser.add_argument("--thresholds", default="0.10,0.20,0.30,0.40,0.50,0.60")
    parser.add_argument("--out-json", default="artifacts/text_cnn_intent/manual_validation/final_iter_calibration_sweep_v1.json")
    parser.add_argument("--out-csv", default="artifacts/text_cnn_intent/manual_validation/final_iter_calibration_sweep_v1.csv")
    parser.add_argument("--out-rows-csv", default="artifacts/text_cnn_intent/manual_validation/final_iter_calibration_rows_v1.csv")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "tourism_project.settings")
    os.environ["CHATBOT_INTENT_CNN_MODEL_PATH"] = str(args.model_path)
    import django

    django.setup()
    from ai_chatbot import views

    # Ensure clean model cache for selected model path.
    views._TEXT_CNN_MODEL_CACHE = None
    views._TEXT_CNN_MODEL_PATH_CACHE = None

    manual_df = parse_manual_queries(Path(args.manual_query_csv))
    thresholds = [float(v.strip()) for v in str(args.thresholds).split(",") if str(v).strip()]

    summary_rows = []
    all_rows = []
    for thr in thresholds:
        s, rdf = run_threshold(views, manual_df, thr)
        summary_rows.append(s)
        all_rows.append(rdf)

    summary_df = pd.DataFrame(summary_rows)
    detail_df = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()

    intent_analysis = analyze_intent_safety(detail_df, thresholds)
    best_row = (
        summary_df.sort_values(["final_routed_accuracy", "fallback_usage_count"], ascending=[False, True]).iloc[0].to_dict()
        if not summary_df.empty
        else {}
    )

    # Compare to current default calibration point 0.60.
    base_060 = summary_df[summary_df["threshold"] == 0.60]
    base_060_acc = float(base_060.iloc[0]["final_routed_accuracy"]) if not base_060.empty else None
    best_acc = float(best_row.get("final_routed_accuracy")) if best_row else None
    improvement_vs_060 = (best_acc - base_060_acc) if (best_acc is not None and base_060_acc is not None) else None

    recommendation = {
        "best_threshold": best_row.get("threshold"),
        "best_final_routed_accuracy": best_row.get("final_routed_accuracy"),
        "improvement_vs_threshold_0_60": improvement_vs_060,
        "note": (
            "Use global threshold at best_threshold for demo if maximizing manual realistic routed accuracy is priority."
            if improvement_vs_060 is not None and improvement_vs_060 > 0
            else "Keep conservative threshold and use fallback-first strategy; no global gain over 0.60."
        ),
        "proposed_selective_routing_rules": [
            "Allow direct CNN routing for intents listed in safest_intents_for_direct_routing when confidence >= 0.35.",
            "Keep fallback-first behavior for intents listed in riskiest_intents_for_direct_routing unless confidence >= 0.70.",
            "For billing intents, require explicit keyword confirmation (tour/hotel/accommodation) before accepting direct CNN at low confidence.",
        ],
    }

    payload = {
        "model_path": str(args.model_path),
        "manual_query_csv": str(args.manual_query_csv),
        "thresholds_tested": thresholds,
        "threshold_summary": summary_rows,
        "intent_level_analysis": intent_analysis,
        "recommendation": recommendation,
        "outputs": {
            "summary_csv": str(args.out_csv),
            "rows_csv": str(args.out_rows_csv),
            "summary_json": str(args.out_json),
        },
    }

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(args.out_csv, index=False, encoding="utf-8")
    detail_df.to_csv(args.out_rows_csv, index=False, encoding="utf-8")
    Path(args.out_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
