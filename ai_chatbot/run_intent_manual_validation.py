import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd


def _load_queries_csv(input_csv: Path) -> pd.DataFrame:
    rows = []
    with input_csv.open("r", encoding="utf-8-sig") as f:
        lines = [line.rstrip("\n\r") for line in f.readlines() if line.strip()]
    if not lines:
        raise ValueError(f"Input CSV is empty: {input_csv}")
    header = [h.strip() for h in lines[0].split(",")]
    if header[:3] != ["query_id", "expected_intent", "input_text"]:
        raise ValueError(
            "Input CSV header must start with: query_id,expected_intent,input_text"
        )
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


def run_validation(input_csv: Path, output_csv: Path, summary_json: Path):
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "tourism_project.settings")
    import django

    django.setup()
    from ai_chatbot import views as chatbot_views

    model_path, model_source = chatbot_views._resolve_intent_text_cnn_model_path()
    label_map_path = chatbot_views._default_label_map_path_for_model(model_path)
    df = _load_queries_csv(input_csv)

    rows = []
    for _, row in df.iterrows():
        query_id = str(row.get("query_id", "")).strip()
        expected_intent = str(row.get("expected_intent", "")).strip()
        input_text = str(row.get("input_text", "")).strip()
        if not expected_intent or not input_text:
            continue

        result = chatbot_views._classify_intent_with_text_cnn(input_text)
        predicted_intent = str(result.get("intent") or "").strip()
        confidence = float(result.get("confidence", 0.0) or 0.0)
        rows.append(
            {
                "query_id": query_id,
                "expected_intent": expected_intent,
                "input_text": input_text,
                "predicted_intent": predicted_intent,
                "confidence": round(confidence, 6),
                "is_correct": int(predicted_intent == expected_intent),
                "source": str(result.get("source") or ""),
                "error": str(result.get("error") or ""),
                "top_3_json": json.dumps(result.get("top_3", []), ensure_ascii=True),
            }
        )

    out_df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False, encoding="utf-8")

    overall_accuracy = float(out_df["is_correct"].mean()) if not out_df.empty else 0.0
    per_intent = {}
    confusion_pairs = []
    if not out_df.empty:
        for intent, group in out_df.groupby("expected_intent"):
            per_intent[intent] = {
                "samples": int(len(group)),
                "accuracy": float(group["is_correct"].mean()),
                "errors": int(len(group) - int(group["is_correct"].sum())),
            }
        wrong_df = out_df[out_df["is_correct"] == 0].copy()
        if not wrong_df.empty:
            pair_counts = (
                wrong_df.groupby(["expected_intent", "predicted_intent"])
                .size()
                .reset_index(name="count")
                .sort_values("count", ascending=False)
            )
            confusion_pairs = pair_counts.to_dict(orient="records")

    summary = {
        "input_csv": str(input_csv),
        "runtime_model_path": str(model_path),
        "runtime_model_source": model_source,
        "runtime_label_map_path": str(label_map_path),
        "runtime_model_exists": bool(Path(model_path).exists()),
        "runtime_label_map_exists": bool(Path(label_map_path).exists()),
        "total_samples": int(len(out_df)),
        "overall_accuracy": overall_accuracy,
        "per_intent": per_intent,
        "top_confusions": confusion_pairs[:10],
        "outputs": {
            "predictions_csv": str(output_csv),
            "summary_json": str(summary_json),
        },
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="Run manual intent validation using the same Text-CNN runtime flow as Django."
    )
    parser.add_argument(
        "--input-csv",
        default="thesis_data_templates/text_cnn_intent_manual_validation_queries_v1.csv",
    )
    parser.add_argument(
        "--output-csv",
        default="artifacts/text_cnn_intent/manual_validation/manual_validation_predictions_v1.csv",
    )
    parser.add_argument(
        "--summary-json",
        default="artifacts/text_cnn_intent/manual_validation/manual_validation_summary_v1.json",
    )
    args = parser.parse_args()
    run_validation(
        input_csv=Path(args.input_csv),
        output_csv=Path(args.output_csv),
        summary_json=Path(args.summary_json),
    )


if __name__ == "__main__":
    main()
