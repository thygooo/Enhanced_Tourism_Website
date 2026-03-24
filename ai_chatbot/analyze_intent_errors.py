import json
from pathlib import Path

import pandas as pd


def assign_cause_and_fix(true_intent: str, pred_intent: str, confidence: float, text: str):
    msg = str(text or "").lower()
    conf = float(confidence or 0.0)

    if conf < 0.40:
        return (
            "low_confidence_distribution_shift",
            "Route to heuristic fallback when confidence is below threshold; expand paraphrase coverage for this intent.",
        )

    if true_intent == "calculate_accommodation_billing" and pred_intent == "calculate_billing":
        return (
            "billing_scope_overlap",
            "Add stronger accommodation-billing cues (hotel/room/night/checkout) in training paraphrases.",
        )

    if true_intent == "calculate_billing" and pred_intent in ("book_accommodation", "get_tourism_information"):
        return (
            "tour_billing_vs_booking_or_info_overlap",
            "Add explicit tour-billing phrase patterns and hard negatives against booking/info phrasings.",
        )

    if true_intent == "book_accommodation" and pred_intent in ("get_accommodation_recommendation", "get_recommendation"):
        return (
            "booking_vs_recommendation_action_overlap",
            "Increase action-oriented booking paraphrases using confirm/finalize/reserve-now style verbs.",
        )

    if true_intent in ("get_accommodation_recommendation", "get_recommendation") and pred_intent == "get_tourism_information":
        return (
            "recommendation_vs_information_question_style_overlap",
            "Add recommendation paraphrases in interrogative form and include contrastive info-only negatives.",
        )

    if true_intent == "get_tourism_information" and pred_intent == "get_recommendation":
        return (
            "information_vs_recommendation_overlap",
            "Add stricter tourism-information paraphrases with operating-hours/contact/fees patterns.",
        )

    if "?" in msg or any(token in msg for token in ["what", "how", "where", "who"]):
        return (
            "generic_interrogative_bias",
            "Balance question-form samples across all intents; not only information-seeking intent.",
        )

    return (
        "general_semantic_overlap",
        "Expand diverse paraphrases and calibrate confidence threshold with validation set.",
    )


def main():
    pred_path = Path("artifacts/text_cnn_intent/manual_validation/manual_validation_predictions_v1.csv")
    summary_path = Path("artifacts/text_cnn_intent/manual_validation/manual_validation_summary_v1.json")
    out_rows_path = Path("artifacts/text_cnn_intent/manual_validation/error_analysis_rows_v1.csv")
    out_summary_path = Path("artifacts/text_cnn_intent/manual_validation/error_analysis_summary_v1.json")

    df = pd.read_csv(pred_path)
    wrong = df[df["is_correct"] == 0].copy()
    causes = []
    fixes = []
    for _, row in wrong.iterrows():
        cause, fix = assign_cause_and_fix(
            true_intent=str(row["expected_intent"]),
            pred_intent=str(row["predicted_intent"]),
            confidence=float(row.get("confidence", 0.0) or 0.0),
            text=str(row.get("input_text", "")),
        )
        causes.append(cause)
        fixes.append(fix)

    wrong["likely_cause_of_failure"] = causes
    wrong["suggested_fix"] = fixes
    wrong = wrong[
        [
            "query_id",
            "expected_intent",
            "predicted_intent",
            "confidence",
            "input_text",
            "likely_cause_of_failure",
            "suggested_fix",
        ]
    ]
    out_rows_path.parent.mkdir(parents=True, exist_ok=True)
    wrong.to_csv(out_rows_path, index=False, encoding="utf-8")

    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    cause_counts = wrong["likely_cause_of_failure"].value_counts().to_dict()
    low_conf_count = int((wrong["confidence"] < 0.60).sum())

    report = {
        "input_predictions_csv": str(pred_path),
        "input_summary_json": str(summary_path),
        "error_rows": int(len(wrong)),
        "error_rate": float(len(wrong) / len(df)) if len(df) else 0.0,
        "low_confidence_errors_below_0_60": low_conf_count,
        "failing_intents_by_error_count": {
            str(k): int(v)
            for k, v in wrong["expected_intent"].value_counts().to_dict().items()
        },
        "top_confusions": summary_payload.get("top_confusions", []),
        "cause_distribution": {str(k): int(v) for k, v in cause_counts.items()},
        "outputs": {
            "error_rows_csv": str(out_rows_path),
            "error_summary_json": str(out_summary_path),
        },
    }
    out_summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
