import json
import os
import sys
from pathlib import Path


def main():
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "tourism_project.settings")

    import django

    django.setup()
    from ai_chatbot import views
    from ai_chatbot.recommenders import predict_accommodation_relevance_from_features

    intent_scenarios = [
        {"expected_intent": "book_accommodation", "message": "book hotel for 2 guests tomorrow"},
        {"expected_intent": "calculate_accommodation_billing", "message": "calculate hotel bill for 2 nights"},
        {"expected_intent": "calculate_billing", "message": "calculate tour billing for 4 guests"},
        {"expected_intent": "get_accommodation_recommendation", "message": "recommend inn in Bayawan under 2000"},
        {"expected_intent": "get_recommendation", "message": "recommend tourist spots for family trip"},
        {"expected_intent": "get_tourism_information", "message": "tourist spot operating hours in bayawan"},
    ]

    intent_results = []
    for item in intent_scenarios:
        pred = views._classify_intent_with_text_cnn(item["message"])
        intent_results.append(
            {
                "message": item["message"],
                "expected_intent": item["expected_intent"],
                "predicted_intent": str(pred.get("intent") or ""),
                "confidence": float(pred.get("confidence", 0.0) or 0.0),
                "source": str(pred.get("source") or ""),
                "error": str(pred.get("error") or ""),
                "is_correct": bool(str(pred.get("intent") or "") == item["expected_intent"]),
            }
        )

    dt_samples = [
        {
            "name": "high_fit_candidate",
            "features": {
                "requested_guests": 2,
                "requested_budget": 2800,
                "requested_location": "Bayawan City",
                "requested_accommodation_type": "hotel",
                "room_price_per_night": 2300,
                "room_capacity": 3,
                "room_available": 2,
                "accom_location": "Bayawan City",
                "company_type": "hotel",
                "nights_requested": 2,
                "cnn_confidence": 0.92,
                "shown_rank": 1,
            },
        },
        {
            "name": "low_fit_candidate",
            "features": {
                "requested_guests": 5,
                "requested_budget": 1200,
                "requested_location": "Poblacion",
                "requested_accommodation_type": "inn",
                "room_price_per_night": 3600,
                "room_capacity": 2,
                "room_available": 1,
                "accom_location": "Banga",
                "company_type": "hotel",
                "nights_requested": 3,
                "cnn_confidence": 0.76,
                "shown_rank": 6,
            },
        },
        {
            "name": "missing_inputs_defaulted",
            "features": {
                "requested_guests": "",
                "requested_budget": "",
                "requested_location": "Bayawan City",
                "requested_accommodation_type": "",
                "room_price_per_night": 1700,
                "room_capacity": 2,
                "room_available": 1,
                "accom_location": "Suba",
                "company_type": "inn",
                "nights_requested": "",
                "cnn_confidence": "",
                "shown_rank": "",
            },
        },
    ]

    dt_results = []
    for sample in dt_samples:
        pred = predict_accommodation_relevance_from_features(sample["features"])
        dt_results.append(
            {
                "sample": sample["name"],
                "prediction": pred,
            }
        )

    weak_points = []
    wrong_intents = [r for r in intent_results if not r["is_correct"]]
    if wrong_intents:
        weak_points.append(
            f"Intent classifier mismatch on {len(wrong_intents)}/{len(intent_results)} end-to-end intent probes."
        )
    low_conf = [r for r in intent_results if r["confidence"] < 0.6]
    if low_conf:
        weak_points.append(
            f"Low-confidence intent predictions observed in {len(low_conf)} probes (<0.60 confidence)."
        )
    dt_all_relevant = all(
        str(d["prediction"].get("predicted_label") or "") == "relevant"
        for d in dt_results
    )
    if dt_all_relevant:
        weak_points.append("Decision Tree probes all predicted 'relevant'; threshold calibration may need tightening.")

    payload = {
        "intent_results": intent_results,
        "decision_tree_results": dt_results,
        "weak_points": weak_points,
        "summary": {
            "intent_probe_total": len(intent_results),
            "intent_probe_correct": len(intent_results) - len(wrong_intents),
            "decision_tree_probe_total": len(dt_results),
        },
    }

    out_path = Path("artifacts/validation/end_to_end_validation_v1.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output_json": str(out_path), "summary": payload["summary"], "weak_points": weak_points}, indent=2))


if __name__ == "__main__":
    main()
