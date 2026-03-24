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
    from ai_chatbot.recommenders import predict_accommodation_relevance_from_features

    samples = [
        {
            "name": "strong_match_hotel",
            "requested_guests": 2,
            "requested_budget": 2800,
            "requested_location": "Bayawan City",
            "requested_accommodation_type": "hotel",
            "room_price_per_night": 2400,
            "room_capacity": 3,
            "room_available": 2,
            "accom_location": "Bayawan City",
            "company_type": "hotel",
            "nights_requested": 2,
            "cnn_confidence": 0.92,
            "shown_rank": 1,
        },
        {
            "name": "budget_mismatch",
            "requested_guests": 2,
            "requested_budget": 1200,
            "requested_location": "Poblacion",
            "requested_accommodation_type": "inn",
            "room_price_per_night": 2600,
            "room_capacity": 2,
            "room_available": 1,
            "accom_location": "Poblacion",
            "company_type": "inn",
            "nights_requested": 1,
            "cnn_confidence": 0.88,
            "shown_rank": 2,
        },
        {
            "name": "incomplete_input_defaults",
            "requested_guests": "",
            "requested_budget": "",
            "requested_location": "Bayawan City",
            "requested_accommodation_type": "",
            "room_price_per_night": 1700,
            "room_capacity": 2,
            "room_available": 1,
            "accom_location": "Banga",
            "company_type": "hotel",
            "nights_requested": "",
            "cnn_confidence": "",
            "shown_rank": "",
        },
    ]

    results = []
    for sample in samples:
        payload = dict(sample)
        name = payload.pop("name")
        pred = predict_accommodation_relevance_from_features(payload)
        results.append({"sample": name, "input": payload, "prediction": pred})

    out_path = Path("artifacts/decision_tree_final/runtime_demo_results_v1.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({"output_json": str(out_path), "results": results}, indent=2))


if __name__ == "__main__":
    main()
