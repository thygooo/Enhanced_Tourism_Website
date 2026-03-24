import argparse
import json
import random
from pathlib import Path

import pandas as pd


LOCATIONS = [
    "Bayawan City",
    "Poblacion",
    "Banga",
    "Villareal",
    "Suba",
    "Nangka",
]


def _weighted_choice(items):
    values, weights = zip(*items)
    return random.choices(values, weights=weights, k=1)[0]


def _pick_requested_type():
    return _weighted_choice(
        [
            ("hotel", 0.45),
            ("inn", 0.35),
            ("either", 0.20),
        ]
    )


def _pick_company_type(requested_type: str) -> str:
    if requested_type == "either":
        return _weighted_choice([("hotel", 0.55), ("inn", 0.45)])
    if random.random() < 0.75:
        return requested_type
    return "inn" if requested_type == "hotel" else "hotel"


def _pick_requested_budget(requested_type: str) -> float:
    if requested_type == "hotel":
        return float(random.randint(1600, 5200))
    if requested_type == "inn":
        return float(random.randint(700, 2800))
    return float(random.randint(1000, 4200))


def _pick_room_price(company_type: str, location_match: bool) -> float:
    if company_type == "hotel":
        base = random.randint(1400, 5600)
    else:
        base = random.randint(600, 3200)
    if location_match and random.random() < 0.25:
        base += random.randint(100, 600)
    return float(max(450, base))


def _pick_capacity(company_type: str) -> int:
    if company_type == "hotel":
        return random.randint(2, 6)
    return random.randint(1, 4)


def _calc_relevance_probability(row: dict) -> float:
    budget = float(row["requested_budget"])
    price = float(row["room_price_per_night"])
    guests = int(row["requested_guests"])
    capacity = int(row["room_capacity"])
    req_loc = str(row["requested_location"]).strip().lower()
    accom_loc = str(row["accom_location"]).strip().lower()
    req_type = str(row["requested_accommodation_type"]).strip().lower()
    company_type = str(row["company_type"]).strip().lower()
    available = int(row["room_available"])
    nights = int(row["nights_requested"])
    rank = int(row["shown_rank"])
    cnn_conf = float(row["cnn_confidence"])

    price_ratio = price / budget if budget > 0 else 1.2
    budget_fit = 1.0 if price_ratio <= 1.0 else 0.45 if price_ratio <= 1.15 else 0.05
    capacity_fit = 1.0 if guests <= capacity else 0.0
    location_fit = 1.0 if req_loc == accom_loc else 0.35
    type_fit = 1.0 if req_type == "either" or req_type == company_type else 0.2
    availability_fit = 1.0 if available >= 1 else 0.0
    nights_penalty = 0.0 if nights <= 3 else 0.08
    rank_bonus = max(0.0, 0.12 - (rank - 1) * 0.012)

    score = (
        (0.31 * budget_fit)
        + (0.24 * capacity_fit)
        + (0.15 * location_fit)
        + (0.14 * type_fit)
        + (0.10 * availability_fit)
        + (0.10 * min(1.0, max(0.0, cnn_conf)))
        + rank_bonus
        - nights_penalty
    )
    noise = random.uniform(-0.08, 0.08)
    return max(0.0, min(1.0, score + noise))


def generate_rows(total_rows: int) -> list[dict]:
    rows = []
    for _ in range(total_rows):
        requested_type = _pick_requested_type()
        company_type = _pick_company_type(requested_type)
        requested_location = random.choice(LOCATIONS)
        location_match = random.random() < 0.68
        accom_location = requested_location if location_match else random.choice(
            [loc for loc in LOCATIONS if loc != requested_location]
        )
        requested_guests = _weighted_choice([(1, 0.22), (2, 0.33), (3, 0.18), (4, 0.16), (5, 0.08), (6, 0.03)])
        room_capacity = _pick_capacity(company_type)
        requested_budget = _pick_requested_budget(requested_type)
        room_price = _pick_room_price(company_type, location_match)
        room_available = _weighted_choice([(0, 0.08), (1, 0.48), (2, 0.27), (3, 0.12), (4, 0.05)])
        nights_requested = _weighted_choice([(1, 0.40), (2, 0.30), (3, 0.17), (4, 0.09), (5, 0.04)])
        shown_rank = random.randint(1, 10)
        cnn_confidence = round(random.uniform(0.52, 0.99), 4)

        row = {
            "requested_guests": int(requested_guests),
            "requested_budget": round(float(requested_budget), 2),
            "requested_location": requested_location,
            "requested_accommodation_type": requested_type,
            "room_price_per_night": round(float(room_price), 2),
            "room_capacity": int(room_capacity),
            "room_available": int(room_available),
            "accom_location": accom_location,
            "company_type": company_type,
            "nights_requested": int(nights_requested),
            "cnn_confidence": float(cnn_confidence),
            "shown_rank": int(shown_rank),
        }
        p_relevant = _calc_relevance_probability(row)
        row["relevance_label"] = "relevant" if p_relevant >= 0.68 else "not_relevant"
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic accommodation recommendation dataset for Decision Tree training."
    )
    parser.add_argument("--rows", type=int, default=2400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out-csv",
        default="thesis_data_templates/accommodation_reco_training_final_v1.csv",
    )
    parser.add_argument(
        "--summary-json",
        default="artifacts/decision_tree_final/dataset_generation_summary_v1.json",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    rows = generate_rows(args.rows)
    df = pd.DataFrame(rows)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8")

    class_dist = df["relevance_label"].value_counts().to_dict()
    summary = {
        "seed": int(args.seed),
        "total_rows": int(len(df)),
        "class_distribution": {str(k): int(v) for k, v in class_dist.items()},
        "schema_version": "decision_tree_accommodation_schema_v1",
        "assumptions": [
            "Each row is one candidate accommodation item for one request context.",
            "Relevance is driven mainly by budget fit, capacity fit, location fit, type fit, and availability.",
            "CNN confidence and shown rank contribute as secondary ranking signals.",
            "Small random noise is added to avoid deterministic labels and support realistic variation.",
            "Location vocabulary is constrained to Bayawan-localized example areas for thesis simulation.",
        ],
        "outputs": {
            "dataset_csv": str(out_csv),
            "summary_json": str(args.summary_json),
        },
    }

    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
