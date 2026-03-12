import argparse
import random
from pathlib import Path

import pandas as pd


INTENTS = [
    "book_accommodation",
    "calculate_accommodation_billing",
    "calculate_billing",
    "get_accommodation_recommendation",
    "get_recommendation",
    "get_tourism_information",
]

SPLITS = ("train", "val", "test")

ROOM_IDS = [3, 4, 5, 6, 7, 8, 10, 12, 14, 18, 22, 27, 31, 35, 40, 48, 55, 59, 64, 68]
GUEST_COUNTS = [1, 2, 3, 4, 5, 6]
NIGHTS = [1, 2, 3, 4, 5]
BUDGETS = [900, 1200, 1500, 1800, 2200, 2500, 3000, 3500]
LOCATIONS = [
    "bayawan",
    "poblacion",
    "kalumboyan",
    "nangka",
    "villareal",
    "terminal area",
    "city proper",
    "near downtown",
]
AMENITIES = [
    "wifi",
    "aircon",
    "parking",
    "hot shower",
    "breakfast",
    "quiet area",
]
DATE_PAIRS = [
    ("2026-03-20", "2026-03-22"),
    ("2026-03-25", "2026-03-27"),
    ("2026-03-28", "2026-03-30"),
    ("2026-04-02", "2026-04-04"),
    ("2026-04-10", "2026-04-12"),
    ("2026-04-17", "2026-04-19"),
    ("2026-04-24", "2026-04-26"),
]
TOUR_TYPES = ["nature", "waterfall", "heritage", "family", "sunset", "city", "weekend", "half-day"]
TRAVEL_MOODS = ["relaxing", "adventure", "educational", "scenic", "budget-friendly", "cultural"]
GROUP_TYPES = ["solo", "couple", "family", "friends", "students", "senior guests"]
DURATIONS = ["half day", "one day", "two days", "weekend", "short trip"]


def _render(template: str, *, rng: random.Random) -> str:
    check_in, check_out = rng.choice(DATE_PAIRS)
    amenity1, amenity2 = rng.sample(AMENITIES, 2)
    values = {
        "room": rng.choice(ROOM_IDS),
        "guests": rng.choice(GUEST_COUNTS),
        "nights": rng.choice(NIGHTS),
        "budget": rng.choice(BUDGETS),
        "location": rng.choice(LOCATIONS),
        "amenity": rng.choice(AMENITIES),
        "amenity_pair": f"{amenity1} and {amenity2}",
        "check_in": check_in,
        "check_out": check_out,
        "tour_type": rng.choice(TOUR_TYPES),
        "travel_mood": rng.choice(TRAVEL_MOODS),
        "group_type": rng.choice(GROUP_TYPES),
        "duration": rng.choice(DURATIONS),
    }
    return template.format(**values).strip().lower()


def _intent_templates() -> dict[str, list[str]]:
    return {
        "book_accommodation": [
            "book room {room} for {guests} guests from {check_in} to {check_out}",
            "reserve room {room} for {nights} nights",
            "i want to book an inn in {location} for {guests} guests",
            "please confirm my booking for room {room}",
            "book an accommodation with {amenity} for tomorrow",
            "reserve a place to stay near {location} this weekend",
            "can you book room {room} with {amenity_pair}",
            "help me reserve an inn from {check_in} to {check_out}",
        ],
        "calculate_accommodation_billing": [
            "calculate accommodation bill for room {room} for {nights} nights",
            "how much is room {room} from {check_in} to {check_out}",
            "estimate hotel charges for {guests} guests for {nights} nights",
            "what is the total room payment if budget is around {budget}",
            "compute lodging cost with {amenity} included",
            "show my accommodation billing breakdown for {nights} nights",
            "please compute the room total for check in {check_in} check out {check_out}",
            "can you calculate the inn bill near {location}",
        ],
        "calculate_billing": [
            "calculate tour bill for {guests} guests",
            "how much is the {tour_type} tour package for {guests} pax",
            "compute tour payment for {nights} day trip",
            "please estimate tourism package cost under {budget}",
            "give me the total tour invoice for a {tour_type} trip",
            "tour billing for {guests} adults and budget {budget}",
            "what is the cost for a {tour_type} itinerary",
            "can you provide a tour price breakdown for {guests} people",
        ],
        "get_accommodation_recommendation": [
            "recommend an inn in {location} for {guests} guests under {budget}",
            "suggest accommodation with {amenity_pair} near {location}",
            "what hotel can you recommend for {guests} people",
            "find me a place to stay from {check_in} to {check_out}",
            "show affordable rooms below {budget} with {amenity}",
            "i need accommodation options around {location}",
            "give me room recommendations with budget {budget}",
            "best inn to stay for {nights} nights with {amenity}",
        ],
        "get_recommendation": [
            "recommend a {tour_type} tour in bayawan",
            "what tour can you suggest for {guests} guests",
            "suggest a tour package under {budget}",
            "i need a {tour_type} itinerary for the weekend",
            "best tourism activity near {location}",
            "recommend top places for a {tour_type} experience",
            "give me a tour suggestion for a short trip",
            "what is a good travel recommendation for family day out",
            "suggest a {travel_mood} {tour_type} trip for {group_type}",
            "recommend a {duration} itinerary near {location}",
            "what tour fits a {group_type} group with budget {budget}",
            "any {travel_mood} destination ideas for {duration}",
        ],
        "get_tourism_information": [
            "what are the tourist spots in {location}",
            "give me tourism information about {tour_type} attractions",
            "show details about local destinations near {location}",
            "what time do attractions open in bayawan",
            "tell me information about admission and schedules",
            "do you have background info for {tour_type} sites",
            "share tourism guide details for weekend visitors",
            "i need destination information before planning a tour",
            "can you share history and details of {tour_type} places in {location}",
            "what are the entrance fees and schedules for a {duration} visit",
            "tourism information for {group_type} exploring {location}",
            "where can i read official details about {tour_type} destinations",
        ],
    }


def _ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    expected = {"message_text", "label_intent", "split"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    out = df.copy()
    out["message_text"] = out["message_text"].fillna("").astype(str).str.strip().str.lower()
    out["label_intent"] = out["label_intent"].fillna("").astype(str).str.strip()
    out["split"] = out["split"].fillna("").astype(str).str.strip().str.lower()
    out = out[(out["message_text"] != "") & (out["label_intent"] != "") & (out["split"].isin(SPLITS))]
    return out


def expand_dataset(
    source_csv: Path,
    output_csv: Path,
    *,
    target_train: int,
    target_val: int,
    target_test: int,
    seed: int,
) -> None:
    rng = random.Random(seed)
    df = _ensure_schema(pd.read_csv(source_csv))
    templates = _intent_templates()

    intent_set = set(df["label_intent"].unique().tolist())
    missing_intents = set(INTENTS) - intent_set
    if missing_intents:
        raise ValueError(f"Source CSV missing expected intents: {', '.join(sorted(missing_intents))}")

    rows = df[["message_text", "label_intent", "split"]].to_dict("records")
    existing_texts = {r["message_text"] for r in rows}

    targets = {"train": target_train, "val": target_val, "test": target_test}

    for intent in INTENTS:
        for split in SPLITS:
            current = sum(1 for r in rows if r["label_intent"] == intent and r["split"] == split)
            needed = max(0, targets[split] - current)
            tries = 0
            while needed > 0 and tries < 20000:
                tries += 1
                candidate = _render(rng.choice(templates[intent]), rng=rng)
                if candidate in existing_texts:
                    continue
                rows.append({"message_text": candidate, "label_intent": intent, "split": split})
                existing_texts.add(candidate)
                needed -= 1
            if needed > 0:
                raise RuntimeError(f"Could not generate enough unique rows for {intent}/{split}. Remaining: {needed}")

    out = pd.DataFrame(rows).drop_duplicates(subset=["message_text"], keep="first")
    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    summary = out.groupby(["label_intent", "split"]).size().unstack(fill_value=0)
    print(f"Saved expanded dataset: {output_csv}")
    print(f"Total rows: {len(out)}")
    print(summary.to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description="Expand Text-CNN intent dataset to target per-intent split counts.")
    parser.add_argument(
        "--source-csv",
        type=str,
        default="thesis_data_templates/text_cnn_messages_final_refined.csv",
        help="Input CSV with message_text,label_intent,split",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="thesis_data_templates/text_cnn_messages_final_expanded_v2.csv",
        help="Output expanded CSV",
    )
    parser.add_argument("--target-train", type=int, default=80)
    parser.add_argument("--target-val", type=int, default=16)
    parser.add_argument("--target-test", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    expand_dataset(
        Path(args.source_csv),
        Path(args.output_csv),
        target_train=args.target_train,
        target_val=args.target_val,
        target_test=args.target_test,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
