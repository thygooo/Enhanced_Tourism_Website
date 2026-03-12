import argparse
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
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

DATE_PAIRS = [
    ("2026-03-20", "2026-03-22"),
    ("2026-03-25", "2026-03-27"),
    ("2026-03-28", "2026-03-30"),
    ("2026-04-02", "2026-04-04"),
    ("2026-04-10", "2026-04-12"),
    ("2026-04-17", "2026-04-19"),
    ("2026-04-24", "2026-04-26"),
    ("2026-05-01", "2026-05-03"),
]
ROOM_IDS = [2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 18, 22, 27, 31, 35, 40, 48, 55, 59, 64, 68, 72]
BUDGETS = [700, 900, 1200, 1500, 1800, 2200, 2500, 3000, 3500, 4000]
GUESTS = [1, 2, 3, 4, 5, 6]
NIGHTS = [1, 2, 3, 4, 5]
LOCATIONS = [
    "bayawan",
    "poblacion",
    "kalumboyan",
    "nangka",
    "villareal",
    "terminal area",
    "city proper",
    "near downtown",
    "near boulevard",
]
AMENITIES = ["wifi", "aircon", "parking", "hot shower", "breakfast", "quiet place", "near terminal"]
TOUR_TYPES = ["nature", "waterfall", "heritage", "city", "family", "sunset", "adventure", "farm"]

SHORT_QUERIES = {
    "book_accommodation": [
        "book room",
        "reserve na",
        "book ko ito",
        "pa reserve",
        "book room please",
        "pwede book now",
    ],
    "calculate_accommodation_billing": [
        "hm room total",
        "bill sa room?",
        "compute room cost",
        "how much stay",
        "room total pls",
    ],
    "calculate_billing": [
        "tour total hm",
        "tour bill pls",
        "how much tour",
        "compute tour fee",
        "tour cost?",
    ],
    "get_accommodation_recommendation": [
        "hotel reco pls",
        "inn suggestion?",
        "need stay options",
        "cheap room reco",
        "pa suggest hotel",
    ],
    "get_recommendation": [
        "tour reco pls",
        "suggest tour",
        "best tour?",
        "trip idea",
        "san maganda puntahan",
    ],
    "get_tourism_information": [
        "tourism info pls",
        "spot details?",
        "open hours?",
        "ano tourist spots",
        "info sa bayawan",
    ],
}

TYPO_MAP = {
    "recommend": "recomend",
    "accommodation": "accomodation",
    "reservation": "resrvation",
    "calculate": "calcuate",
    "tourism": "tourisn",
    "information": "infromation",
    "please": "pls",
    "for": "4",
    "you": "u",
    "today": "2day",
}


@dataclass
class Sample:
    text: str
    intent: str


def normalize_text(text: str) -> str:
    t = str(text or "").strip().lower()
    t = re.sub(r"\s+", " ", t)
    return t


def abstract_text(text: str) -> str:
    t = normalize_text(text)
    t = re.sub(r"\d{4}-\d{2}-\d{2}", " <date> ", t)
    t = re.sub(r"\b\d+\b", " <num> ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def token_jaccard(a: str, b: str) -> float:
    sa = set(re.findall(r"[a-z0-9<>]+", a.lower()))
    sb = set(re.findall(r"[a-z0-9<>]+", b.lower()))
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def similar_enough(a: str, b: str, *, threshold: float = 0.92) -> bool:
    na = normalize_text(a)
    nb = normalize_text(b)
    if na == nb:
        return True
    ar = abstract_text(na)
    br = abstract_text(nb)
    ratio = SequenceMatcher(None, na, nb).ratio()
    jacc = token_jaccard(ar, br)
    if ratio >= threshold:
        return True
    if ar == br and ratio >= 0.82:
        return True
    if jacc >= 0.9 and ratio >= 0.8:
        return True
    return False


def maybe_typo(text: str, rng: random.Random) -> str:
    out = text
    if rng.random() < 0.28:
        for k, v in TYPO_MAP.items():
            if k in out and rng.random() < 0.35:
                out = out.replace(k, v, 1)
    if rng.random() < 0.12:
        out = out.replace("?", "")
    if rng.random() < 0.1:
        out = out.replace("please", "pls")
    return out


def slots(rng: random.Random) -> dict[str, object]:
    check_in, check_out = rng.choice(DATE_PAIRS)
    am1, am2 = rng.sample(AMENITIES, 2)
    return {
        "room": rng.choice(ROOM_IDS),
        "guests": rng.choice(GUESTS),
        "nights": rng.choice(NIGHTS),
        "budget": rng.choice(BUDGETS),
        "location": rng.choice(LOCATIONS),
        "amenity": rng.choice(AMENITIES),
        "amenity_pair": f"{am1} and {am2}",
        "tour_type": rng.choice(TOUR_TYPES),
        "check_in": check_in,
        "check_out": check_out,
    }


def templates() -> dict[str, list[str]]:
    return {
        "book_accommodation": [
            "book room {room} for {guests} guests from {check_in} to {check_out}",
            "pwede pa reserve room {room} from {check_in} to {check_out}",
            "reserve an inn sa {location} for {guests} pax",
            "i need to book a place near {location} this weekend",
            "book na room na may {amenity}",
            "can you reserve for {nights} nights, {guests} guests",
            "book room quick lang near {location}",
            "pa book po ng room from {check_in} to {check_out}",
            "confirm booking ko for room {room}",
            "book accommodation with {amenity_pair}",
            "reserve room now budget around {budget}",
            "need room tonight pls, {guests} kami",
        ],
        "calculate_accommodation_billing": [
            "calculate room bill for {nights} nights",
            "how much room {room} from {check_in} to {check_out}",
            "hm total if {guests} guests for {nights} nights",
            "pa compute ng room payment with {amenity}",
            "what is my accommodation total around budget {budget}",
            "room billing breakdown please",
            "compute lodging cost sa {location}",
            "total stay payment from {check_in} to {check_out}",
            "how much babayaran sa room {room}",
            "bill estimate for inn stay near {location}",
            "calculate accom bill pls",
            "room charge hm with extra guest",
        ],
        "calculate_billing": [
            "calculate tour bill for {guests} guests",
            "hm tour package for {tour_type} trip",
            "pa compute tour cost budget {budget}",
            "tour fee estimate for {guests} pax",
            "what is total sa {tour_type} itinerary",
            "compute excursion payment",
            "tour billing breakdown please",
            "how much if family tour {nights} days",
            "tour total payment near {location}",
            "need tour quote under {budget}",
            "tour bill po for weekend",
            "cost of guided tour hm",
        ],
        "get_accommodation_recommendation": [
            "recommend inn in {location} for {guests} guests under {budget}",
            "hotel suggestion please with {amenity}",
            "pa suggest place to stay near {location}",
            "need accom options with {amenity_pair}",
            "san okay mag stay for {nights} nights",
            "best budget room around {budget}",
            "suggest hotel or inn from {check_in} to {check_out}",
            "looking for rooms near {location}",
            "any affordable accommodation for {guests} pax",
            "find me a quiet room with {amenity}",
            "inn reco near terminal please",
            "room options na cheap lang",
        ],
        "get_recommendation": [
            "recommend a {tour_type} tour in bayawan",
            "tour reco for {guests} guests budget {budget}",
            "suggest trip idea near {location}",
            "san maganda puntahan this weekend",
            "best {tour_type} activity for family",
            "give me quick tour suggestion",
            "recommend chill itinerary one day",
            "what tour bagay for friends",
            "need outdoor destination reco",
            "pa suggest scenic tour spots",
            "travel recommendation for short trip",
            "tour ideas na di mahal",
        ],
        "get_tourism_information": [
            "tourism info about {location}",
            "what are tourist spots in {location}",
            "open hours for attractions in bayawan",
            "details about {tour_type} destinations please",
            "ano entrance fee and schedule",
            "share information sa local attractions",
            "i need destination details before mag tour",
            "facts about tourism sites near {location}",
            "how to get to popular spots",
            "tourist spot background info pls",
            "where to check official attraction details",
            "list must-visit places in bayawan",
        ],
    }


def generate_text(intent: str, rng: random.Random) -> str:
    if rng.random() < 0.16:
        return rng.choice(SHORT_QUERIES[intent])
    s = slots(rng)
    text = rng.choice(templates()[intent]).format(**s)
    if rng.random() < 0.32:
        pre = rng.choice(
            [
                "hi, ",
                "hello ",
                "ask lang, ",
                "quick question: ",
                "uy ",
                "pls ",
                "",
            ]
        )
        text = f"{pre}{text}"
    if rng.random() < 0.26:
        tail = rng.choice([" pls", " thanks", " po", " now", "", "?"])
        text = f"{text}{tail}"
    text = maybe_typo(text.lower(), rng)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_seed_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    req = {"message_text", "label_intent"}
    if not req.issubset(df.columns):
        raise ValueError(f"Missing columns in seed dataset: {sorted(req - set(df.columns))}")
    out = df.copy()
    out["message_text"] = out["message_text"].fillna("").astype(str).str.strip().str.lower()
    out["label_intent"] = out["label_intent"].fillna("").astype(str).str.strip()
    out = out[(out["message_text"] != "") & (out["label_intent"].isin(INTENTS))]
    return out.reset_index(drop=True)


def build_intent_pool(seed_texts: list[str], intent: str, target: int, rng: random.Random) -> tuple[list[str], dict[str, int]]:
    accepted: list[str] = []
    stats = {"seed_added": 0, "gen_added": 0, "near_rejected": 0, "pattern_capped": 0}
    pattern_counter: Counter[str] = Counter()
    norm_seen: set[str] = set()

    def try_add(text: str, *, from_seed: bool) -> bool:
        t = normalize_text(text)
        if not t:
            return False
        if t in norm_seen:
            return False
        pat = abstract_text(t)
        if pattern_counter[pat] >= 5:
            stats["pattern_capped"] += 1
            return False
        for old in accepted:
            if similar_enough(t, old):
                stats["near_rejected"] += 1
                return False
        accepted.append(t)
        norm_seen.add(t)
        pattern_counter[pat] += 1
        if from_seed:
            stats["seed_added"] += 1
        else:
            stats["gen_added"] += 1
        return True

    for s in seed_texts:
        try_add(s, from_seed=True)

    guard = 0
    while len(accepted) < target and guard < 250000:
        guard += 1
        try_add(generate_text(intent, rng), from_seed=False)

    if len(accepted) < target:
        raise RuntimeError(f"Could not generate enough diverse rows for {intent}. Generated={len(accepted)} target={target}")

    rng.shuffle(accepted)
    return accepted[:target], stats


def assign_splits_for_intent(texts: list[str], rng: random.Random, train_ratio: float, val_ratio: float) -> list[str]:
    n = len(texts)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_test = n - n_train - n_val
    if n_test < 1:
        n_test = 1
        n_train = max(1, n_train - 1)
    labels = ["train"] * n_train + ["val"] * n_val + ["test"] * n_test
    rng.shuffle(labels)
    return labels


def leakage_pairs(df: pd.DataFrame, threshold: float) -> list[dict[str, object]]:
    rows = df.reset_index(drop=False).rename(columns={"index": "row_id"})
    findings: list[dict[str, object]] = []
    for intent, g in rows.groupby("label_intent"):
        data = g.to_dict("records")
        for i in range(len(data)):
            for j in range(i + 1, len(data)):
                if data[i]["split"] == data[j]["split"]:
                    continue
                sim = SequenceMatcher(None, normalize_text(data[i]["message_text"]), normalize_text(data[j]["message_text"])).ratio()
                if sim >= threshold:
                    findings.append(
                        {
                            "intent": intent,
                            "similarity": round(float(sim), 6),
                            "row_id_a": int(data[i]["row_id"]),
                            "row_id_b": int(data[j]["row_id"]),
                            "split_a": data[i]["split"],
                            "split_b": data[j]["split"],
                            "text_a": data[i]["message_text"],
                            "text_b": data[j]["message_text"],
                        }
                    )
    return findings


def build_v3_dataset(
    *,
    seed_csv: Path,
    output_csv: Path,
    report_json: Path,
    leakage_csv: Path,
    min_per_intent: int,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> None:
    rng = random.Random(seed)
    seed_df = load_seed_dataset(seed_csv)

    raw_before_counts = seed_df.groupby("label_intent").size().reindex(INTENTS, fill_value=0).to_dict()
    target_per_intent = max(min_per_intent, 300)

    final_rows: list[dict[str, str]] = []
    build_stats: dict[str, dict[str, int]] = {}

    for intent in INTENTS:
        intent_seed = seed_df[seed_df["label_intent"] == intent]["message_text"].tolist()
        texts, stats = build_intent_pool(intent_seed, intent, target_per_intent, rng)
        splits = assign_splits_for_intent(texts, rng, train_ratio=train_ratio, val_ratio=val_ratio)
        build_stats[intent] = stats
        for text, split in zip(texts, splits):
            final_rows.append({"message_text": text, "label_intent": intent, "split": split})

    final_df = pd.DataFrame(final_rows)
    final_df = final_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    leak = leakage_pairs(final_df, threshold=0.9)
    leakage_df = pd.DataFrame(leak)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_csv, index=False)
    leakage_df.to_csv(leakage_csv, index=False)

    after_counts = final_df.groupby("label_intent").size().reindex(INTENTS, fill_value=0).to_dict()
    split_counts = (
        final_df.groupby(["label_intent", "split"]).size().unstack(fill_value=0).reindex(INTENTS, fill_value=0)
    )

    report = {
        "seed_csv": str(seed_csv),
        "output_csv": str(output_csv),
        "raw_before_counts": {k: int(v) for k, v in raw_before_counts.items()},
        "after_clean_counts": {k: int(v) for k, v in after_counts.items()},
        "target_per_intent": int(target_per_intent),
        "total_rows": int(len(final_df)),
        "split_counts": {
            intent: {split: int(split_counts.loc[intent, split]) for split in split_counts.columns}
            for intent in split_counts.index
        },
        "leakage_threshold": 0.9,
        "leakage_pair_count": int(len(leak)),
        "build_stats": build_stats,
    }
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Saved dataset: {output_csv}")
    print(f"Saved leakage candidates: {leakage_csv}")
    print(f"Saved report: {report_json}")
    print(f"Total rows: {len(final_df)}")
    print(split_counts.to_string())
    print(f"Leakage candidate pairs (sim>=0.90 across splits): {len(leak)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and clean Text-CNN intent dataset v3.")
    parser.add_argument(
        "--seed-csv",
        default="thesis_data_templates/text_cnn_messages_final_refined.csv",
        help="Base seed CSV with message_text,label_intent",
    )
    parser.add_argument(
        "--output-csv",
        default="thesis_data_templates/text_cnn_messages_final_expanded_v3.csv",
        help="Output cleaned v3 CSV",
    )
    parser.add_argument(
        "--report-json",
        default="thesis_data_templates/text_cnn_messages_final_expanded_v3_report.json",
        help="Audit report JSON path",
    )
    parser.add_argument(
        "--leakage-csv",
        default="thesis_data_templates/text_cnn_messages_final_expanded_v3_leakage.csv",
        help="Cross-split leakage candidate CSV path",
    )
    parser.add_argument("--min-per-intent", type=int, default=320)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    build_v3_dataset(
        seed_csv=Path(args.seed_csv),
        output_csv=Path(args.output_csv),
        report_json=Path(args.report_json),
        leakage_csv=Path(args.leakage_csv),
        min_per_intent=args.min_per_intent,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
