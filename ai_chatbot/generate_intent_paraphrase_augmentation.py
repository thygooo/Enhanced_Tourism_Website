import argparse
import json
import re
from pathlib import Path

import pandas as pd


def _normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _synonym_rewrite(text: str) -> str:
    replacements = [
        (r"\bbook\b", "reserve"),
        (r"\breserve\b", "book"),
        (r"\brecommend\b", "suggest"),
        (r"\bsuggest\b", "recommend"),
        (r"\bcalculate\b", "compute"),
        (r"\bcompute\b", "calculate"),
        (r"\bbill\b", "billing total"),
        (r"\bhotel\b", "hotel or inn"),
        (r"\btourist spots\b", "tour destinations"),
        (r"\btour package\b", "tour plan"),
        (r"\boperating hours\b", "opening hours"),
    ]
    out = str(text)
    for pat, repl in replacements:
        out = re.sub(pat, repl, out, flags=re.IGNORECASE)
    return _normalize_spaces(out)


def _intent_prefix(intent: str) -> str:
    mapping = {
        "book_accommodation": "Please process this room reservation request:",
        "calculate_accommodation_billing": "Please compute accommodation billing:",
        "calculate_billing": "Please compute tour billing:",
        "get_accommodation_recommendation": "Please suggest accommodation options:",
        "get_recommendation": "Please suggest tourism recommendations:",
        "get_tourism_information": "Please provide tourism information:",
    }
    return mapping.get(intent, "Please assist with this request:")


def _create_paraphrases(intent: str, text: str) -> list[str]:
    base = _normalize_spaces(str(text))
    v1 = _synonym_rewrite(base)
    v2 = f"{_intent_prefix(intent)} {base}"
    v3 = f"I need help with this: {v1}"
    candidates = [base, v1, v2, v3]
    unique = []
    seen = set()
    for c in candidates:
        k = c.strip().lower()
        if k and k not in seen:
            seen.add(k)
            unique.append(c.strip())
    return unique[1:3] if len(unique) >= 3 else unique[1:]


def main():
    parser = argparse.ArgumentParser(
        description="Generate realistic paraphrase augmentation rows from failed manual intent queries."
    )
    parser.add_argument(
        "--input-csv",
        default="artifacts/text_cnn_intent/manual_validation/error_analysis_rows_v1.csv",
    )
    parser.add_argument(
        "--out-csv",
        default="thesis_data_templates/text_cnn_intent_realistic_paraphrase_augmentation_v1.csv",
    )
    parser.add_argument(
        "--summary-json",
        default="artifacts/text_cnn_intent/manual_validation/paraphrase_augmentation_summary_v1.json",
    )
    args = parser.parse_args()

    in_path = Path(args.input_csv)
    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    df = pd.read_csv(in_path)
    required = {"query_id", "expected_intent", "input_text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    rows = []
    counter = 1
    for _, row in df.iterrows():
        source_id = str(row["query_id"]).strip()
        intent = str(row["expected_intent"]).strip()
        text = str(row["input_text"]).strip()
        if not source_id or not intent or not text:
            continue
        paraphrases = _create_paraphrases(intent, text)
        for p in paraphrases:
            rows.append(
                {
                    "aug_id": f"AUG_{counter:04d}",
                    "source_query_id": source_id,
                    "label_intent": intent,
                    "message_text": p,
                    "source": "manual_failure_paraphrase_v1",
                }
            )
            counter += 1

    out_df = pd.DataFrame(rows).drop_duplicates(
        subset=["label_intent", "message_text"]
    ).reset_index(drop=True)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False, encoding="utf-8")

    summary = {
        "input_error_rows": int(len(df)),
        "generated_rows": int(len(out_df)),
        "distribution_by_intent": {
            str(k): int(v) for k, v in out_df["label_intent"].value_counts().to_dict().items()
        },
        "outputs": {
            "augmentation_csv": str(out_path),
            "summary_json": str(args.summary_json),
        },
    }
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
