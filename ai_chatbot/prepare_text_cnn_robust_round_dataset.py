import argparse
import json
import re
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


def normalize_text(text: str) -> str:
    text = str(text or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def synonym_rewrite(text: str) -> str:
    out = str(text)
    replacements = [
        (r"\bbook\b", "reserve"),
        (r"\breserve\b", "book"),
        (r"\brecommend\b", "suggest"),
        (r"\bsuggest\b", "recommend"),
        (r"\bcalculate\b", "compute"),
        (r"\bcompute\b", "calculate"),
        (r"\bbill\b", "billing total"),
        (r"\btourist spots\b", "tour destinations"),
        (r"\boperating hours\b", "opening hours"),
        (r"\bhotel\b", "hotel or inn"),
    ]
    for pattern, repl in replacements:
        out = re.sub(pattern, repl, out, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", out).strip()


def intent_prefix(intent: str) -> str:
    mapping = {
        "book_accommodation": "Please confirm this accommodation reservation:",
        "calculate_accommodation_billing": "Please compute accommodation charges:",
        "calculate_billing": "Please compute tour charges:",
        "get_accommodation_recommendation": "Please suggest hotel or inn options:",
        "get_recommendation": "Please suggest tour and attraction options:",
        "get_tourism_information": "Please provide tourism spot details:",
    }
    return mapping.get(intent, "Please assist with:")


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


def build_candidate_pool(manual_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in manual_df.iterrows():
        intent = str(row["expected_intent"]).strip()
        text = str(row["input_text"]).strip()
        qid = str(row["query_id"]).strip()
        if intent not in INTENTS or not text:
            continue
        candidates = [
            synonym_rewrite(text),
            f"{intent_prefix(intent)} {text}",
            f"I need help with this request: {synonym_rewrite(text)}",
        ]
        for c in candidates:
            c = c.strip()
            if not c:
                continue
            rows.append(
                {
                    "source_query_id": qid,
                    "label_intent": intent,
                    "message_text": c,
                    "source": "manual_query_balancing_v2",
                }
            )
    return pd.DataFrame(rows)


def balance_augmentation(aug_v1: pd.DataFrame, pool_df: pd.DataFrame, target_per_intent: int) -> pd.DataFrame:
    out_rows = []
    next_id = 1

    # Keep v1 first.
    for _, row in aug_v1.iterrows():
        intent = str(row["label_intent"]).strip()
        text = str(row["message_text"]).strip()
        if intent not in INTENTS or not text:
            continue
        out_rows.append(
            {
                "aug_id": f"AUGB_{next_id:05d}",
                "source_query_id": str(row.get("source_query_id", "")),
                "label_intent": intent,
                "message_text": text,
                "source": str(row.get("source", "manual_failure_paraphrase_v1")),
            }
        )
        next_id += 1

    out_df = pd.DataFrame(out_rows)

    # Deduplicate current base.
    out_df["norm"] = out_df["message_text"].apply(normalize_text)
    out_df = out_df.drop_duplicates(subset=["label_intent", "norm"]).reset_index(drop=True)

    for intent in INTENTS:
        current = int((out_df["label_intent"] == intent).sum())
        needed = max(0, target_per_intent - current)
        if needed == 0:
            continue

        candidates = pool_df[pool_df["label_intent"] == intent].copy()
        added = 0
        seen = set(out_df[out_df["label_intent"] == intent]["norm"].tolist())
        for _, cand in candidates.iterrows():
            text = str(cand["message_text"]).strip()
            norm = normalize_text(text)
            if not text or norm in seen:
                continue
            out_df = pd.concat(
                [
                    out_df,
                    pd.DataFrame(
                        [
                            {
                                "aug_id": f"AUGB_{next_id:05d}",
                                "source_query_id": str(cand.get("source_query_id", "")),
                                "label_intent": intent,
                                "message_text": text,
                                "source": str(cand.get("source", "manual_query_balancing_v2")),
                                "norm": norm,
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
            next_id += 1
            seen.add(norm)
            added += 1
            if added >= needed:
                break

    out_df = out_df.drop(columns=["norm"]).reset_index(drop=True)
    return out_df


def remove_cross_label_conflicts(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    out = df.copy()
    out["norm"] = out["message_text"].apply(normalize_text)
    conflict_norms = (
        out.groupby("norm")["label_intent"]
        .nunique()
        .reset_index(name="intent_count")
    )
    conflict_norms = set(conflict_norms[conflict_norms["intent_count"] > 1]["norm"].tolist())
    if not conflict_norms:
        out = out.drop(columns=["norm"])
        return out, 0

    # Keep only original base rows when conflict exists; drop augmentation conflicts.
    is_conflict = out["norm"].isin(conflict_norms)
    drop_mask = is_conflict & (out["source_tag"] == "augmentation")
    removed = int(drop_mask.sum())
    out = out[~drop_mask].copy()
    out = out.drop(columns=["norm"])
    return out, removed


def remove_near_duplicates_against_eval(df: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, int]:
    out = df.copy()
    out["norm"] = out["message_text"].apply(normalize_text)

    eval_rows = out[out["split"].isin(["val", "test"])].copy()
    train_aug = out[(out["split"] == "train") & (out["source_tag"] == "augmentation")].copy()
    to_drop = set()

    eval_by_intent = {
        intent: eval_rows[eval_rows["label_intent"] == intent]["norm"].tolist()
        for intent in INTENTS
    }
    for idx, row in train_aug.iterrows():
        intent = str(row["label_intent"])
        a = str(row["norm"])
        for b in eval_by_intent.get(intent, []):
            if SequenceMatcher(None, a, b).ratio() >= threshold:
                to_drop.add(idx)
                break

    removed = len(to_drop)
    if to_drop:
        out = out.drop(index=list(to_drop))
    out = out.drop(columns=["norm"]).reset_index(drop=True)
    return out, removed


def main():
    parser = argparse.ArgumentParser(description="Prepare robust-round Text-CNN dataset with balanced realistic paraphrases.")
    parser.add_argument("--base-csv", default="thesis_data_templates/text_cnn_messages_final_expanded_v3_clean.csv")
    parser.add_argument("--aug-v1-csv", default="thesis_data_templates/text_cnn_intent_realistic_paraphrase_augmentation_v1.csv")
    parser.add_argument("--manual-query-csv", default="thesis_data_templates/text_cnn_intent_manual_validation_queries_v1.csv")
    parser.add_argument("--target-per-intent", type=int, default=40)
    parser.add_argument("--near-dup-threshold", type=float, default=0.9)
    parser.add_argument("--out-aug-csv", default="thesis_data_templates/text_cnn_intent_realistic_paraphrase_augmentation_v2_balanced.csv")
    parser.add_argument("--out-merged-csv", default="thesis_data_templates/text_cnn_messages_final_expanded_v3_clean_robust_round_v1.csv")
    parser.add_argument("--report-json", default="thesis_data_templates/text_cnn_messages_final_expanded_v3_clean_robust_round_v1_report.json")
    args = parser.parse_args()

    base_df = pd.read_csv(args.base_csv)
    aug_v1_df = pd.read_csv(args.aug_v1_csv)
    manual_df = parse_manual_queries(Path(args.manual_query_csv))

    for col in ("message_text", "label_intent", "split"):
        if col not in base_df.columns:
            raise ValueError(f"Missing required base column: {col}")
    for col in ("message_text", "label_intent"):
        if col not in aug_v1_df.columns:
            raise ValueError(f"Missing required augmentation column: {col}")

    pool_df = build_candidate_pool(manual_df)
    aug_v2_df = balance_augmentation(
        aug_v1=aug_v1_df,
        pool_df=pool_df,
        target_per_intent=int(args.target_per_intent),
    )

    out_aug_path = Path(args.out_aug_csv)
    out_aug_path.parent.mkdir(parents=True, exist_ok=True)
    aug_v2_df.to_csv(out_aug_path, index=False, encoding="utf-8")

    merged_base = base_df.copy()
    merged_base["source_tag"] = "base"

    merged_aug = aug_v2_df[["message_text", "label_intent"]].copy()
    merged_aug["split"] = "train"
    merged_aug["source_tag"] = "augmentation"

    merged = pd.concat([merged_base, merged_aug], ignore_index=True)
    merged["message_text"] = merged["message_text"].fillna("").astype(str).str.strip()
    merged["label_intent"] = merged["label_intent"].fillna("").astype(str).str.strip()
    merged["split"] = merged["split"].fillna("").astype(str).str.strip().str.lower()
    merged = merged[(merged["message_text"] != "") & (merged["label_intent"] != "")]

    # Exact duplicate removal by intent + normalized text.
    merged["norm"] = merged["message_text"].apply(normalize_text)
    before_exact = len(merged)
    merged = merged.drop_duplicates(subset=["label_intent", "norm"], keep="first").reset_index(drop=True)
    exact_removed = before_exact - len(merged)
    merged = merged.drop(columns=["norm"])

    # Remove cross-label conflicting normalized texts from augmentation rows.
    merged, cross_label_removed = remove_cross_label_conflicts(merged)

    # Remove near duplicates between train augmentation and val/test to reduce leakage.
    merged, near_dup_removed = remove_near_duplicates_against_eval(
        merged,
        threshold=float(args.near_dup_threshold),
    )

    out_merged_path = Path(args.out_merged_csv)
    out_merged_path.parent.mkdir(parents=True, exist_ok=True)
    merged.drop(columns=["source_tag"]).to_csv(out_merged_path, index=False, encoding="utf-8")

    split_counts = (
        merged.groupby(["label_intent", "split"]).size().unstack(fill_value=0)
        if not merged.empty
        else pd.DataFrame()
    )
    aug_counts = aug_v2_df["label_intent"].value_counts().to_dict()

    report = {
        "base_csv": str(args.base_csv),
        "augmentation_v1_csv": str(args.aug_v1_csv),
        "manual_query_csv": str(args.manual_query_csv),
        "target_per_intent": int(args.target_per_intent),
        "augmentation_v2_rows": int(len(aug_v2_df)),
        "augmentation_v2_distribution": {str(k): int(v) for k, v in aug_counts.items()},
        "merged_rows": int(len(merged)),
        "exact_duplicates_removed": int(exact_removed),
        "cross_label_conflicts_removed": int(cross_label_removed),
        "near_duplicates_removed_vs_eval": int(near_dup_removed),
        "split_counts": {
            intent: {split: int(split_counts.loc[intent, split]) for split in split_counts.columns}
            for intent in split_counts.index
        },
        "outputs": {
            "augmentation_v2_csv": str(out_aug_path),
            "merged_dataset_csv": str(out_merged_path),
            "report_json": str(args.report_json),
        },
    }

    report_path = Path(args.report_json)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
