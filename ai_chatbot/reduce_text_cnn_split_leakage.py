import argparse
import json
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd


def normalize(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


def find_pairs(df: pd.DataFrame, threshold: float) -> list[tuple[float, int, int]]:
    pairs: list[tuple[float, int, int]] = []
    for _intent, g in df.groupby("label_intent"):
        rows = g.reset_index().to_dict("records")
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                if rows[i]["split"] == rows[j]["split"]:
                    continue
                a = normalize(rows[i]["message_text"])
                b = normalize(rows[j]["message_text"])
                sim = SequenceMatcher(None, a, b).ratio()
                if sim >= threshold:
                    pairs.append((sim, int(rows[i]["index"]), int(rows[j]["index"])))
    pairs.sort(reverse=True, key=lambda x: x[0])
    return pairs


def resolve_leakage(df: pd.DataFrame, threshold: float, max_iter: int) -> tuple[pd.DataFrame, dict[str, int]]:
    out = df.copy().reset_index(drop=True)
    stats = {"pairs_initial": 0, "pairs_final": 0, "rows_moved": 0, "iterations": 0}

    for itr in range(max_iter):
        pairs = find_pairs(out, threshold)
        if itr == 0:
            stats["pairs_initial"] = len(pairs)
        if not pairs:
            stats["iterations"] = itr + 1
            break

        moved_this_iter = 0
        for _sim, ia, ib in pairs:
            if out.at[ia, "split"] == out.at[ib, "split"]:
                continue
            a_split = out.at[ia, "split"]
            b_split = out.at[ib, "split"]
            if "train" in (a_split, b_split):
                if a_split != "train":
                    out.at[ia, "split"] = "train"
                    moved_this_iter += 1
                elif b_split != "train":
                    out.at[ib, "split"] = "train"
                    moved_this_iter += 1
            else:
                # val-test conflict: move to val to reduce test leakage
                if b_split == "test":
                    out.at[ib, "split"] = "val"
                    moved_this_iter += 1
                else:
                    out.at[ia, "split"] = "val"
                    moved_this_iter += 1

        stats["rows_moved"] += moved_this_iter
        stats["iterations"] = itr + 1
        if moved_this_iter == 0:
            break

    stats["pairs_final"] = len(find_pairs(out, threshold))
    return out, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Reduce train/val/test leakage by reassigning highly similar rows.")
    parser.add_argument("--input-csv", default="thesis_data_templates/text_cnn_messages_final_expanded_v3.csv")
    parser.add_argument("--output-csv", default="thesis_data_templates/text_cnn_messages_final_expanded_v3_clean.csv")
    parser.add_argument(
        "--report-json",
        default="thesis_data_templates/text_cnn_messages_final_expanded_v3_clean_report.json",
    )
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--max-iter", type=int, default=4)
    args = parser.parse_args()

    in_path = Path(args.input_csv)
    out_path = Path(args.output_csv)
    report_path = Path(args.report_json)

    df = pd.read_csv(in_path)
    for col in ("message_text", "label_intent", "split"):
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    before_split = df.groupby(["label_intent", "split"]).size().unstack(fill_value=0)
    cleaned, stats = resolve_leakage(df, threshold=args.threshold, max_iter=args.max_iter)
    after_split = cleaned.groupby(["label_intent", "split"]).size().unstack(fill_value=0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(out_path, index=False)

    report = {
        "input_csv": str(in_path),
        "output_csv": str(out_path),
        "threshold": float(args.threshold),
        "stats": {k: int(v) for k, v in stats.items()},
        "before_split_counts": {
            intent: {split: int(before_split.loc[intent, split]) for split in before_split.columns}
            for intent in before_split.index
        },
        "after_split_counts": {
            intent: {split: int(after_split.loc[intent, split]) for split in after_split.columns}
            for intent in after_split.index
        },
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Saved cleaned dataset: {out_path}")
    print(f"Saved report: {report_path}")
    print("Before split counts:")
    print(before_split.to_string())
    print("\nAfter split counts:")
    print(after_split.to_string())
    print(f"\nLeakage stats: {stats}")


if __name__ == "__main__":
    main()
