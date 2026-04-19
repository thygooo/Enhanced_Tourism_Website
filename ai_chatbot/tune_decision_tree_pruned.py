import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import ParameterGrid, RepeatedStratifiedKFold

from ai_chatbot.train_decision_tree_final import (
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    TARGET_COLUMN,
    audit_dataset,
    build_pipeline,
    load_dataset,
)


def _metric_bundle(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def _summary(values: list[float]) -> dict:
    s = pd.Series(values, dtype="float64")
    return {
        "mean": float(s.mean()),
        "std": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
        "min": float(s.min()),
        "max": float(s.max()),
    }


def _evaluate_config(x, y, params: dict, *, n_splits: int, n_repeats: int, random_state: int) -> dict:
    cv = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
    )
    train_metrics = {"accuracy": [], "precision_macro": [], "recall_macro": [], "f1_macro": []}
    val_metrics = {"accuracy": [], "precision_macro": [], "recall_macro": [], "f1_macro": []}

    for train_idx, val_idx in cv.split(x, y):
        x_train = x.iloc[train_idx]
        y_train = y.iloc[train_idx]
        x_val = x.iloc[val_idx]
        y_val = y.iloc[val_idx]

        pipe = build_pipeline(
            criterion=params["criterion"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            min_samples_split=params["min_samples_split"],
            max_features=params["max_features"],
        )
        pipe.fit(x_train, y_train)

        tr = _metric_bundle(y_train, pipe.predict(x_train))
        va = _metric_bundle(y_val, pipe.predict(x_val))
        for k in train_metrics:
            train_metrics[k].append(tr[k])
            val_metrics[k].append(va[k])

    train_summary = {m: _summary(v) for m, v in train_metrics.items()}
    val_summary = {m: _summary(v) for m, v in val_metrics.items()}
    gap = {
        m: float(train_summary[m]["mean"] - val_summary[m]["mean"])
        for m in train_summary.keys()
    }

    # Higher is better: reward validation quality and consistency, penalize overfitting.
    score = (
        val_summary["f1_macro"]["mean"]
        - (0.45 * gap["f1_macro"])
        - (0.20 * val_summary["f1_macro"]["std"])
    )

    return {
        "params": params,
        "train": train_summary,
        "validation": val_summary,
        "gap": gap,
        "selection_score": float(score),
    }


def _markdown_report(payload: dict) -> str:
    baseline = payload["baseline_current_model"]
    best = payload["recommended_pruned_model"]
    comparison = payload["comparison"]
    reason = payload["recommendation_reasoning"]

    def row(model_block, key):
        return model_block["validation"][key]["mean"], model_block["validation"][key]["std"]

    lines = []
    lines.append("# Decision Tree Pruning Search (Defense Profile)")
    lines.append("")
    lines.append("## Baseline vs Recommended Pruned Model")
    lines.append("| Metric | Baseline (mean+/-std) | Recommended Pruned (mean+/-std) |")
    lines.append("|---|---:|---:|")
    for metric in ("accuracy", "precision_macro", "recall_macro", "f1_macro"):
        b_mean, b_std = row(baseline, metric)
        p_mean, p_std = row(best, metric)
        lines.append(f"| {metric} | {b_mean:.4f} +/- {b_std:.4f} | {p_mean:.4f} +/- {p_std:.4f} |")
    lines.append(f"| train-validation gap (f1_macro) | {baseline['gap']['f1_macro']:.4f} | {best['gap']['f1_macro']:.4f} |")
    lines.append("")
    lines.append("## Recommended Final Parameters")
    lines.append("```json")
    lines.append(json.dumps(best["params"], indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Recommendation")
    lines.append(f"- Keep baseline model: **{comparison['keep_baseline']}**")
    lines.append(f"- Switch to pruned model: **{comparison['switch_to_pruned']}**")
    lines.append(f"- Reason: {reason}")
    lines.append("")
    lines.append("## Thesis Alignment")
    lines.append("- CNN remains intent classification.")
    lines.append("- Decision Tree remains recommendation/refinement.")
    lines.append("- Backend/database logic remains unchanged.")
    lines.append("- Gemini remains phrasing-only.")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Find a lower-overfitting Decision Tree config using repeated stratified CV.")
    parser.add_argument("--csv", default="thesis_data_templates/accommodation_reco_training_final_v1.csv")
    parser.add_argument("--out-dir", default="thesis_data_templates")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-repeats", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    df = load_dataset(Path(args.csv))
    x = df[NUMERIC_COLUMNS + CATEGORICAL_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    baseline_params = {
        "criterion": "entropy",
        "max_depth": None,
        "min_samples_leaf": 3,
        "min_samples_split": 2,
        "max_features": None,
    }
    baseline_eval = _evaluate_config(
        x,
        y,
        baseline_params,
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        random_state=args.random_state,
    )

    # Pruning-oriented compact grid for practical repeated-CV runtime.
    grid = {
        "criterion": ["entropy"],
        "max_depth": [8, 10, 12],
        "min_samples_leaf": [5, 8, 12, 16],
        "min_samples_split": [10, 20, 30],
        "max_features": [None],
    }

    results = []
    for params in ParameterGrid(grid):
        evaluated = _evaluate_config(
            x,
            y,
            params,
            n_splits=args.n_splits,
            n_repeats=args.n_repeats,
            random_state=args.random_state,
        )
        results.append(evaluated)

    rows = []
    for item in results:
        rows.append(
            {
                "criterion": item["params"]["criterion"],
                "max_depth": item["params"]["max_depth"],
                "min_samples_leaf": item["params"]["min_samples_leaf"],
                "min_samples_split": item["params"]["min_samples_split"],
                "max_features": item["params"]["max_features"],
                "val_accuracy_mean": item["validation"]["accuracy"]["mean"],
                "val_accuracy_std": item["validation"]["accuracy"]["std"],
                "val_precision_macro_mean": item["validation"]["precision_macro"]["mean"],
                "val_precision_macro_std": item["validation"]["precision_macro"]["std"],
                "val_recall_macro_mean": item["validation"]["recall_macro"]["mean"],
                "val_recall_macro_std": item["validation"]["recall_macro"]["std"],
                "val_f1_macro_mean": item["validation"]["f1_macro"]["mean"],
                "val_f1_macro_std": item["validation"]["f1_macro"]["std"],
                "train_f1_macro_mean": item["train"]["f1_macro"]["mean"],
                "f1_gap_mean": item["gap"]["f1_macro"],
                "selection_score": item["selection_score"],
            }
        )
    ranking_df = pd.DataFrame(rows).sort_values("selection_score", ascending=False).reset_index(drop=True)

    best_row = ranking_df.iloc[0].to_dict()
    best_params = {
        "criterion": best_row["criterion"],
        "max_depth": None if pd.isna(best_row["max_depth"]) else int(best_row["max_depth"]),
        "min_samples_leaf": int(best_row["min_samples_leaf"]),
        "min_samples_split": int(best_row["min_samples_split"]),
        "max_features": None if pd.isna(best_row["max_features"]) else str(best_row["max_features"]),
    }
    best_eval = None
    for item in results:
        if item["params"] == best_params:
            best_eval = item
            break
    if best_eval is None:
        raise RuntimeError("Unable to locate best evaluated configuration.")

    baseline_f1 = baseline_eval["validation"]["f1_macro"]["mean"]
    baseline_gap = baseline_eval["gap"]["f1_macro"]
    best_f1 = best_eval["validation"]["f1_macro"]["mean"]
    best_gap = best_eval["gap"]["f1_macro"]

    acceptable_tradeoff = (best_f1 >= (baseline_f1 - 0.015)) and (best_gap <= (baseline_gap - 0.02))
    switch_to_pruned = bool(acceptable_tradeoff)
    keep_baseline = not switch_to_pruned

    if switch_to_pruned:
        reason = (
            "Pruned model reduces overfitting gap with only small validation-F1 tradeoff, "
            "making it safer for defense generalization claims."
        )
    else:
        reason = (
            "Pruned candidates lowered overfitting but caused too much validation-F1 loss; "
            "baseline remains the best balance for now."
        )

    out = {
        "dataset_csv": str(args.csv),
        "rows": int(len(df)),
        "data_quality_audit": audit_dataset(df),
        "search_space_size": int(len(ranking_df)),
        "cv": {
            "strategy": "RepeatedStratifiedKFold",
            "n_splits": int(args.n_splits),
            "n_repeats": int(args.n_repeats),
            "total_runs_per_config": int(args.n_splits * args.n_repeats),
        },
        "baseline_current_model": baseline_eval,
        "recommended_pruned_model": best_eval,
        "comparison": {
            "baseline_val_f1_macro_mean": float(baseline_f1),
            "pruned_val_f1_macro_mean": float(best_f1),
            "baseline_f1_gap_mean": float(baseline_gap),
            "pruned_f1_gap_mean": float(best_gap),
            "delta_val_f1_macro_mean": float(best_f1 - baseline_f1),
            "delta_f1_gap_mean": float(best_gap - baseline_gap),
            "keep_baseline": keep_baseline,
            "switch_to_pruned": switch_to_pruned,
        },
        "recommendation_reasoning": reason,
        "thesis_alignment": {
            "cnn": "Intent classification only (unchanged).",
            "decision_tree": "Recommendation/refinement only (retained).",
            "backend_database": "Operational truth and retrieval remain unchanged.",
            "gemini": "Phrasing/clarification only (unchanged).",
        },
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "decision_tree_pruning_comparison_v1.csv"
    json_path = out_dir / "decision_tree_pruning_summary_v1.json"
    md_path = out_dir / "decision_tree_pruning_summary_v1.md"

    ranking_df.to_csv(csv_path, index=False, encoding="utf-8")
    json_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    md_path.write_text(_markdown_report(out), encoding="utf-8")

    print(
        json.dumps(
            {
                "comparison_csv": str(csv_path),
                "summary_json": str(json_path),
                "summary_md": str(md_path),
                "baseline_val_f1_macro_mean": baseline_f1,
                "pruned_val_f1_macro_mean": best_f1,
                "baseline_f1_gap_mean": baseline_gap,
                "pruned_f1_gap_mean": best_gap,
                "switch_to_pruned": switch_to_pruned,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
