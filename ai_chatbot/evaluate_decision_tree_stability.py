import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold

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


def _aggregate(values: list[float]) -> dict:
    series = pd.Series(values, dtype="float64")
    return {
        "mean": float(series.mean()),
        "std": float(series.std(ddof=1)) if len(series) > 1 else 0.0,
        "min": float(series.min()),
        "max": float(series.max()),
    }


def _severity_for_imbalance(ratio: float) -> str:
    if ratio >= 3.0:
        return "severe"
    if ratio >= 1.8:
        return "moderate"
    return "low"


def _build_markdown(summary: dict) -> str:
    params = summary["model_params"]
    cv = summary["cross_validation"]
    checks = summary["stability_checks"]
    dq = summary["data_quality_audit"]
    cls = summary["class_balance"]

    lines = []
    lines.append("# Decision Tree Stability Evaluation (Defense Profile)")
    lines.append("")
    lines.append("## Configuration")
    lines.append(f"- Algorithm: `DecisionTreeClassifier`")
    lines.append(f"- Criterion: `{params['criterion']}`")
    lines.append(f"- Max depth: `{params['max_depth']}`")
    lines.append(f"- Min samples leaf: `{params['min_samples_leaf']}`")
    lines.append(f"- Min samples split: `{params['min_samples_split']}`")
    lines.append(f"- Max features: `{params['max_features']}`")
    lines.append(f"- CV strategy: `RepeatedStratifiedKFold` ({cv['n_splits']} folds x {cv['n_repeats']} repeats)")
    lines.append("")
    lines.append("## Validation Metrics (Validation Fold)")
    lines.append("| Metric | Mean | Std | Min | Max |")
    lines.append("|---|---:|---:|---:|---:|")
    for m in ("accuracy", "precision_macro", "recall_macro", "f1_macro"):
        row = cv["validation_metrics"][m]
        lines.append(f"| {m} | {row['mean']:.4f} | {row['std']:.4f} | {row['min']:.4f} | {row['max']:.4f} |")
    lines.append("")
    lines.append("## Overfitting Check (Train vs Validation)")
    lines.append("| Metric | Train Mean | Validation Mean | Gap |")
    lines.append("|---|---:|---:|---:|")
    for m in ("accuracy", "precision_macro", "recall_macro", "f1_macro"):
        tr = cv["train_metrics"][m]["mean"]
        va = cv["validation_metrics"][m]["mean"]
        gp = cv["generalization_gap"][m]["mean_gap"]
        lines.append(f"| {m} | {tr:.4f} | {va:.4f} | {gp:.4f} |")
    lines.append("")
    lines.append("## Stability / Risk Checks")
    lines.append(f"- Split instability flag: **{checks['split_instability_flag']}**")
    lines.append(f"- Overfitting concern flag: **{checks['overfitting_concern_flag']}**")
    lines.append(f"- Leakage risk flag: **{checks['leakage_risk_flag']}**")
    lines.append(f"- Class imbalance severity: **{cls['severity']}** (ratio={cls['ratio']:.4f})")
    lines.append("")
    lines.append("## Data Quality Checks")
    lines.append(f"- Rows: {dq['rows']}")
    lines.append(f"- Duplicate rows: {dq['duplicate_rows']}")
    lines.append(f"- Missing cells (features + target): {dq['missing_cells_total']}")
    lines.append(f"- Feature-label conflict rows: {dq['feature_label_conflict_rows']}")
    lines.append("")
    lines.append("## Thesis Alignment")
    lines.append("- CNN remains for intent classification.")
    lines.append("- Decision Tree remains the recommendation/refinement algorithm.")
    lines.append("- Backend/database logic remains unchanged.")
    lines.append("- Gemini remains phrasing-only.")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Evaluate Decision Tree stability using repeated stratified CV.")
    parser.add_argument("--csv", default="thesis_data_templates/accommodation_reco_training_final_v1.csv")
    parser.add_argument("--out-dir", default="artifacts/decision_tree_final")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-repeats", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--criterion", default="entropy", choices=["gini", "entropy", "log_loss"])
    parser.add_argument("--max-depth", type=int, default=0)
    parser.add_argument("--min-samples-leaf", type=int, default=3)
    parser.add_argument("--min-samples-split", type=int, default=2)
    parser.add_argument("--max-features", default="none", choices=["none", "sqrt", "log2"])
    args = parser.parse_args()

    df = load_dataset(Path(args.csv))
    x = df[NUMERIC_COLUMNS + CATEGORICAL_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    dq_audit = audit_dataset(df)

    model_params = {
        "algorithm": "DecisionTreeClassifier",
        "criterion": str(args.criterion),
        "max_depth": None if args.max_depth in (0, None) else int(args.max_depth),
        "min_samples_leaf": int(args.min_samples_leaf),
        "min_samples_split": int(args.min_samples_split),
        "max_features": None if str(args.max_features).lower() == "none" else str(args.max_features),
        "class_weight": "balanced",
        "random_state": int(args.random_state),
    }

    cv = RepeatedStratifiedKFold(
        n_splits=int(args.n_splits),
        n_repeats=int(args.n_repeats),
        random_state=int(args.random_state),
    )

    fold_rows: list[dict] = []
    train_metrics = {"accuracy": [], "precision_macro": [], "recall_macro": [], "f1_macro": []}
    val_metrics = {"accuracy": [], "precision_macro": [], "recall_macro": [], "f1_macro": []}

    for idx, (train_idx, val_idx) in enumerate(cv.split(x, y), start=1):
        repeat_index = ((idx - 1) // int(args.n_splits)) + 1
        fold_index = ((idx - 1) % int(args.n_splits)) + 1

        x_train = x.iloc[train_idx]
        y_train = y.iloc[train_idx]
        x_val = x.iloc[val_idx]
        y_val = y.iloc[val_idx]

        pipe = build_pipeline(
            criterion=model_params["criterion"],
            max_depth=model_params["max_depth"],
            min_samples_leaf=model_params["min_samples_leaf"],
            min_samples_split=model_params["min_samples_split"],
            max_features=model_params["max_features"],
        )
        pipe.fit(x_train, y_train)

        pred_train = pipe.predict(x_train)
        pred_val = pipe.predict(x_val)
        train_bundle = _metric_bundle(y_train, pred_train)
        val_bundle = _metric_bundle(y_val, pred_val)

        for key in train_metrics:
            train_metrics[key].append(train_bundle[key])
            val_metrics[key].append(val_bundle[key])

        fold_rows.append(
            {
                "run_index": idx,
                "repeat_index": repeat_index,
                "fold_index": fold_index,
                "train_size": int(len(train_idx)),
                "val_size": int(len(val_idx)),
                "train_accuracy": train_bundle["accuracy"],
                "train_precision_macro": train_bundle["precision_macro"],
                "train_recall_macro": train_bundle["recall_macro"],
                "train_f1_macro": train_bundle["f1_macro"],
                "val_accuracy": val_bundle["accuracy"],
                "val_precision_macro": val_bundle["precision_macro"],
                "val_recall_macro": val_bundle["recall_macro"],
                "val_f1_macro": val_bundle["f1_macro"],
                "gap_f1_macro": train_bundle["f1_macro"] - val_bundle["f1_macro"],
            }
        )

    train_summary = {metric: _aggregate(values) for metric, values in train_metrics.items()}
    val_summary = {metric: _aggregate(values) for metric, values in val_metrics.items()}
    gap_summary = {
        metric: {
            "mean_gap": float(train_summary[metric]["mean"] - val_summary[metric]["mean"]),
            "std_gap_proxy": float(val_summary[metric]["std"]),
        }
        for metric in train_summary
    }

    class_counts = y.value_counts()
    imbalance_ratio = float(class_counts.max() / max(1, class_counts.min()))
    class_balance = {
        "distribution": {k: int(v) for k, v in class_counts.to_dict().items()},
        "ratio": imbalance_ratio,
        "severity": _severity_for_imbalance(imbalance_ratio),
    }

    stability_checks = {
        "split_instability_flag": bool(val_summary["f1_macro"]["std"] > 0.03),
        "overfitting_concern_flag": bool(gap_summary["f1_macro"]["mean_gap"] > 0.05),
        "leakage_risk_flag": bool(
            (dq_audit.get("duplicate_rows", 0) > 0)
            or (dq_audit.get("feature_label_conflict_rows", 0) > 0)
        ),
        "notes": {
            "split_instability_rule": "Flag true if validation F1 std > 0.03.",
            "overfitting_rule": "Flag true if mean train-validation F1 gap > 0.05.",
            "leakage_rule": "Flag true if duplicate rows or conflicting labels for identical feature keys are detected.",
        },
    }

    summary = {
        "dataset_csv": str(args.csv),
        "rows": int(len(df)),
        "model_params": model_params,
        "cross_validation": {
            "strategy": "RepeatedStratifiedKFold",
            "n_splits": int(args.n_splits),
            "n_repeats": int(args.n_repeats),
            "total_runs": int(len(fold_rows)),
            "train_metrics": train_summary,
            "validation_metrics": val_summary,
            "generalization_gap": gap_summary,
        },
        "class_balance": class_balance,
        "data_quality_audit": dq_audit,
        "stability_checks": stability_checks,
        "thesis_alignment": {
            "cnn": "Intent classification only (unchanged).",
            "decision_tree": "Recommendation/refinement algorithm (retained).",
            "backend_database": "Operational logic and data retrieval (unchanged).",
            "gemini": "Phrasing/clarification only (unchanged).",
        },
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fold_csv_path = out_dir / "stability_cv_runs_v1.csv"
    summary_json_path = out_dir / "stability_summary_v1.json"
    summary_md_path = out_dir / "stability_summary_v1.md"

    pd.DataFrame(fold_rows).to_csv(fold_csv_path, index=False, encoding="utf-8")
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary_md_path.write_text(_build_markdown(summary), encoding="utf-8")

    print(
        json.dumps(
            {
                "fold_csv": str(fold_csv_path),
                "summary_json": str(summary_json_path),
                "summary_md": str(summary_md_path),
                "validation_f1_mean": summary["cross_validation"]["validation_metrics"]["f1_macro"]["mean"],
                "validation_f1_std": summary["cross_validation"]["validation_metrics"]["f1_macro"]["std"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
