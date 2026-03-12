import argparse
import json
import pickle
from pathlib import Path

import pandas as pd

try:
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.tree import DecisionTreeClassifier
except ImportError as exc:
    raise SystemExit(
        "scikit-learn is not installed. Run: pip install scikit-learn pandas\n"
        f"Original error: {exc}"
    )


REQUIRED_COLUMNS = {
    "requested_guests",
    "requested_budget",
    "requested_location",
    "requested_accommodation_type",
    "room_price_per_night",
    "room_capacity",
    "room_available",
    "accom_location",
    "company_type",
    "nights_requested",
    "cnn_confidence",
    "shown_rank",
    "relevance_label",
}


FEATURE_COLUMNS = [
    "requested_guests",
    "requested_budget",
    "requested_location",
    "requested_accommodation_type",
    "room_price_per_night",
    "room_capacity",
    "room_available",
    "accom_location",
    "company_type",
    "nights_requested",
    "cnn_confidence",
    "shown_rank",
]

NUMERIC_COLUMNS = [
    "requested_guests",
    "requested_budget",
    "room_price_per_night",
    "room_capacity",
    "room_available",
    "nights_requested",
    "cnn_confidence",
    "shown_rank",
]

CATEGORICAL_COLUMNS = [
    "requested_location",
    "requested_accommodation_type",
    "accom_location",
    "company_type",
]


def load_dataset(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip() for c in df.columns]
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    df = df.copy()
    df["relevance_label"] = df["relevance_label"].fillna("").astype(str).str.strip().str.lower()
    df = df[df["relevance_label"].isin({"relevant", "not_relevant"})]
    if df.empty:
        raise ValueError("Dataset is empty after filtering invalid relevance_label values.")

    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in CATEGORICAL_COLUMNS:
        df[col] = df[col].fillna("").astype(str).str.strip()

    return df


def build_pipeline(max_depth: int | None = None) -> Pipeline:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_COLUMNS),
            ("cat", categorical_pipe, CATEGORICAL_COLUMNS),
        ]
    )
    clf = DecisionTreeClassifier(
        random_state=42,
        max_depth=max_depth,
        min_samples_leaf=1,
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", clf),
        ]
    )


def main():
    parser = argparse.ArgumentParser(description="Train a pilot Decision Tree on sample accommodation recommendation data.")
    parser.add_argument(
        "--csv",
        type=str,
        default="thesis_data_templates/accommodation_reco_training.csv",
        help="Path to accommodation_reco_training.csv",
    )
    parser.add_argument("--target", type=str, default="relevance_label", choices=["relevance_label", "was_booked", "was_selected", "was_clicked"])
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--save-dir", type=str, default="artifacts/decision_tree_demo")
    parser.add_argument("--test-size", type=float, default=0.25)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    print(f"Loading dataset from: {csv_path}")
    df = load_dataset(csv_path)
    print("\nDataset summary")
    print("-" * 40)
    print(f"Total rows: {len(df)}")
    print("Target distribution:")
    if args.target == "relevance_label":
        y = df["relevance_label"].astype(str)
    else:
        df[args.target] = pd.to_numeric(df[args.target], errors="coerce").fillna(0).astype(int)
        y = df[args.target].astype(int).astype(str)
    print(y.value_counts().to_string())

    x = df[FEATURE_COLUMNS].copy()
    y_trainable = y

    stratify = y_trainable if y_trainable.nunique() > 1 and len(df) >= 4 else None
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y_trainable,
        test_size=args.test_size,
        random_state=42,
        stratify=stratify,
    )

    model = build_pipeline(max_depth=args.max_depth)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)

    print("\nEvaluation (pilot/demo)")
    print("-" * 40)
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification report")
    print(classification_report(y_test, y_pred, zero_division=0))

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model_path = save_dir / "decision_tree_demo.pkl"
    meta_path = save_dir / "metadata.json"

    with model_path.open("wb") as f:
        pickle.dump(model, f)

    metadata = {
        "feature_columns": FEATURE_COLUMNS,
        "numeric_columns": NUMERIC_COLUMNS,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "target": args.target,
        "rows": int(len(df)),
        "test_size": float(args.test_size),
        "max_depth": args.max_depth,
        "note": "Pilot/demo training only. Retrain with real system data for final thesis results.",
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Print basic feature importance summary if possible.
    clf = model.named_steps["model"]
    preprocess = model.named_steps["preprocess"]
    try:
        feature_names = preprocess.get_feature_names_out().tolist()
        importances = clf.feature_importances_.tolist()
        ranked = sorted(zip(feature_names, importances), key=lambda t: t[1], reverse=True)
        print("\nTop feature importances")
        print("-" * 40)
        for name, score in ranked[:10]:
            print(f"{name}: {score:.4f}")
    except Exception:
        pass

    print("\nSaved artifacts")
    print("-" * 40)
    print(f"Model:    {model_path}")
    print(f"Metadata: {meta_path}")


if __name__ == "__main__":
    main()

