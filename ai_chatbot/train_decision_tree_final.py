import argparse
import json
import pickle
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


TARGET_COLUMN = "relevance_label"
BASE_FEATURE_COLUMNS = [
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
ENGINEERED_NUMERIC_COLUMNS = [
    "budget_ratio",
    "budget_within",
    "budget_gap_pct",
    "capacity_gap",
    "capacity_fit",
    "availability_flag",
    "location_match",
    "type_match",
    "city_proper_match",
]
BASE_NUMERIC_COLUMNS = [
    "requested_guests",
    "requested_budget",
    "room_price_per_night",
    "room_capacity",
    "room_available",
    "nights_requested",
    "cnn_confidence",
    "shown_rank",
]
NUMERIC_COLUMNS = BASE_NUMERIC_COLUMNS + ENGINEERED_NUMERIC_COLUMNS
CATEGORICAL_COLUMNS = [
    "requested_location",
    "requested_accommodation_type",
    "accom_location",
    "company_type",
]
VALID_TARGETS = {"relevant", "not_relevant"}


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    eps = 1e-6
    out["budget_ratio"] = out["room_price_per_night"] / (out["requested_budget"] + eps)
    out["budget_within"] = (out["room_price_per_night"] <= out["requested_budget"]).astype(int)
    out["budget_gap_pct"] = (
        (out["requested_budget"] - out["room_price_per_night"]) / (out["requested_budget"] + eps)
    ).clip(-2, 2)
    out["capacity_gap"] = out["room_capacity"] - out["requested_guests"]
    out["capacity_fit"] = (out["capacity_gap"] >= 0).astype(int)
    out["availability_flag"] = (out["room_available"] > 0).astype(int)
    out["location_match"] = (
        out["requested_location"].astype(str).str.lower()
        == out["accom_location"].astype(str).str.lower()
    ).astype(int)

    req_type = out["requested_accommodation_type"].astype(str).str.lower()
    company_type = out["company_type"].astype(str).str.lower()
    out["type_match"] = ((req_type == "either") | (req_type == company_type)).astype(int)

    requested_city_proper = out["requested_location"].astype(str).str.lower().str.contains(
        r"poblacion|city proper",
        regex=True,
    )
    accom_city_proper = out["accom_location"].astype(str).str.lower().str.contains(
        r"poblacion|city proper",
        regex=True,
    )
    out["city_proper_match"] = (requested_city_proper & accom_city_proper).astype(int)
    return out


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    missing = set(BASE_FEATURE_COLUMNS + [TARGET_COLUMN]) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = df.copy()
    out[TARGET_COLUMN] = out[TARGET_COLUMN].fillna("").astype(str).str.strip().str.lower()
    out = out[out[TARGET_COLUMN].isin(VALID_TARGETS)]
    if out.empty:
        raise ValueError("No valid rows after target filtering.")

    for col in BASE_NUMERIC_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in CATEGORICAL_COLUMNS:
        out[col] = out[col].fillna("").astype(str).str.strip().str.lower()
    out = add_engineered_features(out)
    for col in ENGINEERED_NUMERIC_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def build_pipeline(
    *,
    criterion: str,
    max_depth: int | None,
    min_samples_leaf: int,
    min_samples_split: int,
    max_features: str | None,
) -> Pipeline:
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
    model = DecisionTreeClassifier(
        random_state=42,
        criterion=criterion,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        max_features=max_features,
        class_weight="balanced",
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )


def save_confusion_matrix_figure(cm, class_names, out_path: Path):
    n = len(class_names)
    cell = 120
    margin = 240
    width = margin + (n * cell) + 40
    height = margin + (n * cell) + 40
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    max_val = max(max(row) for row in cm) if n else 1
    if max_val <= 0:
        max_val = 1

    draw.text((20, 15), "Decision Tree Confusion Matrix", fill="black")
    draw.text((margin + (n * cell // 2) - 50, 45), "Predicted", fill="black")
    draw.text((20, margin - 30), "True", fill="black")

    for j, name in enumerate(class_names):
        x = margin + (j * cell) + 8
        y = margin - 45
        draw.text((x, y), str(name), fill="black")

    for i, name in enumerate(class_names):
        x = 20
        y = margin + (i * cell) + (cell // 2) - 8
        draw.text((x, y), str(name), fill="black")

    for i in range(n):
        for j in range(n):
            value = int(cm[i][j])
            ratio = float(value) / float(max_val)
            shade = int(255 - (ratio * 180))
            color = (shade, shade, 255)
            x0 = margin + (j * cell)
            y0 = margin + (i * cell)
            x1 = x0 + cell
            y1 = y0 + cell
            draw.rectangle([x0, y0, x1, y1], fill=color, outline="black", width=1)
            draw.text((x0 + (cell // 2) - 10, y0 + (cell // 2) - 8), str(value), fill="black")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="PNG")


def audit_dataset(df: pd.DataFrame) -> dict:
    feature_keys = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS
    label_conflicts = (
        df.groupby(feature_keys)[TARGET_COLUMN]
        .nunique()
        .rename("label_variants")
        .reset_index()
    )
    conflict_rows = int((label_conflicts["label_variants"] > 1).sum())

    return {
        "rows": int(len(df)),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_cells_total": int(df[feature_keys + [TARGET_COLUMN]].isna().sum().sum()),
        "class_distribution": {k: int(v) for k, v in df[TARGET_COLUMN].value_counts().to_dict().items()},
        "class_imbalance_ratio": float(df[TARGET_COLUMN].value_counts().max() / max(1, df[TARGET_COLUMN].value_counts().min())),
        "feature_label_conflict_rows": conflict_rows,
    }


def auto_tune_params(x_train: pd.DataFrame, y_train: pd.Series) -> tuple[dict, dict]:
    grid = {
        "criterion": ["gini", "entropy", "log_loss"],
        "max_depth": [8, 10, 12, None],
        "min_samples_leaf": [1, 3, 5, 8],
        "min_samples_split": [2, 5, 10, 20],
        "max_features": [None, "sqrt", "log2"],
    }
    best_params = None
    best_score = None
    checked = 0

    for params in ParameterGrid(grid):
        checked += 1
        pipeline = build_pipeline(**params)
        pipeline.fit(x_train, y_train)
        pred_train = pipeline.predict(x_train)
        score = f1_score(y_train, pred_train, average="macro", zero_division=0)
        if best_score is None or score > best_score:
            best_score = score
            best_params = params

    return best_params or {}, {"grid_size": checked, "selection_metric": "train_f1_macro", "best_train_f1_macro": float(best_score or 0.0)}


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate final Decision Tree accommodation recommender model.")
    parser.add_argument("--csv", default="thesis_data_templates/accommodation_reco_training_final_v1.csv")
    parser.add_argument("--save-dir", default="artifacts/decision_tree_final")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--criterion", default="entropy", choices=["gini", "entropy", "log_loss"])
    parser.add_argument("--max-depth", type=int, default=0)
    parser.add_argument("--min-samples-leaf", type=int, default=3)
    parser.add_argument("--min-samples-split", type=int, default=2)
    parser.add_argument("--max-features", default="none", choices=["none", "sqrt", "log2"])
    parser.add_argument("--auto-tune", action="store_true")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(csv_path)
    x = df[NUMERIC_COLUMNS + CATEGORICAL_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=args.test_size,
        random_state=42,
        stratify=y,
    )

    model_params = {
        "criterion": str(args.criterion),
        "max_depth": None if args.max_depth in (None, 0) else int(args.max_depth),
        "min_samples_leaf": int(args.min_samples_leaf),
        "min_samples_split": int(args.min_samples_split),
        "max_features": None if str(args.max_features).lower() == "none" else str(args.max_features),
    }
    tune_summary = {}
    if args.auto_tune:
        tuned_params, tune_summary = auto_tune_params(x_train, y_train)
        if tuned_params:
            model_params = tuned_params

    pipeline = build_pipeline(**model_params)
    pipeline.fit(x_train, y_train)

    y_pred = pipeline.predict(x_test)
    class_names = sorted(VALID_TARGETS)
    cm = confusion_matrix(y_test, y_pred, labels=class_names)
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    metrics = {
        "dataset_csv": str(csv_path),
        "total_rows": int(len(df)),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "target_distribution_full": {k: int(v) for k, v in y.value_counts().to_dict().items()},
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "classification_report": report_dict,
        "classes": class_names,
        "confusion_matrix": cm.tolist(),
        "model_params": {
            "algorithm": "DecisionTreeClassifier",
            "criterion": str(model_params["criterion"]),
            "max_depth": (
                None
                if model_params["max_depth"] is None
                else int(model_params["max_depth"])
            ),
            "min_samples_leaf": int(model_params["min_samples_leaf"]),
            "min_samples_split": int(model_params["min_samples_split"]),
            "max_features": (
                None
                if model_params["max_features"] is None
                else str(model_params["max_features"])
            ),
            "class_weight": "balanced",
            "random_state": 42,
            "test_size": float(args.test_size),
        },
        "feature_columns_base": BASE_FEATURE_COLUMNS,
        "feature_columns_engineered": ENGINEERED_NUMERIC_COLUMNS,
        "data_quality_audit": audit_dataset(df),
        "tuning_summary": tune_summary,
    }

    preprocess = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    feature_names = preprocess.get_feature_names_out().tolist()
    importances = model.feature_importances_.tolist()
    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False)
    metrics["feature_importance_top10"] = importance_df.head(10).to_dict(orient="records")

    pipeline_path = save_dir / "decision_tree_final.pkl"
    model_only_path = save_dir / "decision_tree_classifier_v1.pkl"
    preprocessor_path = save_dir / "decision_tree_preprocessor_v1.pkl"
    metrics_path = save_dir / "evaluation_metrics_v1.json"
    cm_csv_path = save_dir / "confusion_matrix_v1.csv"
    cm_json_path = save_dir / "confusion_matrix_v1.json"
    cm_img_path = save_dir / "confusion_matrix_v1.png"
    importance_csv_path = save_dir / "feature_importance_v1.csv"

    with pipeline_path.open("wb") as f:
        pickle.dump(pipeline, f)
    with model_only_path.open("wb") as f:
        pickle.dump(model, f)
    with preprocessor_path.open("wb") as f:
        pickle.dump(preprocess, f)

    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(cm_csv_path, encoding="utf-8")
    cm_json_payload = {
        "classes": class_names,
        "matrix": cm.tolist(),
    }
    cm_json_path.write_text(json.dumps(cm_json_payload, indent=2), encoding="utf-8")
    save_confusion_matrix_figure(cm, class_names, cm_img_path)
    importance_df.to_csv(importance_csv_path, index=False, encoding="utf-8")

    metrics["outputs"] = {
        "pipeline_model": str(pipeline_path),
        "model_only": str(model_only_path),
        "preprocessor": str(preprocessor_path),
        "metrics_json": str(metrics_path),
        "confusion_matrix_csv": str(cm_csv_path),
        "confusion_matrix_json": str(cm_json_path),
        "confusion_matrix_image": str(cm_img_path),
        "feature_importance_csv": str(importance_csv_path),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
