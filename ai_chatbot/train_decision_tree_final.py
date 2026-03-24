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
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


TARGET_COLUMN = "relevance_label"
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
VALID_TARGETS = {"relevant", "not_relevant"}


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    missing = set(FEATURE_COLUMNS + [TARGET_COLUMN]) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = df.copy()
    out[TARGET_COLUMN] = out[TARGET_COLUMN].fillna("").astype(str).str.strip().str.lower()
    out = out[out[TARGET_COLUMN].isin(VALID_TARGETS)]
    if out.empty:
        raise ValueError("No valid rows after target filtering.")

    for col in NUMERIC_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in CATEGORICAL_COLUMNS:
        out[col] = out[col].fillna("").astype(str).str.strip().str.lower()
    return out


def build_pipeline(max_depth: int, min_samples_leaf: int) -> Pipeline:
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
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
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


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate final Decision Tree accommodation recommender model.")
    parser.add_argument("--csv", default="thesis_data_templates/accommodation_reco_training_final_v1.csv")
    parser.add_argument("--save-dir", default="artifacts/decision_tree_final")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--min-samples-leaf", type=int, default=5)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(csv_path)
    x = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=args.test_size,
        random_state=42,
        stratify=y,
    )

    pipeline = build_pipeline(
        max_depth=int(args.max_depth),
        min_samples_leaf=int(args.min_samples_leaf),
    )
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
            "max_depth": int(args.max_depth),
            "min_samples_leaf": int(args.min_samples_leaf),
            "class_weight": "balanced",
            "random_state": 42,
            "test_size": float(args.test_size),
        },
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
