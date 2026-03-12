import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import tensorflow as tf
except ImportError as exc:
    raise SystemExit(
        "TensorFlow is not installed. Run: pip install tensorflow pandas\n"
        f"Original error: {exc}"
    )


REQUIRED_COLUMNS = {"text_content", "class_label", "split"}
VALID_SPLITS = {"train", "val", "test"}


def default_csv_path() -> Path:
    docs = Path.home() / "Documents"
    candidates = [
        docs / "cnn_test_samples.csv",
        docs / "cnn text samples.csv",
        docs / "cnn_text_samples.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def load_dataset(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing))
            + "\nExpected columns include: text_content, class_label, split"
        )

    df = df.copy()
    df["text_content"] = df["text_content"].fillna("").astype(str).str.strip()
    df["class_label"] = df["class_label"].fillna("").astype(str).str.strip()
    df["split"] = df["split"].fillna("").astype(str).str.strip().str.lower()

    df = df[(df["text_content"] != "") & (df["class_label"] != "")]
    df = df[df["split"].isin(VALID_SPLITS)]

    if df.empty:
        raise ValueError("Dataset is empty after filtering invalid rows.")

    return df


def print_summary(df: pd.DataFrame) -> None:
    print("\nDataset summary")
    print("-" * 40)
    print(f"Total rows: {len(df)}")
    print("Rows per split:")
    print(df["split"].value_counts().to_string())
    print("\nRows per class:")
    print(df["class_label"].value_counts().to_string())
    print()


def ensure_minimum_splits(df: pd.DataFrame) -> None:
    present = set(df["split"].unique())
    if "train" not in present:
        raise ValueError("Dataset must contain at least one 'train' row.")
    if "val" not in present:
        print("Warning: no 'val' rows found. Training will run without validation.")
    if "test" not in present:
        print("Warning: no 'test' rows found. Test evaluation will be skipped.")


def to_xy(df: pd.DataFrame, label_to_index: dict[str, int]):
    x = df["text_content"].astype(str).tolist()
    y = np.array([label_to_index[label] for label in df["class_label"].tolist()], dtype=np.int32)
    return x, y


def build_model(num_classes: int, max_tokens: int, sequence_length: int):
    text_input = tf.keras.Input(shape=(1,), dtype=tf.string, name="text")

    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=max_tokens,
        output_mode="int",
        output_sequence_length=sequence_length,
        standardize="lower_and_strip_punctuation",
    )

    x = vectorizer(text_input)
    x = tf.keras.layers.Embedding(input_dim=max_tokens, output_dim=64)(x)
    x = tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation="relu")(x)
    x = tf.keras.layers.GlobalMaxPooling1D()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    output = tf.keras.layers.Dense(num_classes, activation="softmax", name="class_probs")(x)

    model = tf.keras.Model(inputs=text_input, outputs=output)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, vectorizer


def main():
    parser = argparse.ArgumentParser(description="Train a text CNN on thesis progress-check sample data.")
    parser.add_argument("--csv", type=str, default=str(default_csv_path()), help="Path to CSV file")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=2000)
    parser.add_argument("--sequence-length", type=int, default=40)
    parser.add_argument("--save-dir", type=str, default="artifacts/text_cnn_demo")
    args = parser.parse_args()

    tf.random.set_seed(42)
    np.random.seed(42)

    csv_path = Path(args.csv)
    print(f"Loading dataset from: {csv_path}")
    df = load_dataset(csv_path)
    ensure_minimum_splits(df)
    print_summary(df)

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    classes = sorted(df["class_label"].unique().tolist())
    if len(classes) < 2:
        raise ValueError("Need at least 2 classes for classification.")

    label_to_index = {label: i for i, label in enumerate(classes)}
    index_to_label = {i: label for label, i in label_to_index.items()}

    x_train, y_train = to_xy(train_df, label_to_index)
    x_val, y_val = to_xy(val_df, label_to_index) if not val_df.empty else ([], np.array([], dtype=np.int32))
    x_test, y_test = to_xy(test_df, label_to_index) if not test_df.empty else ([], np.array([], dtype=np.int32))

    model, vectorizer = build_model(
        num_classes=len(classes),
        max_tokens=args.max_tokens,
        sequence_length=args.sequence_length,
    )

    vectorizer.adapt(tf.data.Dataset.from_tensor_slices(x_train).batch(args.batch_size))

    print("Model summary")
    print("-" * 40)
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    ] if len(x_val) > 0 else []

    fit_kwargs = {
        "x": np.array(x_train, dtype=object),
        "y": y_train,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "verbose": 1,
    }
    if len(x_val) > 0:
        fit_kwargs["validation_data"] = (np.array(x_val, dtype=object), y_val)
        fit_kwargs["callbacks"] = callbacks

    history = model.fit(**fit_kwargs)

    print("\nTraining complete")
    print("-" * 40)
    final_train_acc = history.history.get("accuracy", [None])[-1]
    final_val_acc = history.history.get("val_accuracy", [None])[-1]
    print(f"Final train accuracy: {final_train_acc}")
    if final_val_acc is not None:
        print(f"Final val accuracy:   {final_val_acc}")

    if len(x_test) > 0:
        test_loss, test_acc = model.evaluate(np.array(x_test, dtype=object), y_test, verbose=0)
        print(f"Test accuracy:        {test_acc}")
        print(f"Test loss:            {test_loss}")

        preds = model.predict(np.array(x_test, dtype=object), verbose=0)
        pred_ids = preds.argmax(axis=1)

        print("\nSample predictions (test split)")
        print("-" * 40)
        for i, text in enumerate(x_test[:5]):
            pred_label = index_to_label[int(pred_ids[i])]
            true_label = index_to_label[int(y_test[i])]
            confidence = float(preds[i][pred_ids[i]])
            print(f"Text: {text}")
            print(f"  true={true_label}  pred={pred_label}  conf={confidence:.3f}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model_path = save_dir / "text_cnn_demo.keras"
    label_map_path = save_dir / "label_map.json"

    model.save(model_path)
    label_map_path.write_text(json.dumps({"classes": classes}, indent=2), encoding="utf-8")

    print("\nSaved artifacts")
    print("-" * 40)
    print(f"Model:     {model_path}")
    print(f"Label map: {label_map_path}")

    print("\nNote: This is for progress-check pipeline demo only. Replace sample data with real gathered data for thesis results.")


if __name__ == "__main__":
    main()
