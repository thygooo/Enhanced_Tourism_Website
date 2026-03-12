import argparse
import json
from pathlib import Path

import pandas as pd

from ai_chatbot.train_text_cnn_demo import (
    ensure_minimum_splits,
    print_summary,
    to_xy,
    build_model,
)

try:
    import numpy as np
    import tensorflow as tf
except ImportError as exc:
    raise SystemExit(
        "TensorFlow is not installed. Run: pip install tensorflow pandas\n"
        f"Original error: {exc}"
    )


REQUIRED_COLUMNS = {"message_text", "label_intent", "split"}
VALID_SPLITS = {"train", "val", "test"}


def load_template_dataset(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip() for c in df.columns]
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    df = df.copy()
    df["message_text"] = df["message_text"].fillna("").astype(str).str.strip()
    df["label_intent"] = df["label_intent"].fillna("").astype(str).str.strip()
    df["split"] = df["split"].fillna("").astype(str).str.strip().str.lower()

    df = df[(df["message_text"] != "") & (df["label_intent"] != "")]
    df = df[df["split"].isin(VALID_SPLITS)]
    if df.empty:
        raise ValueError("Dataset is empty after filtering invalid rows.")

    # Adapt to the existing Text-CNN demo trainer schema.
    return df.rename(
        columns={
            "message_text": "text_content",
            "label_intent": "class_label",
        }
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train a pilot Text-CNN using thesis_data_templates/text_cnn_messages.csv schema."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="thesis_data_templates/text_cnn_messages.csv",
        help="Path to text_cnn_messages.csv",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=2000)
    parser.add_argument("--sequence-length", type=int, default=40)
    parser.add_argument("--save-dir", type=str, default="artifacts/text_cnn_template_demo")
    args = parser.parse_args()

    tf.random.set_seed(42)
    np.random.seed(42)

    csv_path = Path(args.csv)
    print(f"Loading template dataset from: {csv_path}")
    df = load_template_dataset(csv_path)
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

    callbacks = (
        [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)]
        if len(x_val) > 0 else []
    )
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
    print(f"Final train accuracy: {history.history.get('accuracy', [None])[-1]}")
    if "val_accuracy" in history.history:
        print(f"Final val accuracy:   {history.history.get('val_accuracy', [None])[-1]}")

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
    model_path = save_dir / "text_cnn_template_demo.keras"
    label_map_path = save_dir / "label_map.json"
    model.save(model_path)
    label_map_path.write_text(json.dumps({"classes": classes}, indent=2), encoding="utf-8")
    print("\nSaved artifacts")
    print("-" * 40)
    print(f"Model:     {model_path}")
    print(f"Label map: {label_map_path}")
    print(
        "\nNote: Pilot/demo training only. Retrain with real labeled data before thesis performance reporting."
    )


if __name__ == "__main__":
    main()
