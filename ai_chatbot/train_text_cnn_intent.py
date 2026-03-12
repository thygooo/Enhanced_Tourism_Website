import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from ai_chatbot.train_text_cnn_demo import build_model


def load_dataset(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"message_text", "label_intent", "split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")
    out = df.copy()
    out["message_text"] = out["message_text"].fillna("").astype(str).str.strip()
    out["label_intent"] = out["label_intent"].fillna("").astype(str).str.strip()
    out["split"] = out["split"].fillna("").astype(str).str.strip().str.lower()
    out = out[(out["message_text"] != "") & (out["label_intent"] != "")]
    out = out[out["split"].isin({"train", "val", "test"})].reset_index(drop=True)
    if out.empty:
        raise ValueError("Dataset is empty after filtering.")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Train intent Text-CNN and save runtime artifacts.")
    parser.add_argument("--csv", default="thesis_data_templates/text_cnn_messages_final_expanded_v2.csv")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=4000)
    parser.add_argument("--sequence-length", type=int, default=40)
    parser.add_argument("--save-dir", default="artifacts/text_cnn_intent")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    df = load_dataset(Path(args.csv))
    classes = sorted(df["label_intent"].unique().tolist())
    label_to_index = {label: i for i, label in enumerate(classes)}

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    x_train = train_df["message_text"].astype(str).tolist()
    y_train = np.array([label_to_index[v] for v in train_df["label_intent"].tolist()], dtype=np.int32)
    x_val = val_df["message_text"].astype(str).tolist()
    y_val = np.array([label_to_index[v] for v in val_df["label_intent"].tolist()], dtype=np.int32)
    x_test = test_df["message_text"].astype(str).tolist()
    y_test = np.array([label_to_index[v] for v in test_df["label_intent"].tolist()], dtype=np.int32)

    model, vectorizer = build_model(
        num_classes=len(classes),
        max_tokens=args.max_tokens,
        sequence_length=args.sequence_length,
    )
    vectorizer.adapt(tf.data.Dataset.from_tensor_slices(x_train).batch(args.batch_size))

    callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)]
    history = model.fit(
        np.array(x_train, dtype=object),
        y_train,
        validation_data=(np.array(x_val, dtype=object), y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    test_loss, test_acc = model.evaluate(np.array(x_test, dtype=object), y_test, verbose=0)
    probs = model.predict(np.array(x_test, dtype=object), verbose=0)
    y_pred = probs.argmax(axis=1)
    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(classes))))
    per_class_report = classification_report(
        y_test,
        y_pred,
        labels=list(range(len(classes))),
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )

    def _save_cm_image(matrix: np.ndarray, labels: list[str], image_path: Path) -> None:
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
            ax.figure.colorbar(im, ax=ax)
            ax.set(
                xticks=np.arange(len(labels)),
                yticks=np.arange(len(labels)),
                xticklabels=labels,
                yticklabels=labels,
                ylabel="True label",
                xlabel="Predicted label",
                title="Text-CNN Intent Confusion Matrix",
            )
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            thresh = matrix.max() / 2.0 if matrix.size else 0.0
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    ax.text(
                        j,
                        i,
                        format(matrix[i, j], "d"),
                        ha="center",
                        va="center",
                        color="white" if matrix[i, j] > thresh else "black",
                    )
            fig.tight_layout()
            fig.savefig(image_path, dpi=180)
            plt.close(fig)
            return
        except ModuleNotFoundError:
            pass

        from PIL import Image, ImageDraw, ImageFont

        n = len(labels)
        cell = 90
        left = 280
        top = 120
        width = left + n * cell + 40
        height = top + n * cell + 40
        img = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()

        max_val = int(matrix.max()) if matrix.size else 1
        for i in range(n):
            for j in range(n):
                val = int(matrix[i, j])
                shade = 255 - int((val / max_val) * 180) if max_val > 0 else 255
                color = (shade, shade, 255)
                x0 = left + j * cell
                y0 = top + i * cell
                x1 = x0 + cell
                y1 = y0 + cell
                draw.rectangle([x0, y0, x1, y1], fill=color, outline="black")
                draw.text((x0 + 32, y0 + 36), str(val), fill="black", font=font)

        draw.text((left, 20), "Confusion Matrix (Text-CNN Intent)", fill="black", font=font)
        draw.text((left, 45), "Rows=True, Cols=Pred", fill="black", font=font)
        for idx, name in enumerate(labels):
            draw.text((left - 260, top + idx * cell + 35), name[:34], fill="black", font=font)
            draw.text((left + idx * cell + 10, top - 25), name[:10], fill="black", font=font)
        img.save(image_path)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model_path = save_dir / "text_cnn_intent.h5"
    label_map_path = save_dir / "label_map.json"
    metrics_path = save_dir / "evaluation_metrics_expanded_v2.json"
    full_metrics_path = save_dir / "evaluation_metrics_v3_full.json"
    cm_csv_path = save_dir / "confusion_matrix_v3.csv"
    cm_json_path = save_dir / "confusion_matrix_v3.json"
    cm_img_path = save_dir / "confusion_matrix_v3.png"
    vocab_path = save_dir / "text_cnn_intent_vocab.json"

    model.save(model_path)
    label_map_path.write_text(json.dumps({"classes": classes}, indent=2), encoding="utf-8")
    vocab_path.write_text(json.dumps(vectorizer.get_vocabulary(), indent=2), encoding="utf-8")
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(cm_csv_path)
    cm_json_path.write_text(json.dumps({"classes": classes, "matrix": cm.tolist()}, indent=2), encoding="utf-8")
    _save_cm_image(cm, classes, cm_img_path)

    metrics = {
        "dataset": str(Path(args.csv)),
        "epochs_ran": len(history.history.get("loss", [])),
        "train_accuracy_final": float(history.history.get("accuracy", [0.0])[-1]),
        "val_accuracy_final": float(history.history.get("val_accuracy", [0.0])[-1]),
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "class_count": len(classes),
        "classes": classes,
        "rows": int(len(df)),
        "rows_train": int(len(train_df)),
        "rows_val": int(len(val_df)),
        "rows_test": int(len(test_df)),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    per_class = {
        c: {
            "precision": float(per_class_report[c]["precision"]),
            "recall": float(per_class_report[c]["recall"]),
            "f1_score": float(per_class_report[c]["f1-score"]),
            "support": int(per_class_report[c]["support"]),
        }
        for c in classes
    }
    full_metrics = {
        "dataset": str(Path(args.csv)),
        "test_size": int(len(test_df)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "per_class_metrics": per_class,
        "confusion_matrix": cm.tolist(),
        "classes": classes,
        "outputs": {
            "metrics_json": str(full_metrics_path),
            "confusion_matrix_csv": str(cm_csv_path),
            "confusion_matrix_json": str(cm_json_path),
            "confusion_matrix_image": str(cm_img_path),
        },
    }
    full_metrics_path.write_text(json.dumps(full_metrics, indent=2), encoding="utf-8")

    print(f"Saved model: {model_path}")
    print(f"Saved labels: {label_map_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved full metrics: {full_metrics_path}")
    print(f"Saved confusion matrix: {cm_csv_path}, {cm_json_path}, {cm_img_path}")
    print(f"Saved vectorizer vocab: {vocab_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
