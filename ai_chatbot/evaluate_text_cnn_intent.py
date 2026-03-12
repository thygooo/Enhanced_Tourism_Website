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


def load_model_with_repair(model_path: Path, corpus: list[str]):
    model = tf.keras.models.load_model(model_path, compile=False)
    try:
        model.predict(np.array(["healthcheck"], dtype=object), verbose=0)
        return model
    except Exception as exc:
        if "table not initialized" not in str(exc).lower():
            raise

    old_vectorizer = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.TextVectorization):
            old_vectorizer = layer
            break
    if old_vectorizer is None:
        raise RuntimeError("TextVectorization layer not found for repair.")

    vec_cfg = old_vectorizer.get_config()
    max_tokens = int(vec_cfg.get("max_tokens") or 6000)
    sequence_length = int(vec_cfg.get("output_sequence_length") or 48)

    emb_weights = model.get_layer("embedding").get_weights()
    conv_weights = model.get_layer("conv1d").get_weights()
    dense_weights = model.get_layer("class_probs").get_weights()
    class_count = int(dense_weights[0].shape[1])

    repaired, vectorizer = build_model(
        num_classes=class_count,
        max_tokens=max_tokens,
        sequence_length=sequence_length,
    )
    vectorizer.adapt(tf.data.Dataset.from_tensor_slices(corpus).batch(32))
    repaired.get_layer("embedding").set_weights(emb_weights)
    repaired.get_layer("conv1d").set_weights(conv_weights)
    repaired.get_layer("class_probs").set_weights(dense_weights)
    repaired.predict(np.array(["healthcheck"], dtype=object), verbose=0)
    return repaired


def save_confusion_matrix_image(cm: np.ndarray, classes: list[str], image_path: Path) -> None:
    image_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.figure.colorbar(im, ax=ax)
        ax.set(
            xticks=np.arange(len(classes)),
            yticks=np.arange(len(classes)),
            xticklabels=classes,
            yticklabels=classes,
            ylabel="True label",
            xlabel="Predicted label",
            title="Text-CNN Intent Confusion Matrix",
        )
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        thresh = cm.max() / 2.0 if cm.size else 0.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    format(cm[i, j], "d"),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                )
        fig.tight_layout()
        fig.savefig(image_path, dpi=180)
        plt.close(fig)
        return
    except ModuleNotFoundError:
        pass

    from PIL import Image, ImageDraw, ImageFont

    n = len(classes)
    cell = 90
    left = 280
    top = 120
    width = left + n * cell + 40
    height = top + n * cell + 40
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    max_val = int(cm.max()) if cm.size else 1
    for i in range(n):
        for j in range(n):
            val = int(cm[i, j])
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

    for idx, name in enumerate(classes):
        draw.text((left - 260, top + idx * cell + 35), name[:34], fill="black", font=font)
        draw.text((left + idx * cell + 10, top - 25), name[:10], fill="black", font=font)

    img.save(image_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Text-CNN intent model and export full metrics.")
    parser.add_argument("--csv", default="thesis_data_templates/text_cnn_messages_final_expanded_v3_clean.csv")
    parser.add_argument("--model-path", default="artifacts/text_cnn_intent/text_cnn_intent.h5")
    parser.add_argument("--label-map", default="artifacts/text_cnn_intent/label_map.json")
    parser.add_argument("--out-dir", default="artifacts/text_cnn_intent")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    model_path = Path(args.model_path)
    label_map_path = Path(args.label_map)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    for col in ("message_text", "label_intent", "split"):
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    df["message_text"] = df["message_text"].astype(str)
    df["label_intent"] = df["label_intent"].astype(str)
    df["split"] = df["split"].astype(str).str.lower()

    label_payload = json.loads(label_map_path.read_text(encoding="utf-8"))
    classes = [str(c) for c in label_payload.get("classes", [])]
    if not classes:
        raise ValueError("Label map classes are missing.")
    label_to_index = {c: i for i, c in enumerate(classes)}

    corpus = df["message_text"].tolist()
    model = load_model_with_repair(model_path, corpus)

    test_df = df[df["split"] == "test"].copy()
    if test_df.empty:
        raise ValueError("Test split is empty.")

    x_test = np.array(test_df["message_text"].tolist(), dtype=object)
    y_true = np.array([label_to_index[v] for v in test_df["label_intent"].tolist()], dtype=np.int32)
    probs = model.predict(x_test, verbose=0)
    y_pred = probs.argmax(axis=1)

    acc = float(accuracy_score(y_true, y_pred))
    prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    prec_weighted = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    rec_weighted = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    per_class_report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(classes))),
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )

    cm_csv = out_dir / "confusion_matrix_v3.csv"
    cm_json = out_dir / "confusion_matrix_v3.json"
    cm_png = out_dir / "confusion_matrix_v3.png"
    full_json = out_dir / "evaluation_metrics_v3_full.json"

    pd.DataFrame(cm, index=classes, columns=classes).to_csv(cm_csv)
    cm_json.write_text(json.dumps({"classes": classes, "matrix": cm.tolist()}, indent=2), encoding="utf-8")
    save_confusion_matrix_image(cm, classes, cm_png)

    per_class = {
        c: {
            "precision": float(per_class_report[c]["precision"]),
            "recall": float(per_class_report[c]["recall"]),
            "f1_score": float(per_class_report[c]["f1-score"]),
            "support": int(per_class_report[c]["support"]),
        }
        for c in classes
    }
    payload = {
        "dataset": str(csv_path),
        "model_path": str(model_path),
        "test_size": int(len(test_df)),
        "accuracy": acc,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
        "precision_weighted": prec_weighted,
        "recall_weighted": rec_weighted,
        "f1_weighted": f1_weighted,
        "per_class_metrics": per_class,
        "confusion_matrix": cm.tolist(),
        "classes": classes,
        "outputs": {
            "metrics_json": str(full_json),
            "confusion_matrix_csv": str(cm_csv),
            "confusion_matrix_json": str(cm_json),
            "confusion_matrix_image": str(cm_png),
        },
    }
    full_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
