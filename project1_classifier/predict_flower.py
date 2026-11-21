"""
Inference script for Project 1: Traditional Medicinal Flower Identifier.

Loads the trained CNN model (`saved_models/flower_cnn_model.h5`) and class label
mapping (`saved_models/class_labels.json`), classifies a flower image, and
retrieves its medicinal uses, active compounds, and warnings from `medical_database.py`.
"""

import argparse
import json
from pathlib import Path
from typing import Any, BinaryIO
import numpy as np
from PIL import Image
import tensorflow as tf

try:
    from .medical_database import get_flower_info
except ImportError:
    from medical_database import get_flower_info

DEFAULT_IMG_SIZE: tuple[int, int] = (150, 150)


def get_project_root() -> Path:
    """Resolve repository root from <repo>/project1_classifier/predict_flower.py."""
    return Path(__file__).resolve().parent.parent


def get_default_model_path() -> Path:
    return get_project_root() / "saved_models" / "flower_cnn_model.h5"


def get_default_labels_path() -> Path:
    return get_project_root() / "saved_models" / "class_labels.json"


def load_class_labels(labels_path: Path | str | None = None) -> dict[int, str]:
    """Load index-to-class mapping from `saved_models/class_labels.json`."""
    resolved_path = (
        Path(labels_path) if labels_path is not None else get_default_labels_path()
    )
    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Class labels file not found at '{resolved_path}'. "
            "Run `project1_classifier/model_train.py` first."
        )

    with open(resolved_path, "r", encoding="utf-8") as f:
        raw_mapping = json.load(f)

    return {int(idx): str(label) for idx, label in raw_mapping.items()}


def load_flower_model(model_path: Path | str | None = None) -> tf.keras.Model:
    """Load the trained TensorFlow/Keras CNN flower classifier."""
    resolved_path = (
        Path(model_path) if model_path is not None else get_default_model_path()
    )
    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Trained CNN model not found at '{resolved_path}'. "
            "Run `project1_classifier/model_train.py` first."
        )

    return tf.keras.models.load_model(str(resolved_path))


def preprocess_image(
    image_source: Path | str | Image.Image | BinaryIO,
    img_size: tuple[int, int] = DEFAULT_IMG_SIZE,
) -> np.ndarray:
    """Convert an image path, file buffer, or PIL Image into a normalized batch tensor."""
    if isinstance(image_source, Image.Image):
        pil_img = image_source.convert("RGB")
    else:
        pil_img = Image.open(image_source).convert("RGB")

    resized_img = pil_img.resize(img_size, Image.Resampling.BILINEAR)
    img_array = np.asarray(resized_img, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)


def predict_flower(
    image_source: Path | str | Image.Image | BinaryIO,
    model: tf.keras.Model | None = None,
    class_labels: dict[int, str] | None = None,
    model_path: Path | str | None = None,
    labels_path: Path | str | None = None,
    img_size: tuple[int, int] = DEFAULT_IMG_SIZE,
    top_k: int = 3,
) -> dict[str, Any]:
    """
    Classify a medicinal flower image and return prediction + medical metadata.
    """
    classifier = model if model is not None else load_flower_model(model_path)
    labels_map = (
        class_labels if class_labels is not None else load_class_labels(labels_path)
    )

    # Infer expected input size from model if available
    if getattr(classifier, "input_shape", None) and len(classifier.input_shape) >= 3:
        h, w = classifier.input_shape[1], classifier.input_shape[2]
        if isinstance(h, int) and isinstance(w, int):
            img_size = (w, h)

    input_batch = preprocess_image(image_source, img_size=img_size)
    probs = np.asarray(classifier.predict(input_batch, verbose=0)[0], dtype=float)

    predicted_idx = int(np.argmax(probs))
    predicted_label = labels_map.get(predicted_idx, f"class_{predicted_idx}")
    confidence = float(probs[predicted_idx])

    k = min(max(1, top_k), len(probs))
    top_indices = np.argsort(probs)[::-1][:k]
    top_predictions = [
        {
            "label": labels_map.get(int(idx), f"class_{int(idx)}"),
            "confidence": float(probs[int(idx)]),
        }
        for idx in top_indices
    ]

    medical_info = get_flower_info(predicted_label)

    return {
        "predicted_class": predicted_label,
        "confidence": confidence,
        "top_predictions": top_predictions,
        "medical_info": medical_info,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict medicinal flower from an image.")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to flower image (defaults to `assets/sample_hibiscus.jpg`).",
    )
    args = parser.parse_args()

    image_path = (
        Path(args.image)
        if args.image is not None
        else get_project_root() / "assets" / "sample_hibiscus.jpg"
    )
    if not image_path.exists():
        raise FileNotFoundError(
            f"Sample image not found at '{image_path}'. Pass `--image <path>`."
        )

    result = predict_flower(image_path)
    info = result["medical_info"]

    print("--- Medicinal Flower Identification Result ---")
    print(f"Identified Plant     : {info['common_name']} ({result['predicted_class']})")
    print(f"Botanical Name       : {info['botanical_name']}")
    print(f"Confidence           : {result['confidence'] * 100:.2f}%")
    print(f"Active Compounds     : {', '.join(info['active_compounds'])}")
    print(f"Medicinal Uses       : {info['uses']}")
    print(f"Precautions/Warnings : {info['warning']}")


if __name__ == "__main__":
    main()