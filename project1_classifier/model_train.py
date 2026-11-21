"""
CNN model training and metric logging script for Project 1:
Traditional Medicinal Flower Identifier.

Loads training and validation flower images from `data/flower_images/`,
trains a Convolutional Neural Network (CNN) with data augmentation,
saves `saved_models/flower_cnn_model.h5` and `saved_models/class_labels.json`,
and optionally saves a training history plot to `assets/accuracy_loss_curve.png`.
"""

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import callbacks, layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DEFAULT_IMG_SIZE: tuple[int, int] = (150, 150)
DEFAULT_BATCH_SIZE: int = 32
DEFAULT_EPOCHS: int = 15


def get_project_root() -> Path:
    """Resolve repository root from <repo>/project1_classifier/model_train.py."""
    return Path(__file__).resolve().parent.parent


def create_data_generators(
    train_dir: Path,
    val_dir: Path,
    img_size: tuple[int, int] = DEFAULT_IMG_SIZE,
    batch_size: int = DEFAULT_BATCH_SIZE,
):
    """Create augmented training and normalized validation image generators."""
    if not train_dir.exists():
        raise FileNotFoundError(
            f"Training directory not found at '{train_dir}'. "
            "Prepare your train/ and validation/ image folders first."
        )
    if not val_dir.exists():
        raise FileNotFoundError(
            f"Validation directory not found at '{val_dir}'. "
            "Prepare your train/ and validation/ image folders first."
        )

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_generator = train_datagen.flow_from_directory(
        directory=str(train_dir),
        target_size=img_size,
        batch_size=batch_size,
        class_mode="sparse",
        shuffle=True,
    )

    val_generator = val_datagen.flow_from_directory(
        directory=str(val_dir),
        target_size=img_size,
        batch_size=batch_size,
        class_mode="sparse",
        shuffle=False,
    )

    return train_generator, val_generator


def build_cnn_model(
    num_classes: int,
    img_size: tuple[int, int] = DEFAULT_IMG_SIZE,
    learning_rate: float = 1e-3,
) -> tf.keras.Model:
    """Build and compile a CNN classifier for medicinal flower images."""
    if num_classes < 2:
        raise ValueError(f"Expected at least 2 flower classes, found {num_classes}.")

    input_shape = (img_size[0], img_size[1], 3)

    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_training_plot(history: tf.keras.callbacks.History, output_path: Path) -> None:
    """Save accuracy and loss curves across epochs to disk."""
    hist = history.history
    epochs_range = range(1, len(hist.get("accuracy", [])) + 1)

    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax_acc.plot(epochs_range, hist.get("accuracy", []), label="Train Accuracy")
    if "val_accuracy" in hist:
        ax_acc.plot(epochs_range, hist["val_accuracy"], label="Val Accuracy")
    ax_acc.set_title("Model Accuracy")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.legend()
    ax_acc.grid(True, alpha=0.3)

    ax_loss.plot(epochs_range, hist.get("loss", []), label="Train Loss")
    if "val_loss" in hist:
        ax_loss.plot(epochs_range, hist["val_loss"], label="Val Loss")
    ax_loss.set_title("Model Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def train_model(
    data_dir: Path | str | None = None,
    model_output_path: Path | str | None = None,
    labels_output_path: Path | str | None = None,
    plot_output_path: Path | str | None = None,
    img_size: tuple[int, int] = DEFAULT_IMG_SIZE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = 1e-3,
    seed: int = 42,
) -> dict:
    """Train the CNN classifier and save model weights, class labels, and metrics."""
    tf.keras.utils.set_random_seed(seed)
    project_root = get_project_root()

    base_data_dir = (
        Path(data_dir)
        if data_dir is not None
        else project_root / "data" / "flower_images"
    )
    train_dir = base_data_dir / "train"
    val_dir = base_data_dir / "validation"

    resolved_model_path = (
        Path(model_output_path)
        if model_output_path is not None
        else project_root / "saved_models" / "flower_cnn_model.h5"
    )
    resolved_labels_path = (
        Path(labels_output_path)
        if labels_output_path is not None
        else project_root / "saved_models" / "class_labels.json"
    )
    resolved_plot_path = (
        Path(plot_output_path)
        if plot_output_path is not None
        else project_root / "assets" / "accuracy_loss_curve.png"
    )

    train_gen, val_gen = create_data_generators(
        train_dir=train_dir,
        val_dir=val_dir,
        img_size=img_size,
        batch_size=batch_size,
    )

    # Map numeric class index (as string for JSON) -> folder/flower label
    index_to_label = {str(idx): label for label, idx in train_gen.class_indices.items()}
    num_classes = len(index_to_label)

    resolved_labels_path.parent.mkdir(parents=True, exist_ok=True)
    with open(resolved_labels_path, "w", encoding="utf-8") as f:
        json.dump(index_to_label, f, indent=2)

    model = build_cnn_model(
        num_classes=num_classes,
        img_size=img_size,
        learning_rate=learning_rate,
    )

    resolved_model_path.parent.mkdir(parents=True, exist_ok=True)
    model_callbacks = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
        ),
        callbacks.ModelCheckpoint(
            filepath=str(resolved_model_path),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]

    print(f"Training CNN on {num_classes} classes: {list(index_to_label.values())}")
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=model_callbacks,
    )

    # Ensure final model file exists even if val_accuracy did not improve
    if not resolved_model_path.exists():
        model.save(str(resolved_model_path))

    save_training_plot(history, resolved_plot_path)

    val_loss, val_acc = model.evaluate(val_gen, verbose=0)
    final_train_acc = float(history.history["accuracy"][-1])
    final_val_acc = float(val_acc)
    final_val_loss = float(val_loss)

    print("\n--- CNN Training Summary ---")
    print(f"Classes Saved      : {resolved_labels_path}")
    print(f"Model Saved        : {resolved_model_path}")
    print(f"Plot Saved         : {resolved_plot_path}")
    print(f"Final Train Acc    : {final_train_acc:.4f}")
    print(f"Validation Acc     : {final_val_acc:.4f}")
    print(f"Validation Loss    : {final_val_loss:.4f}")

    return {
        "num_classes": num_classes,
        "final_train_accuracy": final_train_acc,
        "validation_accuracy": final_val_acc,
        "validation_loss": final_val_loss,
        "class_labels": index_to_label,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train medicinal flower CNN classifier.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of epochs.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size.")
    parser.add_argument("--img-size", type=int, default=150, help="Square image width/height.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_model(
        img_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()