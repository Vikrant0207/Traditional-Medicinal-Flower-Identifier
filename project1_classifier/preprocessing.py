"""
Dataset organization, cleaning, and ImageDataGenerator setup for Project 1:
Traditional Medicinal Flower Identifier.

Splits a raw class-per-folder image dataset (such as Kaggle's Medicinal Flower
Dataset) into `data/flower_images/train/` and `data/flower_images/validation/`,
verifies image readability, and provides augmented Keras data generators.
"""

import argparse
from pathlib import Path
import random
import shutil
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.preprocessing.image import ImageDataGenerator

VALID_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_IMG_SIZE: tuple[int, int] = (150, 150)
DEFAULT_BATCH_SIZE: int = 32


def get_project_root() -> Path:
    """Resolve repository root from <repo>/project1_classifier/preprocessing.py."""
    return Path(__file__).resolve().parent.parent


def normalize_class_name(folder_name: str) -> str:
    """Normalize raw folder names (e.g., 'Hibiscus Flower' -> 'hibiscus')."""
    cleaned = folder_name.strip().lower().replace("-", "_").replace(" ", "_")
    if cleaned.endswith("_flower"):
        cleaned = cleaned[: -len("_flower")]
    return cleaned


def is_valid_image(file_path: Path) -> bool:
    """Check extension and verify image integrity with PIL."""
    if not file_path.is_file() or file_path.suffix.lower() not in VALID_EXTENSIONS:
        return False
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except (UnidentifiedImageError, OSError, ValueError):
        return False


def split_dataset(
    raw_dir: Path | str,
    output_dir: Path | str | None = None,
    val_ratio: float = 0.2,
    seed: int = 42,
    clear_existing: bool = False,
) -> dict[str, dict[str, int]]:
    """
    Split `raw_dir/<class_name>/*` images into:
      `output_dir/train/<normalized_class_name>/`
      `output_dir/validation/<normalized_class_name>/`
    """
    if not (0.0 < val_ratio < 1.0):
        raise ValueError("`val_ratio` must be strictly between 0.0 and 1.0.")

    source_root = Path(raw_dir)
    if not source_root.exists():
        raise FileNotFoundError(f"Raw dataset directory not found: '{source_root}'")

    target_root = (
        Path(output_dir)
        if output_dir is not None
        else get_project_root() / "data" / "flower_images"
    )
    train_root = target_root / "train"
    val_root = target_root / "validation"

    if clear_existing:
        if train_root.exists():
            shutil.rmtree(train_root)
        if val_root.exists():
            shutil.rmtree(val_root)

    train_root.mkdir(parents=True, exist_ok=True)
    val_root.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    class_dirs = sorted([d for d in source_root.iterdir() if d.is_dir()])
    if not class_dirs:
        raise ValueError(
            f"No class subdirectories found inside '{source_root}'. "
            "Ensure each flower species is in its own folder."
        )

    summary: dict[str, dict[str, int]] = {}

    for class_dir in class_dirs:
        class_key = normalize_class_name(class_dir.name)
        valid_files = [
            f for f in sorted(class_dir.iterdir()) if is_valid_image(f)
        ]
        skipped_count = sum(1 for f in class_dir.iterdir() if f.is_file()) - len(valid_files)

        if len(valid_files) < 2:
            print(f"[WARN] Skipping '{class_dir.name}' (found fewer than 2 valid images).")
            continue

        rng.shuffle(valid_files)
        val_count = max(1, int(round(len(valid_files) * val_ratio)))
        val_files = valid_files[:val_count]
        train_files = valid_files[val_count:]

        train_class_dir = train_root / class_key
        val_class_dir = val_root / class_key
        train_class_dir.mkdir(parents=True, exist_ok=True)
        val_class_dir.mkdir(parents=True, exist_ok=True)

        for idx, src_path in enumerate(train_files, start=1):
            dst_name = f"{class_key}_train_{idx:04d}{src_path.suffix.lower()}"
            shutil.copy2(src_path, train_class_dir / dst_name)

        for idx, src_path in enumerate(val_files, start=1):
            dst_name = f"{class_key}_val_{idx:04d}{src_path.suffix.lower()}"
            shutil.copy2(src_path, val_class_dir / dst_name)

        summary[class_key] = {
            "train": len(train_files),
            "validation": len(val_files),
            "skipped_corrupt": skipped_count,
        }

    return summary


def build_image_generators(
    data_dir: Path | str | None = None,
    img_size: tuple[int, int] = DEFAULT_IMG_SIZE,
    batch_size: int = DEFAULT_BATCH_SIZE,
):
    """Build augmented train and normalized validation ImageDataGenerators."""
    base_dir = (
        Path(data_dir)
        if data_dir is not None
        else get_project_root() / "data" / "flower_images"
    )
    train_dir = base_dir / "train"
    val_dir = base_dir / "validation"

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split raw medicinal flower dataset into train/ and validation/ folders."
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        required=True,
        help="Path to extracted Kaggle dataset folder containing flower class subfolders.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Target output directory (defaults to `data/flower_images`).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation split fraction (default: 0.2).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible train/val splitting.",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing train/ and validation/ folders before copying.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = split_dataset(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
        clear_existing=args.clear,
    )

    print("--- Dataset Split Summary ---")
    total_train = 0
    total_val = 0
    for cls, counts in summary.items():
        total_train += counts["train"]
        total_val += counts["validation"]
        print(
            f"  {cls:<18} -> train: {counts['train']:<4} | "
            f"val: {counts['validation']:<4} | skipped: {counts['skipped_corrupt']}"
        )
    print(f"\nTotal Train Images      : {total_train}")
    print(f"Total Validation Images : {total_val}")


if __name__ == "__main__":
    main()