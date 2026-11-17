"""Pair and preprocess real images and binary segmentation masks."""

from pathlib import Path

import tensorflow as tf


SUPPORTED_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png"}


def _index_images(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {directory}")

    indexed: dict[str, Path] = {}
    for path in directory.iterdir():
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            key = path.stem.lower()
            if key in indexed:
                raise ValueError(f"Duplicate image filename stem in {directory}: {path.stem}")
            indexed[key] = path
    if not indexed:
        raise ValueError(f"No supported images found in {directory}")
    return indexed


def find_pairs(image_dir: Path, mask_dir: Path) -> list[tuple[Path, Path]]:
    """Find one-to-one image/mask pairs whose filenames match by stem."""
    images = _index_images(image_dir)
    masks = _index_images(mask_dir)
    missing_masks = sorted(images.keys() - masks.keys())
    missing_images = sorted(masks.keys() - images.keys())
    if missing_masks or missing_images:
        raise ValueError(
            "Image/mask filenames must match by stem. "
            f"Masks missing for: {missing_masks}; images missing for: {missing_images}"
        )
    return [(images[key], masks[key]) for key in sorted(images)]


def _load_pair(
    image_path: tf.Tensor,
    mask_path: tf.Tensor,
    image_size: int,
) -> tuple[tf.Tensor, tf.Tensor]:
    image = tf.io.decode_image(
        tf.io.read_file(image_path), channels=3, expand_animations=False
    )
    mask = tf.io.decode_image(
        tf.io.read_file(mask_path), channels=1, expand_animations=False
    )
    image.set_shape([None, None, 3])
    mask.set_shape([None, None, 1])

    image = tf.image.resize(image, [image_size, image_size], method="bilinear")
    mask = tf.image.resize(mask, [image_size, image_size], method="nearest")
    image = tf.cast(image, tf.float32) / 255.0

    # Accept binary masks stored as either 0/1 or 0/255.
    threshold = tf.where(tf.reduce_max(mask) <= 1, 0.0, 127.5)
    mask = tf.cast(mask > threshold, tf.float32)
    return image, mask


def make_dataset(
    pairs: list[tuple[Path, Path]],
    image_size: int = 256,
    batch_size: int = 8,
    shuffle: bool = False,
    seed: int = 42,
) -> tf.data.Dataset:
    """Build a batched dataset from image/mask pairs."""
    if not pairs:
        raise ValueError("Cannot build a dataset from zero image/mask pairs")
    if image_size < 1:
        raise ValueError("image_size must be positive")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    image_paths = [str(image) for image, _ in pairs]
    mask_paths = [str(mask) for _, mask in pairs]
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    if shuffle:
        dataset = dataset.shuffle(len(pairs), seed=seed, reshuffle_each_iteration=True)
    dataset = dataset.map(
        lambda image, mask: _load_pair(image, mask, image_size),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def load_dataset(
    image_dir: Path,
    mask_dir: Path,
    image_size: int = 256,
    batch_size: int = 8,
    shuffle: bool = False,
    seed: int = 42,
) -> tf.data.Dataset:
    """Load real paired images and masks from separate directories."""
    pairs = find_pairs(Path(image_dir), Path(mask_dir))
    return make_dataset(pairs, image_size, batch_size, shuffle, seed)
