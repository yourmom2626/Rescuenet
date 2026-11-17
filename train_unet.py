"""Train a binary U-Net from matching image and mask directories."""

import argparse
import random
from pathlib import Path

import tensorflow as tf


SUPPORTED_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


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
    mask = tf.cast(mask > 127.5, tf.float32)
    return image, mask


def make_dataset(
    pairs: list[tuple[Path, Path]],
    image_size: int,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> tf.data.Dataset:
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


def _conv_block(inputs: tf.Tensor, filters: int) -> tf.Tensor:
    x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(inputs)
    return tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)


def build_unet(image_size: int, base_filters: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(image_size, image_size, 3))
    x = inputs
    skips: list[tf.Tensor] = []

    for filters in (base_filters, base_filters * 2, base_filters * 4, base_filters * 8):
        x = _conv_block(x, filters)
        skips.append(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)

    x = _conv_block(x, base_filters * 16)
    for filters, skip in zip(
        (base_filters * 8, base_filters * 4, base_filters * 2, base_filters),
        reversed(skips),
    ):
        x = tf.keras.layers.Conv2DTranspose(filters, 2, strides=2, padding="same")(x)
        x = tf.keras.layers.Concatenate()([x, skip])
        x = _conv_block(x, filters)

    outputs = tf.keras.layers.Conv2D(1, 1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="unet")
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.BinaryIoU(
                target_class_ids=[0, 1],
                threshold=0.5,
                name="mean_iou",
            ),
        ],
    )
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True, help="Directory of input images")
    parser.add_argument("--masks", type=Path, required=True, help="Directory of binary masks")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("unet_model.keras"),
        help="Output Keras model path (default: unet_model.keras)",
    )
    parser.add_argument("--image-size", type=int, default=256, help="Square training size")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--base-filters", type=int, default=32)
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.2,
        help="Fraction of pairs held out for validation; use 0 to disable",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.image_size < 16 or args.image_size % 16 != 0:
        raise ValueError("--image-size must be at least 16 and divisible by 16")
    if args.batch_size < 1 or args.epochs < 1 or args.base_filters < 1:
        raise ValueError("--batch-size, --epochs, and --base-filters must be positive")
    if not 0 <= args.validation_split < 1:
        raise ValueError("--validation-split must be in the range [0, 1)")

    pairs = find_pairs(args.images, args.masks)
    random.Random(args.seed).shuffle(pairs)
    validation_count = int(len(pairs) * args.validation_split)
    if args.validation_split > 0 and validation_count == 0:
        raise ValueError("At least five image/mask pairs are needed for the default validation split")
    if validation_count >= len(pairs):
        raise ValueError("The validation split leaves no training samples")

    validation_pairs = pairs[:validation_count]
    training_pairs = pairs[validation_count:]
    training_data = make_dataset(
        training_pairs, args.image_size, args.batch_size, shuffle=True, seed=args.seed
    )
    validation_data = (
        make_dataset(
            validation_pairs, args.image_size, args.batch_size, shuffle=False, seed=args.seed
        )
        if validation_pairs
        else None
    )

    model = build_unet(args.image_size, args.base_filters)
    monitor = "val_loss" if validation_data is not None else "loss"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor, patience=8, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(monitor=monitor, patience=4, factor=0.5),
    ]
    model.fit(
        training_data,
        validation_data=validation_data,
        epochs=args.epochs,
        callbacks=callbacks,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output)
    print(f"Saved trained model to {args.output}")


if __name__ == "__main__":
    main()
