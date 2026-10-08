"""Train a small, explainable CNN for the three potato leaf classes."""
from pathlib import Path
import json

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "planttVillage"
MODEL_DIR = ROOT / "model"
MODEL_PATH = MODEL_DIR / "potato_leaf_cnn.keras"
LABELS_PATH = MODEL_DIR / "labels.json"
IMG_SIZE = (160, 160)
BATCH_SIZE = 32
SEED = 42


def main():
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {DATA_DIR}")
    # Load paths/labels first, then split images, so train/validation/test
    # each include a fair sample from every disease class.
    all_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR, image_size=IMG_SIZE, batch_size=None, shuffle=False
    )
    class_names = all_ds.class_names
    paths = np.array(all_ds.file_paths)
    labels = np.array([class_names.index(Path(p).parent.name) for p in paths])
    rng = np.random.default_rng(SEED)
    train_parts, val_parts, test_parts = [], [], []
    for class_id in range(len(class_names)):
        class_paths = paths[labels == class_id].copy()
        rng.shuffle(class_paths)
        n_train, n_val = int(len(class_paths) * 0.8), int(len(class_paths) * 0.1)
        train_parts.append(class_paths[:n_train])
        val_parts.append(class_paths[n_train:n_train + n_val])
        test_parts.append(class_paths[n_train + n_val:])
    train_paths = np.concatenate(train_parts)
    val_paths = np.concatenate(val_parts)
    test_paths = np.concatenate(test_parts)
    train_y = np.array([class_names.index(Path(p).parent.name) for p in train_paths])
    val_y = np.array([class_names.index(Path(p).parent.name) for p in val_paths])
    test_y = np.array([class_names.index(Path(p).parent.name) for p in test_paths])

    def make_dataset(file_paths, y, training=False):
        ds = tf.data.Dataset.from_tensor_slices((file_paths, y))
        if training:
            ds = ds.shuffle(len(file_paths), seed=SEED)

        def load_image(path, label):
            image = tf.io.read_file(path)
            image = tf.io.decode_image(image, channels=3, expand_animations=False)
            image.set_shape([None, None, 3])
            image = tf.image.resize(image, IMG_SIZE)
            return image, label

        return ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    train_ds = make_dataset(train_paths, train_y, training=True)
    val_ds = make_dataset(val_paths, val_y)
    test_ds = make_dataset(test_paths, test_y)

    counts = np.bincount(train_y, minlength=len(class_names))
    class_weights = {i: float(len(train_y) / (len(class_names) * counts[i])) for i in range(len(class_names))}
    print("Classes:", class_names)
    print(f"Images: train={len(train_y)}, validation={len(val_y)}, test={len(test_y)}")
    print("Class weights:", class_weights)

    # Rescaling puts pixel values in 0..1. Each Conv2D learns visual patterns;
    # pooling reduces image size; Dense produces one score per disease class.
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.1),
    ], name="training_augmentation")
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(*IMG_SIZE, 3)),
        tf.keras.layers.Rescaling(1.0 / 255),
        augmentation,
        tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(len(class_names), activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    MODEL_DIR.mkdir(exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=25, class_weight=class_weights, callbacks=callbacks)
    model = tf.keras.models.load_model(MODEL_PATH)
    loss, accuracy = model.evaluate(test_ds, verbose=0)
    print(f"Held-out test accuracy: {accuracy:.1%} (loss={loss:.4f})")
    LABELS_PATH.write_text(json.dumps(class_names, indent=2), encoding="utf-8")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved labels: {LABELS_PATH}")


if __name__ == "__main__":
    main()
