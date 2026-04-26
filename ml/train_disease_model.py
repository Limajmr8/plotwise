"""
Plotwise — Crop Disease Detection Model
Author: Limawapang Jamir

Uses EfficientNetB0 (transfer learning) to classify crop leaf diseases.
Dataset: PlantVillage (Kaggle) — 54,000 images, 38 disease classes
         filtered to crops relevant to Nagaland.

Usage:
    python ml/train_disease_model.py --epochs 20 --batch 32
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report

np.random.seed(42)
tf.random.set_seed(42)

# Expanded disease classes for Nagaland crops
# Use PlantVillage dataset folder names — the data generator auto-detects them.
# To retrain: download PlantVillage from Kaggle, organize into data/train/ and data/test/
# with subfolders matching these class names. Run on Google Colab for GPU.
#
# PlantVillage classes relevant to Nagaland (26 classes):
CLASSES = [
    # Rice (Nagaland's primary crop)
    "Rice_Blast",
    "Rice_BacterialBlight",
    "Rice_BrownSpot",
    # Maize (major Kharif crop)
    "Maize_Cercospora_GrayLeafSpot",
    "Maize_CommonRust",
    "Maize_NorthernLeafBlight",
    # Potato (major vegetable crop)
    "Potato_EarlyBlight",
    "Potato_LateBlight",
    # Chilli/Pepper (widely grown)
    "Pepper_BacterialSpot",
    "Chilli_LeafCurl",
    # Tomato (major vegetable)
    "Tomato_BacterialSpot",
    "Tomato_EarlyBlight",
    "Tomato_LateBlight",
    "Tomato_LeafMold",
    "Tomato_SeptoriaLeafSpot",
    "Tomato_YellowLeafCurl",
    # Citrus (orange is important in Nagaland)
    "Orange_Haunglongbing",
    # Soybean (grown in several districts)
    "Soybean_Healthy",
    # Apple (grown in higher elevations)
    "Apple_AppleScab",
    "Apple_BlackRot",
    # Grape (emerging crop)
    "Grape_BlackRot",
    "Grape_Esca",
    # Healthy baselines
    "Healthy_Maize",
    "Healthy_Potato",
    "Healthy_Tomato",
    "Healthy_Pepper",
    "Healthy_Soybean",
]
# NOTE: The actual classes used depend on what folders exist in your data/train/ directory.
# The generator auto-detects them. The CLASSES list above is a reference for which
# PlantVillage folders to include when downloading data.

IMG_SIZE   = 224   # EfficientNet default
N_CLASSES  = len(CLASSES)


def build_model(n_classes: int, fine_tune: bool = False) -> tf.keras.Model:
    """
    EfficientNetB0 with transfer learning.
    Phase 1: Train only the classification head.
    Phase 2: Fine-tune top layers of EfficientNet.
    """
    base = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base.trainable = fine_tune

    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(n_classes, activation="softmax")
    ], name="PlotWise_DiseaseDetector")

    return model


def get_generators(data_dir: str, batch_size: int):
    # Aggressive augmentation to handle real-world field photos:
    # - Rotation: farmers take photos at any angle
    # - Shift/zoom: leaf may not be centered
    # - Brightness: field lighting varies widely
    # - Shear: perspective distortion from phone cameras
    train_gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.3,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.6, 1.4],
        channel_shift_range=30,        # Color variation (different phones)
        fill_mode="reflect",
        validation_split=0.2
    )
    test_gen = ImageDataGenerator(rescale=1./255)

    train = train_gen.flow_from_directory(
        f"{data_dir}/train",
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        class_mode="categorical",
        subset="training",
        shuffle=True
    )
    val = train_gen.flow_from_directory(
        f"{data_dir}/train",
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation",
        shuffle=False
    )
    test = test_gen.flow_from_directory(
        f"{data_dir}/test",
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False
    )
    return train, val, test


def plot_history(history, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Plotwise — Disease Model Training", fontweight="bold")

    axes[0].plot(history.history["accuracy"],     label="Train", color="steelblue", lw=2)
    axes[0].plot(history.history["val_accuracy"], label="Val",   color="tomato",    lw=2)
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history["loss"],     label="Train", color="steelblue", lw=2)
    axes[1].plot(history.history["val_loss"], label="Val",   color="tomato",    lw=2)
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def train(args):
    Path("ml/saved_models").mkdir(parents=True, exist_ok=True)
    Path("ml/results").mkdir(exist_ok=True)

    print(f"\nTensorFlow: {tf.__version__}")
    print(f"GPU: {len(tf.config.list_physical_devices('GPU')) > 0}")

    train_gen, val_gen, test_gen = get_generators(args.data_dir, args.batch)

    n_classes     = train_gen.num_classes
    active_classes = list(train_gen.class_indices.keys())
    print(f"Classes found ({n_classes}): {active_classes}")

    # Phase 1 — Train head only
    print("\n--- Phase 1: Training classification head ---")
    model = build_model(n_classes, fine_tune=False)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    cb = [
        EarlyStopping(monitor="val_accuracy", patience=8,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=4, min_lr=1e-6, verbose=1),
        ModelCheckpoint("ml/saved_models/disease_model.h5",
                        monitor="val_accuracy", save_best_only=True, verbose=1)
    ]

    h1 = model.fit(train_gen, validation_data=val_gen,
                   epochs=args.epochs // 2, callbacks=cb, verbose=1)

    # Phase 2 — Fine-tune top layers only (load Phase 1 weights)
    print("\n--- Phase 2: Fine-tuning EfficientNet top layers ---")
    model.layers[0].trainable = True
    # Freeze all but the top 20 layers of EfficientNet
    for layer in model.layers[0].layers[:-20]:
        layer.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    h2 = model.fit(train_gen, validation_data=val_gen,
                   epochs=args.epochs // 2, callbacks=cb, verbose=1)

    # Evaluate
    loss, acc = model.evaluate(test_gen, verbose=0)
    print(f"\nTest Accuracy: {acc:.4f} ({acc*100:.2f}%)")

    y_pred = np.argmax(model.predict(test_gen), axis=1)
    y_true = test_gen.classes
    print(classification_report(y_true, y_pred, target_names=active_classes))

    # Save class index map so backend loads correct labels
    idx_to_class = {v: k for k, v in train_gen.class_indices.items()}
    with open("ml/saved_models/class_indices.json", "w") as f:
        json.dump(idx_to_class, f, indent=2)
    print("Saved: ml/saved_models/class_indices.json")

    plot_history(h2, "ml/results/training_history.png")
    print("\nDone. Model saved to ml/saved_models/disease_model.h5")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--epochs",   type=int, default=20)
    p.add_argument("--batch",    type=int, default=32)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
