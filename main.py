import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

np.random.seed(42)
tf.random.set_seed(42)

train_dir = "drive/MyDrive/BrainTumorDataset/train"
test_dir = "drive/MyDrive/BrainTumorDataset/test"

IMG_SIZE = 256
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=True,
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False,
)


print("Class distribution in training set:")
print(train_generator.class_indices)
print(f"Total training samples: {train_generator.samples}")
print(f"Class counts: {np.bincount(train_generator.classes)}")

print("\nData is IMBALANCED - Using RECALL as primary metric")


def build_simple_cnn():
    """Architecture 1: Simple CNN - Baseline model"""
    model = keras.Sequential(
        [
            layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="Simple_CNN",
    )

    return model


def build_deeper_cnn():
    """Architecture 2: Deeper CNN - More layers for complex patterns"""
    model = keras.Sequential(
        [
            layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="Deeper_CNN",
    )

    return model


def build_batchnorm_cnn():
    """Architecture 3: CNN with Batch Normalization - Better training stability"""
    model = keras.Sequential(
        [
            layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
            layers.Conv2D(32, (3, 3), padding="same"),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), padding="same"),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), padding="same"),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(256, (3, 3), padding="same"),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(256),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.Dropout(0.5),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="BatchNorm_CNN",
    )

    return model


def train_model(model, epochs=20):
    """Compile and train a model"""
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.Recall(), keras.metrics.Precision()],
    )

    history = model.fit(
        train_generator, epochs=epochs, validation_data=test_generator, verbose=1
    )

    return history


def plot_learning_curves(history, model_name):
    """Plot training and validation curves"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    recall_key = None
    for key in history.history.keys():
        if "recall" in key.lower() and "val" not in key:
            recall_key = key
            break

    val_recall_key = "val_" + recall_key if recall_key else None

    axes[0].plot(history.history["loss"], label="Train Loss")
    axes[0].plot(history.history["val_loss"], label="Val Loss")
    axes[0].set_title(f"{model_name} - Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history.history["accuracy"], label="Train Accuracy")
    axes[1].plot(history.history["val_accuracy"], label="Val Accuracy")
    axes[1].set_title(f"{model_name} - Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    if recall_key and val_recall_key:
        axes[2].plot(history.history[recall_key], label="Train Recall")
        axes[2].plot(history.history[val_recall_key], label="Val Recall")
        axes[2].set_title(f"{model_name} - Recall")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Recall")
        axes[2].legend()
        axes[2].grid(True)
    else:
        print(f"Warning: Could not find recall metrics for {model_name}.")

    plt.tight_layout()
    plt.show()


print("\n" + "=" * 80)
print("TRAINING ARCHITECTURE 1: Simple CNN")
print("=" * 80)
model1 = build_simple_cnn()
model1.summary()
history1 = train_model(model1, epochs=20)
plot_learning_curves(history1, "Simple CNN")

print("\n" + "=" * 80)
print("TRAINING ARCHITECTURE 2: Deeper CNN")
print("=" * 80)
model2 = build_deeper_cnn()
model2.summary()
history2 = train_model(model2, epochs=20)
plot_learning_curves(history2, "Deeper CNN")

print("\n" + "=" * 80)
print("TRAINING ARCHITECTURE 3: BatchNorm CNN")
print("=" * 80)
model3 = build_batchnorm_cnn()
model3.summary()
history3 = train_model(model3, epochs=20)
plot_learning_curves(history3, "BatchNorm CNN")


def evaluate_model(model, model_name):
    """Comprehensive evaluation of the model"""
    print(f"\n{'=' * 80}")
    print(f"EVALUATION: {model_name}")
    print("=" * 80)

    test_generator.reset()
    y_pred_prob = model.predict(test_generator, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    y_true = test_generator.classes

    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    bac = balanced_accuracy_score(y_true, y_pred)

    print(f"\nMetrics:")
    print(f"Recall (Sensitivity): {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Balanced Accuracy (BAC): {bac:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Tumor", "Tumor"],
        yticklabels=["No Tumor", "Tumor"],
    )
    plt.title(f"Confusion Matrix - {model_name}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()

    return recall, precision, f1, bac


results = {}
results["Simple CNN"] = evaluate_model(model1, "Simple CNN")
results["Deeper CNN"] = evaluate_model(model2, "Deeper CNN")
results["BatchNorm CNN"] = evaluate_model(model3, "BatchNorm CNN")

print("\n" + "=" * 80)
print("FINAL COMPARISON")
print("=" * 80)
print(f"{'Model':<20} {'Recall':<12} {'Precision':<12} {'F1-Score':<12} {'BAC':<12}")
print("-" * 80)
for model_name, metrics in results.items():
    print(
        f"{model_name:<20} {metrics[0]:<12.4f} {metrics[1]:<12.4f} {metrics[2]:<12.4f} {metrics[3]:<12.4f}"
    )

best_model_name = max(results, key=lambda x: results[x][0])
print(f"\nBest model based on RECALL: {best_model_name}")
