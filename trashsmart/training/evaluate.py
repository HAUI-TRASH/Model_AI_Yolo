import itertools
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import classification_report, precision_score, recall_score
from tqdm import tqdm
from sklearn.metrics import f1_score

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trashsmart.shared.model import load_resnet_checkpoint
from trashsmart.training.config import BEST_MODEL_PATH, DEVICE, INDEX_TO_NAME, NUM_CLASSES


def plot_training_history(history_combined):
    epochs_range = range(1, len(history_combined["loss"]) + 1)
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Loss vs Epoch", "Accuracy vs Epoch"))

    fig.add_trace(go.Scatter(x=list(epochs_range), y=history_combined["loss"], mode="lines", name="Train Loss"), row=1, col=1)
    fig.add_trace(go.Scatter(x=list(epochs_range), y=history_combined["val_loss"], mode="lines", name="Val Loss"), row=1, col=1)
    fig.add_trace(go.Scatter(x=list(epochs_range), y=history_combined["accuracy"], mode="lines", name="Train Accuracy (MixUp soft)"), row=1, col=2)
    fig.add_trace(go.Scatter(x=list(epochs_range), y=history_combined["val_accuracy"], mode="lines", name="Val Accuracy"), row=1, col=2)

    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_xaxes(title_text="Epoch", row=1, col=2)
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Accuracy", row=1, col=2)
    fig.update_layout(template="plotly_dark", height=500, width=1200, showlegend=True, title_text="Training History")
    fig.show()


def evaluate_model(val_loader, model_path=BEST_MODEL_PATH):
    if len(val_loader.dataset) == 0:
        raise ValueError("Validation dataset is empty; add more labelled images.")
    model = load_resnet_checkpoint(model_path, num_classes=NUM_CLASSES, device=DEVICE)

    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Evaluating")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_loss = running_loss / total
    test_acc = correct / total
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    print(
        f"\nTest Loss: {test_loss:.4f}, Accuracy: {test_acc * 100:.2f}%, "
        f"Precision: {precision * 100:.2f}%, Recall: {recall * 100:.2f}%"
        f", F1 Score: {f1 * 100:.2f}%"
    )
    return model, all_preds, all_labels


def print_classification_report(y_pred_classes, y_test_classes):
    unique_classes = np.unique(np.concatenate([y_test_classes, y_pred_classes]))
    labels_present = sorted(unique_classes)
    target_names_present = [INDEX_TO_NAME[i] for i in labels_present]

    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)
    print(
        classification_report(
            y_test_classes,
            y_pred_classes,
            labels=labels_present,
            target_names=target_names_present,
            zero_division=0,
        )
    )


def plot_confusion_matrix(cm, classes, normalize=False, title="Confusion Matrix", cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title, fontsize=16, fontweight="bold")
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha="right")
    plt.yticks(tick_marks, classes)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            verticalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
            fontsize=12,
        )

    plt.tight_layout()
    plt.ylabel("True Label", fontsize=14, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=14, fontweight="bold")
    plt.show()


def main():
    """Evaluate the existing checkpoint without starting training."""
    from trashsmart.training.config import DIR_PATH
    from trashsmart.training.data_utils import get_data_loaders
    from trashsmart.pipeline.config import CLASS_NAMES

    if not DIR_PATH.is_dir():
        raise FileNotFoundError(
            f"ResNet dataset not found: {DIR_PATH}. Set DATASET_PATH to an "
            "ImageFolder dataset containing glass, metal, paper, plastic folders."
        )
    if not Path(BEST_MODEL_PATH).is_file():
        raise FileNotFoundError(f"ResNet checkpoint not found: {BEST_MODEL_PATH}")
    _, val_loader, classes = get_data_loaders()
    if classes != CLASS_NAMES:
        raise ValueError(
            f"Dataset class order {classes} does not match API class order {CLASS_NAMES}."
        )
    print(f"ResNet checkpoint: {BEST_MODEL_PATH}")
    print(f"Dataset: {DIR_PATH}")
    print(f"Validation samples: {len(val_loader.dataset)} (10% split, seed=42)")
    print(f"Classes: {classes}")
    print("Evaluation uses Resize + CenterCrop, matching training validation.")
    _, predictions, labels = evaluate_model(val_loader)
    print_classification_report(predictions, labels)


if __name__ == "__main__":
    main()
