import pickle
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trashsmart.training.config import (
    DEVICE,
    DIR_PATH,
    HISTORY_SAVE_PATH,
    INDEX_TO_NAME,
    MODELS_DIR,
    NUM_CLASSES,
)
from trashsmart.training.data_utils import get_class_weights, get_data_loaders
from trashsmart.training.evaluate import (
    evaluate_model,
    plot_confusion_matrix,
    plot_training_history,
    print_classification_report,
)
from trashsmart.training.model_utils import build_model, count_parameters
from trashsmart.training.test_utils import test_random_samples
from trashsmart.training.train import combine_history, train_stage1, train_stage2

def main():
    if not DIR_PATH.exists():
        raise FileNotFoundError(
            f"Dataset path not found: {DIR_PATH}. Set DATASET_PATH to your ImageFolder dataset."
        )
    if NUM_CLASSES == 0:
        raise ValueError(f"No class folders found in dataset path: {DIR_PATH}")

    print("=" * 70)
    print("GPU AND CONFIG CHECK")
    print("=" * 70)

    if torch.cuda.is_available():
        print(f"Found {torch.cuda.device_count()} GPU(s)")
        for i in range(torch.cuda.device_count()):
            print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
        print(f"Using device: {DEVICE}")
    else:
        print("No GPU detected. Using CPU.")

    print("=" * 70 + "\n")

    print("Loading data...")
    train_loader, val_loader, class_names = get_data_loaders()
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Classes: {class_names}\n")

    print("Computing class weights...")
    class_weights = get_class_weights(train_loader.dataset)
    print("Class weights computed\n")

    print("Building model...")
    model = build_model()
    count_parameters(model)
    print()

    history_stage1 = train_stage1(model, train_loader, val_loader, class_weights)
    history_stage2 = train_stage2(model, train_loader, val_loader, class_weights)
    history_combined = combine_history(history_stage1, history_stage2)

    print("\nPlotting training history...")
    plot_training_history(history_combined)

    print("\nEvaluating model on validation set...")
    _, y_pred_classes, y_test_classes = evaluate_model(val_loader)

    print_classification_report(y_pred_classes, y_test_classes)

    print("\nPlotting confusion matrices...")
    unique_classes = np.unique(np.concatenate([y_test_classes, y_pred_classes]))
    labels_present = sorted(unique_classes)
    target_names_present = [INDEX_TO_NAME[i] for i in labels_present]

    cm = confusion_matrix(y_test_classes, y_pred_classes, labels=labels_present)
    plot_confusion_matrix(cm, target_names_present, title="Confusion Matrix", cmap=plt.cm.OrRd)
    plot_confusion_matrix(cm, target_names_present, normalize=True, title="Normalized Confusion Matrix", cmap=plt.cm.Blues)

    print("\nTesting random samples...")
    test_random_samples(num_samples_per_class=1, use_tta=False)

    print("\nSaving history...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_SAVE_PATH, "wb") as f:
        pickle.dump(history_combined, f)
    print(f"History saved to {HISTORY_SAVE_PATH}")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
