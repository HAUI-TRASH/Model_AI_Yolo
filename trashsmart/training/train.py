import copy
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.config import (
    BEST_MODEL_PATH,
    DEVICE,
    EPOCHS,
    LEARNING_RATE,
    MODELS_DIR,
    STAGE1_EPOCHS,
)

from shared.model import *
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0, mode="min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, val_metric, model):
        score = -val_metric if self.mode == "min" else val_metric

        if self.best_score is None:
            self.best_score = score
            self.best_model_state = copy.deepcopy(model.state_dict())
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            self.best_model_state = copy.deepcopy(model.state_dict())

def mixup_batch(inputs, labels, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(inputs.size(0), device=inputs.device)
    mixed = lam * inputs + (1 - lam) * inputs[idx]
    return mixed, labels, labels[idx], lam


def mixup_soft_correct(outputs, labels_a, labels_b, lam):
    _, predicted = torch.max(outputs, 1)
    correct_a = (predicted == labels_a).float()
    correct_b = (predicted == labels_b).float()
    return (lam * correct_a + (1 - lam) * correct_b).sum().item()

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    pbar = tqdm(train_loader, desc="Training")
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        mixed, labels_a, labels_b, lam = mixup_batch(inputs, labels)
        optimizer.zero_grad()
        outputs = model(mixed)
        loss = lam * criterion(outputs, labels_a) + (1-lam) * criterion(outputs, labels_b)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += mixup_soft_correct(outputs, labels_a, labels_b, lam)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        pbar.set_postfix({"loss": loss.item(), "acc": 100 * correct / total})

    return running_loss / total, correct / total, np.array(all_preds), np.array(all_labels)


def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            pbar.set_postfix({"loss": loss.item(), "acc": 100 * correct / total})

    return running_loss / total, correct / total, np.array(all_preds), np.array(all_labels)


def _make_scheduler(optimizer, T_0=10):
    return CosineAnnealingWarmRestarts(
        optimizer,
        T_0=T_0,       # restart mỗi 10 epoch
        T_mult=2,      # chu kỳ tăng dần
        eta_min=1e-7,
    )


def _save_if_best(model, val_acc, best_val_acc):
    if val_acc <= best_val_acc:
        return best_val_acc

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), BEST_MODEL_PATH)
    print(f"Saved best model with val_acc: {val_acc * 100:.2f}%")
    return val_acc


def _run_stage(model, train_loader, val_loader, class_weights, lr, num_epochs, title, unfreeze=False):
    print(f"\n{'='*70}\n{title}\n{'='*70}\n")
    if unfreeze:
        model.unfreeze_base_model(freeze_bn=True)
        model.zero_grad(set_to_none=True)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.001)
    early_stopping = EarlyStopping(patience=10, mode="min")
    lr_scheduler = _make_scheduler(optimizer)
    history = {"loss":[],"val_loss":[],"accuracy":[],"val_accuracy":[]}
    best_val_acc = 0.0
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        train_loss, train_acc, _, _ = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc, _, _ = validate_epoch(model, val_loader, criterion, DEVICE)

        history["loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["accuracy"].append(train_acc)
        history["val_accuracy"].append(val_acc)

        print(f"Train Loss: {train_loss:.4f}, Train Acc (MixUp soft): {train_acc * 100:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc * 100:.2f}%")

        best_val_acc = _save_if_best(model, val_acc, best_val_acc)
        lr_scheduler.step()
        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("\nEarly stopping triggered!")
            model.load_state_dict(early_stopping.best_model_state)
            break
    return history


def train_stage1(model, train_loader, val_loader, class_weights):
    return _run_stage(model, train_loader, val_loader, class_weights,  
                    lr=LEARNING_RATE, num_epochs=STAGE1_EPOCHS,
                    title="STAGE 1: Training with frozen base model")


def train_stage2(model, train_loader, val_loader, class_weights):
    return _run_stage(model, train_loader, val_loader, class_weights,
                    lr=LEARNING_RATE/10, num_epochs=EPOCHS,
                    title="STAGE 2: Fine-tuning layer3/layer4",
                    unfreeze=True)
    
    
def combine_history(history_stage1, history_stage2):
    return {
        "loss": history_stage1["loss"] + history_stage2["loss"],
        "val_loss": history_stage1["val_loss"] + history_stage2["val_loss"],
        "accuracy": history_stage1["accuracy"] + history_stage2["accuracy"],
        "val_accuracy": history_stage1["val_accuracy"] + history_stage2["val_accuracy"],
    }
