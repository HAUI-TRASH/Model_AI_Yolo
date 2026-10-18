import os
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trashsmart.shared.model import load_resnet_checkpoint
from trashsmart.shared.transforms import get_eval_transform
from trashsmart.training.config import BEST_MODEL_PATH, DEVICE, DIR_PATH, INDEX_TO_NAME, NUM_CLASSES, WASTE_LABELS
from trashsmart.shared.constants import IMAGENET_MEAN, IMAGENET_STD

def get_tta_transforms():
    spatial_augs = [
        None,  # no-op
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.RandomRotation(20),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    ]
    normalize = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    result = []
    for aug in spatial_augs:
        if aug is None:
            result.append(normalize)
        else:
            result.append(transforms.Compose([aug, normalize]))
    return result

def predict_with_tta(model, img, num_augmentations=5):
    model.eval()
    tta_transforms = get_tta_transforms()
    predictions = []

    with torch.no_grad():
        for i in range(min(num_augmentations, len(tta_transforms))):
            img_tensor = tta_transforms[i](img).unsqueeze(0).to(DEVICE)
            output = model(img_tensor)
            probabilities = torch.softmax(output, dim=1)
            predictions.append(probabilities.cpu().numpy())

    return np.mean(predictions, axis=0)


def test_single_image(model, image_path, use_tta=False):
    img = Image.open(image_path).convert("RGB")

    if use_tta:
        probabilities = predict_with_tta(model, img)
    else:
        transform = get_eval_transform()
        model.eval()
        with torch.no_grad():
            img_tensor = transform(img).unsqueeze(0).to(DEVICE)
            output = model(img_tensor)
            probabilities = torch.softmax(output, dim=1).cpu().numpy()

    predicted_class = np.argmax(probabilities[0])
    predicted_label = INDEX_TO_NAME[predicted_class]
    confidence = probabilities[0][predicted_class]
    return img, probabilities, predicted_class, predicted_label, confidence


def visualize_prediction(image_path, model, true_label=None, use_tta=False):
    img, probs, pred_class, pred_label, confidence = test_single_image(model, image_path, use_tta)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.imshow(img)
    ax1.axis("off")

    tta_text = " (with TTA)" if use_tta else ""
    title = f"Predicted: {pred_label}\nConfidence: {confidence:.2%}{tta_text}"
    if true_label:
        color = "green" if pred_label == true_label else "red"
        title = f"True: {true_label}\n{title}"
        ax1.set_title(title, fontsize=14, fontweight="bold", color=color)
    else:
        ax1.set_title(title, fontsize=14, fontweight="bold")

    class_names = [INDEX_TO_NAME[i] for i in range(NUM_CLASSES)]
    colors = ["green" if i == pred_class else "skyblue" for i in range(NUM_CLASSES)]

    ax2.barh(class_names, probs[0], color=colors)
    ax2.set_xlabel("Probability", fontsize=12)
    ax2.set_title("Class Probabilities", fontsize=14, fontweight="bold")
    ax2.set_xlim(0, 1)

    for i, v in enumerate(probs[0]):
        ax2.text(v + 0.02, i, f"{v:.2%}", va="center", fontsize=10)

    plt.tight_layout()
    plt.show()
    return pred_label, confidence


def test_random_samples(num_samples_per_class=1, use_tta=False, model_path=BEST_MODEL_PATH):
    print(f"\n{'=' * 70}")
    print(f"TEST RANDOM SAMPLES - {num_samples_per_class} images/class")
    if use_tta:
        print("USING TEST TIME AUGMENTATION")
    print("=" * 70 + "\n")

    model = load_resnet_checkpoint(model_path, num_classes=NUM_CLASSES, device=DEVICE)

    results = []
    for class_name in WASTE_LABELS.keys():
        class_path = os.path.join(DIR_PATH, class_name)
        image_files = [f for f in os.listdir(class_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if not image_files:
            continue

        samples = random.sample(image_files, min(num_samples_per_class, len(image_files)))
        for img_file in samples:
            image_path = os.path.join(class_path, img_file)
            img, _, _, pred_label, confidence = test_single_image(model, image_path, use_tta)
            results.append({
                "img": img,
                "true_label": class_name,
                "predicted_label": pred_label,
                "confidence": confidence,
                "correct": class_name == pred_label,
            })
            status = "OK" if class_name == pred_label else "FAIL"
            print(f"{status} True: {class_name:15s} | Pred: {pred_label:15s} | Conf: {confidence:.2%}")

    correct = sum(1 for r in results if r["correct"])
    total = len(results)
    accuracy = correct / total if total > 0 else 0
    print(f"\n{'=' * 70}")
    print(f"Accuracy: {accuracy:.2%} ({correct}/{total})")
    print("=" * 70 + "\n")

    if not results:
        return results

    cols = min(NUM_CLASSES, len(results))
    rows = (len(results) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).flatten()

    for idx, result in enumerate(results):
        axes[idx].imshow(result["img"])
        axes[idx].axis("off")
        color = "green" if result["correct"] else "red"
        title = f"True: {result['true_label']}\nPred: {result['predicted_label']}\nConf: {result['confidence']:.2%}"
        axes[idx].set_title(title, fontsize=10, color=color, fontweight="bold")

    for idx in range(len(results), len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.show()
    return results
