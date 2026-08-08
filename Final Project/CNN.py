import os
import time
import csv
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score

from PIL import Image
import glob
import yaml



MODEL_TYPE = "resnet18"
# Options:
#   "custom_cnn"
#   "resnet18"

DATA_ROOT = "../not_uploaded/Industrial Hazards.v1i.yolov8"

RESULTS_DIR = "hazard_cnn_results"
RESULTS_CSV = os.path.join(RESULTS_DIR, "hazard_cnn_results.csv")

IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4

VALID_TEST_SPLIT = 0.5
SEED = 42
NUM_WORKERS = 2

USE_PRETRAINED_RESNET = True
FREEZE_RESNET_BACKBONE = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")




def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True

# YOLO dataset
def load_class_names_from_yaml(data_yaml_path):
    with open(data_yaml_path, "r") as f:
        data = yaml.safe_load(f)

    names = data["names"]

    if isinstance(names, dict):
        names = [names[i] for i in range(len(names))]

    return names    



def get_dataloaders():
    train_images_dir = os.path.join(DATA_ROOT, "train", "images")
    train_labels_dir = os.path.join(DATA_ROOT, "train", "labels")

    valid_images_dir = os.path.join(DATA_ROOT, "valid", "images")
    valid_labels_dir = os.path.join(DATA_ROOT, "valid", "labels")

    data_yaml_path = os.path.join(DATA_ROOT, "data.yaml")

    if not os.path.exists(train_images_dir):
        raise FileNotFoundError(f"Could not find train images folder: {train_images_dir}")

    if not os.path.exists(train_labels_dir):
        raise FileNotFoundError(f"Could not find train labels folder: {train_labels_dir}")

    if not os.path.exists(valid_images_dir):
        raise FileNotFoundError(f"Could not find valid images folder: {valid_images_dir}")

    if not os.path.exists(valid_labels_dir):
        raise FileNotFoundError(f"Could not find valid labels folder: {valid_labels_dir}")

    class_names = load_class_names_from_yaml(data_yaml_path)
    num_classes = len(class_names)

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.1
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_dataset = YOLOClassificationDataset(
        images_dir=train_images_dir,
        labels_dir=train_labels_dir,
        transform=train_transform
    )

    valid_full_dataset = YOLOClassificationDataset(
        images_dir=valid_images_dir,
        labels_dir=valid_labels_dir,
        transform=eval_transform
    )

    print("Classes:", class_names)
    print("Number of classes:", num_classes)
    print("Train images:", len(train_dataset))
    print("Original valid images:", len(valid_full_dataset))

    valid_indices = list(range(len(valid_full_dataset)))

    valid_targets = [valid_full_dataset.labels[i] for i in valid_indices]

    val_indices, test_indices = train_test_split(
        valid_indices,
        test_size=VALID_TEST_SPLIT,
        random_state=SEED,
        shuffle=True,
        stratify=valid_targets
    )

    val_dataset = Subset(valid_full_dataset, val_indices)
    test_dataset = Subset(valid_full_dataset, test_indices)

    print("Validation images:", len(val_dataset))
    print("Test images:", len(test_dataset))

    trainloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    valloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    testloader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    return trainloader, valloader, testloader, class_names, num_classes




class CustomHazardCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.05),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x




def build_resnet18_model(num_classes):
    if USE_PRETRAINED_RESNET:
        print("Using ImageNet-pretrained ResNet-18")
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    else:
        print("Using ResNet-18 from scratch")
        model = resnet18(weights=None)

    if FREEZE_RESNET_BACKBONE:
        for param in model.parameters():
            param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


def build_model(num_classes):
    if MODEL_TYPE == "custom_cnn":
        model = CustomHazardCNN(num_classes)
        config_name = "Custom CNN from scratch"

    elif MODEL_TYPE == "resnet18":
        model = build_resnet18_model(num_classes)

        if USE_PRETRAINED_RESNET and FREEZE_RESNET_BACKBONE:
            config_name = "ResNet-18 pretrained frozen backbone"
        elif USE_PRETRAINED_RESNET:
            config_name = "ResNet-18 pretrained fine-tuned"
        else:
            config_name = "ResNet-18 from scratch"

    else:
        raise ValueError("MODEL_TYPE must be 'custom_cnn' or 'resnet18'.")

    return model, config_name

# YOLO model
class YOLOClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, labels_dir, transform=None, skip_empty=True):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.skip_empty = skip_empty

        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]

        all_image_paths = []
        for ext in image_extensions:
            all_image_paths.extend(glob.glob(os.path.join(images_dir, ext)))

        all_image_paths = sorted(all_image_paths)

        self.image_paths = []
        self.labels = []

        skipped_empty = 0
        skipped_missing = 0

        for image_path in all_image_paths:
            image_name = os.path.basename(image_path)
            label_name = os.path.splitext(image_name)[0] + ".txt"
            label_path = os.path.join(labels_dir, label_name)

            if not os.path.exists(label_path):
                skipped_missing += 1
                continue

            with open(label_path, "r") as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]

            if len(lines) == 0:
                skipped_empty += 1
                continue

            first_label = lines[0].split()
            class_id = int(first_label[0])

            self.image_paths.append(image_path)
            self.labels.append(class_id)

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No usable labeled images found in {images_dir}")

        print(f"Loaded {len(self.image_paths)} usable images from {images_dir}")
        print(f"Skipped empty labels: {skipped_empty}")
        print(f"Skipped missing labels: {skipped_missing}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        class_id = self.labels[idx]

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, class_id



def count_total_parameters(model):
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_size_mb(model):
    total_bytes = 0

    for param in model.parameters():
        total_bytes += param.numel() * param.element_size()

    for buffer in model.buffers():
        total_bytes += buffer.numel() * buffer.element_size()

    return total_bytes / (1024 ** 2)




def train_one_epoch(model, trainloader, criterion, optimizer):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    start_time = time.time()

    for images, labels in trainloader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_time = time.time() - start_time
    avg_loss = running_loss / len(trainloader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy, epoch_time


def evaluate(model, dataloader, criterion):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    avg_loss = running_loss / len(dataloader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy, all_labels, all_predictions




def save_results(row):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    file_exists = os.path.exists(RESULTS_CSV)

    with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def save_confusion_matrix(cm, class_names, config_name):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    plt.figure(figsize=(10, 8))
    plt.imshow(cm)
    plt.title(f"Confusion Matrix - {config_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    plt.tight_layout()

    filename = config_name.replace(" ", "_").replace("-", "_").lower()
    save_path = os.path.join(RESULTS_DIR, f"{filename}_confusion_matrix.png")

    plt.savefig(save_path)
    plt.close()

    print("Saved confusion matrix to:", save_path)


def save_loss_accuracy_plots(train_losses, val_losses, train_accs, val_accs, config_name):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    safe_name = config_name.replace(" ", "_").replace("-", "_").lower()

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss Curve - {config_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{safe_name}_loss_curve.png"))
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(train_accs, label="Training Accuracy")
    plt.plot(val_accs, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(f"Accuracy Curve - {config_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{safe_name}_accuracy_curve.png"))
    plt.close()




def main():
    set_seed(SEED)

    print("Device:", DEVICE)
    print("Model type:", MODEL_TYPE)
    print("Dataset root:", DATA_ROOT)

    trainloader, valloader, testloader, class_names, num_classes = get_dataloaders()

    model, config_name = build_model(num_classes)
    model = model.to(DEVICE)

    total_params = count_total_parameters(model)
    trainable_params = count_trainable_parameters(model)
    size_mb = model_size_mb(model)

    print("\nConfiguration:", config_name)
    print("Total parameters:", total_params)
    print("Trainable parameters:", trainable_params)
    print("Model size MB:", f"{size_mb:.4f}")
    print("Batch size:", BATCH_SIZE)
    print("Epochs:", EPOCHS)
    print("Learning rate:", LEARNING_RATE)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5
    )

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    epoch_times = []

    best_val_acc = 0.0
    best_model_path = os.path.join(RESULTS_DIR, "best_model.pth")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    total_start = time.time()

    for epoch in range(EPOCHS):
        train_loss, train_acc, epoch_time = train_one_epoch(
            model,
            trainloader,
            criterion,
            optimizer
        )

        val_loss, val_acc, _, _ = evaluate(
            model,
            valloader,
            criterion
        )

        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        epoch_times.append(epoch_time)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} "
            f"Train Acc: {train_acc:.2f}% "
            f"Val Loss: {val_loss:.4f} "
            f"Val Acc: {val_acc:.2f}% "
            f"Time: {epoch_time:.2f}s"
        )

    total_training_time = time.time() - total_start
    avg_epoch_time = sum(epoch_times) / len(epoch_times)

    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))

    test_loss, test_acc, test_labels, test_predictions = evaluate(
        model,
        testloader,
        criterion
    )

    precision = precision_score(
        test_labels,
        test_predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        test_labels,
        test_predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        test_labels,
        test_predictions,
        average="weighted",
        zero_division=0
    )

    cm = confusion_matrix(test_labels, test_predictions)

    print("\nFinal Test Results")
    print("Config:", config_name)
    print("Best validation accuracy:", f"{best_val_acc:.2f}%")
    print("Test loss:", f"{test_loss:.4f}")
    print("Test accuracy:", f"{test_acc:.2f}%")
    print("Weighted precision:", f"{precision:.4f}")
    print("Weighted recall:", f"{recall:.4f}")
    print("Weighted F1:", f"{f1:.4f}")
    print("Average epoch time:", f"{avg_epoch_time:.2f}s")
    print("Total training time:", f"{total_training_time:.2f}s")
    print("Total parameters:", total_params)
    print("Trainable parameters:", trainable_params)
    print("Model size MB:", f"{size_mb:.4f}")

    print("\nClassification Report")
    print(classification_report(
        test_labels,
        test_predictions,
        target_names=class_names,
        zero_division=0
    ))

    result_row = {
        "model_type": MODEL_TYPE,
        "config_name": config_name,
        "num_classes": num_classes,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "best_val_acc": best_val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "weighted_precision": precision,
        "weighted_recall": recall,
        "weighted_f1": f1,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "model_size_MB": size_mb,
        "avg_epoch_time_sec": avg_epoch_time,
        "total_training_time_sec": total_training_time,
        "device": str(DEVICE)
    }

    save_results(result_row)
    save_confusion_matrix(cm, class_names, config_name)
    save_loss_accuracy_plots(
        train_losses,
        val_losses,
        train_accs,
        val_accs,
        config_name
    )

    final_model_path = os.path.join(
        RESULTS_DIR,
        config_name.replace(" ", "_").replace("-", "_").lower() + ".pth"
    )

    torch.save(model.state_dict(), final_model_path)

    print("\nSaved results to:", RESULTS_CSV)
    print("Saved best model to:", best_model_path)
    print("Saved final model to:", final_model_path)


if __name__ == "__main__":
    main()