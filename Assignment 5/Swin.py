import os
import time
import csv
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100

from transformers import SwinForImageClassification, SwinConfig



# settings

MODEL_TYPE = "swin_scratch"
# Options:
#   "swin_tiny_pretrained"
#   "swin_small_pretrained"
#   "swin_scratch"

DATA_ROOT = "../not_uploaded"
DOWNLOAD_DATASET = False

RESULTS_DIR = "problem2_results"
RESULTS_CSV = os.path.join(RESULTS_DIR, "problem2_swin_results.csv")

NUM_CLASSES = 100
IMAGE_SIZE = 224

BATCH_SIZE = 32
EPOCHS = 5

PRETRAINED_LR = 2e-5
SCRATCH_LR = 0.001

SEED = 42
NUM_WORKERS = 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# Reproducibility

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True


# CIFAR-100 data loading

def get_dataloaders():
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
    ])

    test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
    ])

    train_dataset = CIFAR100(
        root=DATA_ROOT,
        train=True,
        download=DOWNLOAD_DATASET,
        transform=train_transform
    )

    test_dataset = CIFAR100(
        root=DATA_ROOT,
        train=False,
        download=DOWNLOAD_DATASET,
        transform=test_transform
    )

    trainloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
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

    return trainloader, testloader



# Model Setup

def freeze_swin_backbone_train_head_only(model):
    for param in model.parameters():
        param.requires_grad = False


    for param in model.classifier.parameters():
        param.requires_grad = True

    return model


def build_pretrained_swin(model_name):
    model = SwinForImageClassification.from_pretrained(
        model_name,
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True
    )

    model = freeze_swin_backbone_train_head_only(model)

    return model


def build_swin_from_scratch():
    config = SwinConfig(
        image_size=IMAGE_SIZE,
        patch_size=4,
        num_channels=3,
        num_labels=NUM_CLASSES,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        drop_path_rate=0.1
    )

    model = SwinForImageClassification(config)

    return model


def build_model():
    if MODEL_TYPE == "swin_tiny_pretrained":
        print("Loading pretrained Swin-Tiny")
        model = build_pretrained_swin("microsoft/swin-tiny-patch4-window7-224")
        learning_rate = PRETRAINED_LR
        config_name = "Swin-Tiny pretrained, frozen backbone"

    elif MODEL_TYPE == "swin_small_pretrained":
        print("Loading pretrained Swin-Small")
        model = build_pretrained_swin("microsoft/swin-small-patch4-window7-224")
        learning_rate = PRETRAINED_LR
        config_name = "Swin-Small pretrained, frozen backbone"

    elif MODEL_TYPE == "swin_scratch":
        print("Building Swin from scratch")
        model = build_swin_from_scratch()
        learning_rate = SCRATCH_LR
        config_name = "Swin-Tiny scratch"

    else:
        raise ValueError("Invalid MODEL_TYPE.")

    return model, learning_rate, config_name



# Parameter and model-size calculations

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


# Training and evaluation

def get_logits(outputs):

    return outputs.logits


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

        outputs = model(pixel_values=images)
        logits = get_logits(outputs)

        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_time = time.time() - start_time
    avg_loss = running_loss / len(trainloader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy, epoch_time


def evaluate(model, testloader, criterion):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    start_time = time.time()

    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            outputs = model(pixel_values=images)
            logits = get_logits(outputs)

            loss = criterion(logits, labels)

            running_loss += loss.item()

            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    eval_time = time.time() - start_time
    avg_loss = running_loss / len(testloader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy, eval_time



# Results saving


def save_results(row):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    file_exists = os.path.exists(RESULTS_CSV)

    with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)



# Main


def main():
    set_seed(SEED)

    print("Device:", DEVICE)
    print("Model type:", MODEL_TYPE)

    trainloader, testloader = get_dataloaders()

    model, learning_rate, config_name = build_model()
    model = model.to(DEVICE)

    total_params = count_total_parameters(model)
    trainable_params = count_trainable_parameters(model)
    size_mb = model_size_mb(model)

    print("\nConfiguration:", config_name)
    print("Total parameters:", total_params)
    print("Trainable parameters:", trainable_params)
    print("Frozen parameters:", total_params - trainable_params)
    print("Model size MB:", f"{size_mb:.4f}")
    print("Learning rate:", learning_rate)
    print("Batch size:", BATCH_SIZE)
    print("Epochs:", EPOCHS)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate
    )

    epoch_times = []
    train_losses = []
    train_accuracies = []

    total_start_time = time.time()

    for epoch in range(EPOCHS):
        train_loss, train_acc, epoch_time = train_one_epoch(
            model=model,
            trainloader=trainloader,
            criterion=criterion,
            optimizer=optimizer
        )

        epoch_times.append(epoch_time)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} "
            f"Train Acc: {train_acc:.2f}% "
            f"Time: {epoch_time:.2f}s"
        )

    total_training_time = time.time() - total_start_time
    avg_epoch_time = sum(epoch_times) / len(epoch_times)

    test_loss, test_acc, eval_time = evaluate(
        model=model,
        testloader=testloader,
        criterion=criterion
    )

    print("\nFinal Results")
    print("Config:", config_name)
    print("Final train loss:", f"{train_losses[-1]:.4f}")
    print("Final train accuracy:", f"{train_accuracies[-1]:.2f}%")
    print("Test loss:", f"{test_loss:.4f}")
    print("Test accuracy:", f"{test_acc:.2f}%")
    print("Average epoch time:", f"{avg_epoch_time:.2f}s")
    print("Total training time:", f"{total_training_time:.2f}s")
    print("Evaluation time:", f"{eval_time:.2f}s")
    print("Total parameters:", total_params)
    print("Trainable parameters:", trainable_params)
    print("Frozen parameters:", total_params - trainable_params)
    print("Model size MB:", f"{size_mb:.4f}")

    result_row = {
        "model_type": MODEL_TYPE,
        "config_name": config_name,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": learning_rate,
        "final_train_loss": train_losses[-1],
        "final_train_acc": train_accuracies[-1],
        "test_loss": test_loss,
        "test_acc": test_acc,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": total_params - trainable_params,
        "model_size_MB": size_mb,
        "avg_epoch_time_sec": avg_epoch_time,
        "total_training_time_sec": total_training_time,
        "eval_time_sec": eval_time,
        "device": str(DEVICE)
    }

    save_results(result_row)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    safe_name = MODEL_TYPE.replace("/", "_").replace(" ", "_")
    model_path = os.path.join(RESULTS_DIR, f"{safe_name}.pth")
    torch.save(model.state_dict(), model_path)

    print(f"\nSaved results to: {RESULTS_CSV}")
    print(f"Saved model to: {model_path}")


if __name__ == "__main__":
    main()