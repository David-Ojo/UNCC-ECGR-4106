import os
import time
import csv
import math
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100
from torchvision.models import resnet18



#Settings

MODEL_TYPE = "vit"
# Options:
#   "vit"
#   "resnet18"

DATA_ROOT = "../not_uploaded"
DOWNLOAD_DATASET = False

RESULTS_DIR = "problem1_results"
RESULTS_CSV = os.path.join(RESULTS_DIR, "problem1_results.csv")

BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001
NUM_CLASSES = 100
IMAGE_SIZE = 32
IN_CHANNELS = 3

SEED = 42
NUM_WORKERS = 2

# ViT settings
PATCH_SIZE = 4          
EMBED_DIM = 256         
NUM_BLOCKS = 4         
NUM_HEADS = 8          
MLP_RATIO = 4           
DROPOUT = 0.1



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
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5071, 0.4867, 0.4408),
            std=(0.2675, 0.2565, 0.2761)
        )
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5071, 0.4867, 0.4408),
            std=(0.2675, 0.2565, 0.2761)
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



# Vision Transformer from scratch

class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio, dropout):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm2 = nn.LayerNorm(embed_dim)

        mlp_hidden_dim = embed_dim * mlp_ratio

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        norm_x = self.norm1(x)
        attn_output, _ = self.attn(norm_x, norm_x, norm_x, need_weights=False)
        x = x + attn_output

        norm_x = self.norm2(x)
        x = x + self.mlp(norm_x)

        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size,
        patch_size,
        in_channels,
        num_classes,
        embed_dim,
        num_blocks,
        num_heads,
        mlp_ratio,
        dropout
    ):
        super().__init__()

        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )

        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for _ in range(num_blocks)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        batch_size = x.size(0)

        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        cls_output = x[:, 0]
        logits = self.head(cls_output)

        return logits



# ResNet-18 model for CIFAR-100

def build_resnet18():
    if USE_PRETRAINED_RESNET:
        model = resnet18(weights="IMAGENET1K_V1")
    else:
        model = resnet18(weights=None)

    # Modify ResNet for 32x32 CIFAR images
    model.conv1 = nn.Conv2d(
        3,
        64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False
    )

    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    return model



# Parameter and FLOP calculations


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_size_mb(model):
    total_bytes = 0

    for param in model.parameters():
        total_bytes += param.numel() * param.element_size()

    for buffer in model.buffers():
        total_bytes += buffer.numel() * buffer.element_size()

    return total_bytes / (1024 ** 2)


def vit_theoretical_params(
    img_size,
    patch_size,
    in_channels,
    num_classes,
    embed_dim,
    num_blocks,
    num_heads,
    mlp_ratio
):
    num_patches = (img_size // patch_size) ** 2
    seq_len = num_patches + 1
    mlp_hidden_dim = embed_dim * mlp_ratio


    patch_params = embed_dim * in_channels * patch_size * patch_size + embed_dim


    cls_params = embed_dim
    pos_params = seq_len * embed_dim

    attention_params = 3 * embed_dim * embed_dim + 3 * embed_dim
    attention_params += embed_dim * embed_dim + embed_dim


    norm_params = 2 * 2 * embed_dim

   
    mlp_params = embed_dim * mlp_hidden_dim + mlp_hidden_dim
    mlp_params += mlp_hidden_dim * embed_dim + embed_dim

    block_params = attention_params + norm_params + mlp_params


    final_norm_params = 2 * embed_dim
    head_params = embed_dim * num_classes + num_classes

    total_params = (
        patch_params
        + cls_params
        + pos_params
        + num_blocks * block_params
        + final_norm_params
        + head_params
    )

    return total_params


def vit_manual_flops(
    img_size,
    patch_size,
    in_channels,
    num_classes,
    embed_dim,
    num_blocks,
    num_heads,
    mlp_ratio
):
    num_patches = (img_size // patch_size) ** 2
    seq_len = num_patches + 1
    mlp_hidden_dim = embed_dim * mlp_ratio


    patch_flops = num_patches * embed_dim * in_channels * patch_size * patch_size


    qkv_flops = 3 * seq_len * embed_dim * embed_dim


    attention_scores_flops = seq_len * seq_len * embed_dim


    attention_v_flops = seq_len * seq_len * embed_dim


    attn_out_flops = seq_len * embed_dim * embed_dim

 
    mlp_flops = seq_len * embed_dim * mlp_hidden_dim
    mlp_flops += seq_len * mlp_hidden_dim * embed_dim

    block_flops = (
        qkv_flops
        + attention_scores_flops
        + attention_v_flops
        + attn_out_flops
        + mlp_flops
    )

 
    head_flops = embed_dim * num_classes

    total_flops = patch_flops + num_blocks * block_flops + head_flops

    return total_flops


def generic_conv_linear_flops(model, input_size):
   
    flops = []

    def conv_hook(module, input_tensor, output_tensor):
        output = output_tensor
        batch_size = output.shape[0]
        out_channels = output.shape[1]
        out_h = output.shape[2]
        out_w = output.shape[3]

        kernel_h = module.kernel_size[0]
        kernel_w = module.kernel_size[1]
        in_channels = module.in_channels
        groups = module.groups

        conv_per_position_flops = kernel_h * kernel_w * in_channels * out_channels / groups
        active_elements = batch_size * out_h * out_w

        flops.append(active_elements * conv_per_position_flops)

    def linear_hook(module, input_tensor, output_tensor):
        input_features = module.in_features
        output_features = module.out_features

        if len(input_tensor[0].shape) == 2:
            batch_size = input_tensor[0].shape[0]
            flops.append(batch_size * input_features * output_features)
        else:
            batch_size = input_tensor[0].shape[0]
            seq_len = input_tensor[0].shape[1]
            flops.append(batch_size * seq_len * input_features * output_features)

    hooks = []

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))

    model.eval()

    with torch.no_grad():
        dummy = torch.randn(*input_size).to(DEVICE)
        model(dummy)

    for hook in hooks:
        hook.remove()


    return sum(flops) / input_size[0]



# Training and testing

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


def evaluate(model, testloader, criterion):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / len(testloader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy



# Results saving


def save_results(row):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    file_exists = os.path.exists(RESULTS_CSV)

    with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)



# Main run

def main():
    set_seed(SEED)

    print("Device:", DEVICE)
    print("Model type:", MODEL_TYPE)

    trainloader, testloader = get_dataloaders()

    if MODEL_TYPE.lower() == "vit":
        model = VisionTransformer(
            img_size=IMAGE_SIZE,
            patch_size=PATCH_SIZE,
            in_channels=IN_CHANNELS,
            num_classes=NUM_CLASSES,
            embed_dim=EMBED_DIM,
            num_blocks=NUM_BLOCKS,
            num_heads=NUM_HEADS,
            mlp_ratio=MLP_RATIO,
            dropout=DROPOUT
        )

        config_name = (
            f"ViT_patch{PATCH_SIZE}_embed{EMBED_DIM}_"
            f"blocks{NUM_BLOCKS}_heads{NUM_HEADS}"
        )

        theoretical_params = vit_theoretical_params(
            img_size=IMAGE_SIZE,
            patch_size=PATCH_SIZE,
            in_channels=IN_CHANNELS,
            num_classes=NUM_CLASSES,
            embed_dim=EMBED_DIM,
            num_blocks=NUM_BLOCKS,
            num_heads=NUM_HEADS,
            mlp_ratio=MLP_RATIO
        )

        flops_per_forward = vit_manual_flops(
            img_size=IMAGE_SIZE,
            patch_size=PATCH_SIZE,
            in_channels=IN_CHANNELS,
            num_classes=NUM_CLASSES,
            embed_dim=EMBED_DIM,
            num_blocks=NUM_BLOCKS,
            num_heads=NUM_HEADS,
            mlp_ratio=MLP_RATIO
        )

    elif MODEL_TYPE.lower() == "resnet18":
        model = build_resnet18()

        config_name = "ResNet18_CIFAR100"

        theoretical_params = count_trainable_parameters(model)

        model = model.to(DEVICE)

        flops_per_forward = generic_conv_linear_flops(
            model=model,
            input_size=(1, 3, 32, 32)
        )

    else:
        raise ValueError("MODEL_TYPE must be either 'vit' or 'resnet18'.")

    model = model.to(DEVICE)

    actual_params = count_trainable_parameters(model)
    size_mb = model_size_mb(model)

    print("\nConfiguration:", config_name)
    print("Actual trainable parameters:", actual_params)
    print("Theoretical parameters:", theoretical_params)
    print("Model size MB:", f"{size_mb:.4f}")
    print("Estimated FLOPs per forward pass:", f"{flops_per_forward:,.0f}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=3
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
        scheduler.step()
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

    test_loss, test_acc = evaluate(
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
    print("Actual parameters:", actual_params)
    print("Theoretical parameters:", theoretical_params)
    print("Model size MB:", f"{size_mb:.4f}")
    print("Estimated FLOPs:", f"{flops_per_forward:,.0f}")

    result_row = {
        "model": MODEL_TYPE,
        "config_name": config_name,
        "patch_size": PATCH_SIZE if MODEL_TYPE.lower() == "vit" else "N/A",
        "embed_dim": EMBED_DIM if MODEL_TYPE.lower() == "vit" else "N/A",
        "num_blocks": NUM_BLOCKS if MODEL_TYPE.lower() == "vit" else "N/A",
        "num_heads": NUM_HEADS if MODEL_TYPE.lower() == "vit" else "N/A",
        "mlp_hidden_dim": EMBED_DIM * MLP_RATIO if MODEL_TYPE.lower() == "vit" else "N/A",
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "final_train_loss": train_losses[-1],
        "final_train_acc": train_accuracies[-1],
        "test_loss": test_loss,
        "test_acc": test_acc,
        "actual_params": actual_params,
        "theoretical_params": theoretical_params,
        "model_size_MB": size_mb,
        "estimated_FLOPs_per_forward": flops_per_forward,
        "avg_epoch_time_sec": avg_epoch_time,
        "total_training_time_sec": total_training_time,
        "device": str(DEVICE)
    }

    save_results(result_row)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    model_path = os.path.join(RESULTS_DIR, f"{config_name}.pth")
    torch.save(model.state_dict(), model_path)

    print(f"\nSaved results to: {RESULTS_CSV}")
    print(f"Saved model to: {model_path}")


if __name__ == "__main__":
    main()