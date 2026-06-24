import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
all_preds = []
all_labels = []




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

batch_size = 128

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2023, 0.1994, 0.2010)
    )
])

from torch.utils.data import random_split

full_trainset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=False,
    transform=transform
)

trainset, valset = random_split(
    full_trainset,
    [45000, 5000],
    generator=torch.Generator().manual_seed(42)
)


testset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=False,
    transform=transform
)

trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=batch_size,
    shuffle=True
)

valloader = torch.utils.data.DataLoader(
    valset,
    batch_size=batch_size,
    shuffle=False
)

testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=batch_size,
    shuffle=False
)

model = nn.Sequential(
    nn.Conv2d(3, 16, 3, padding=1),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(2),

    nn.Conv2d(16, 32, 3, padding=1),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(2),

    nn.Conv2d(32, 64, 3, padding=1),
    nn.ReLU(inplace=True),

    nn.Conv2d(64, 64, 3, padding=1),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(2),

    nn.Conv2d(64, 128, 3, padding=1),
    nn.ReLU(inplace=True),

    nn.Conv2d(128, 128, 3, padding=1),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(2),

    nn.Conv2d(128, 128, 3, padding=1),
    nn.ReLU(inplace=True),

    nn.Conv2d(128, 128, 3, padding=1),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(2),

    nn.AdaptiveAvgPool2d((1,1)),

    nn.Flatten(),
    nn.Linear(128, 256),
    nn.ReLU(inplace=True),
    nn.Dropout(0.5),
    nn.Linear(256, 10)
)




model = model.to(device)
images, labels = next(iter(trainloader))

images = images.to(device)





print(next(model.parameters()).device)

print(model)



images, labels = next(iter(trainloader))

images = images.to(device)

with torch.no_grad():
    outputs = model(images)

print(outputs.shape)
print(next(model.parameters()).device)


model = model.to(device)
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=10,
    gamma=0.1
)

num_epochs = 30
print(next(model.parameters()).device)

train_losses = []
val_losses = []
val_accuracies = []

for epoch in range(num_epochs):

    # Training

    model.train()

    running_loss = 0.0

    for images, labels in trainloader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(trainloader)

    train_losses.append(epoch_loss)


    # Validation

    model.eval()

    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in valloader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss /= len(valloader)

    val_accuracy = 100 * correct / total

    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)
    scheduler.step()
    print(
        f"Epoch [{epoch+1}/{num_epochs}] "
        f"LR: {scheduler.get_last_lr()[0]:.6f} "
        f"Train Loss: {epoch_loss:.4f} "
        f"Val Loss: {val_loss:.4f} "
        f"Val Acc: {val_accuracy:.2f}%"
    )

# Testing Section

test_losses = []
test_accuracies = []
model.eval()

test_loss = 0.0
correct = 0
total = 0

with torch.no_grad():

    for images, labels in testloader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)
        test_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_loss /= len(testloader)

test_accuracy = 100 * correct / total

test_losses.append(test_loss)
test_accuracies.append(test_accuracy)

print(
    f"Epoch [{epoch+1}/{num_epochs}] "
    f"Train Loss: {epoch_loss:.4f} "
    f"Test Loss: {test_loss:.4f} "
    f"Test Acc: {test_accuracy:.2f}%"
)

total_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print(f"Trainable Parameters: {total_params:,}")



plt.figure(figsize=(8,5))

plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curves")

plt.legend()
plt.show()

plt.figure(figsize=(8,5))

plt.plot(val_accuracies)

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Validation Accuracy")

plt.show()

cm = confusion_matrix(all_labels, all_preds)

classes = [
    'plane',
    'car',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
]

plt.figure(figsize=(10,8))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=classes,
    yticklabels=classes
)

plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("CIFAR-10 Confusion Matrix")

plt.show()

print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))

first_layer = model[0]
weights = first_layer.weight.data.cpu()
fig, axes = plt.subplots(4, 4, figsize=(8,8))

for i, ax in enumerate(axes.flat):

    filt = weights[i]

    filt = (filt - filt.min()) / (filt.max() - filt.min())

    ax.imshow(filt.permute(1,2,0))
    ax.axis('off')

plt.suptitle("First Convolutional Layer Filters")
plt.show()

torch.save(model.state_dict(), "best_model.pth")