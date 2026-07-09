import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import matplotlib.pyplot as plt
import time
import os
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset


output_dir = "problem_1_RNN_results"
os.makedirs(output_dir, exist_ok=True)
training_start = time.time()
epoch_start = time.time()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Sample text
#text = "This is a simple example to demonstrate how to predict the next character using RNN in PyTorch."
with open("Problem_1_sequence.txt") as f:
    text = f.read()


# Creating character vocabulary
# part of the data preprocessing step for a character-level text modeling task. 
# Create mappings between characters in the text and numerical indices

#set(text): Creates a set of unique characters found in the text. The set function removes any duplicate characters.
#list(set(text)): Converts the set back into a list so that it can be sorted. 
# sorted(list(set(text))): Sorts the list of unique characters. 
chars = sorted(list(set(text)))
#This line creates a dictionary that maps each character to a unique index (integer)."
ix_to_char = {i: ch for i, ch in enumerate(chars)}
#Similar to the previous line, but in reverse. This line creates a dictionary that maps each unique index (integer) back to its corresponding character.
char_to_ix = {ch: i for i, ch in enumerate(chars)} 
chars = sorted(list(set(text)))




# Preparing the dataset
max_length = 10  # Maximum length of input sequences
X = []
y = []
for i in range(len(text) - max_length):
    sequence = text[i:i + max_length]
    label = text[i + max_length]
    X.append([char_to_ix[char] for char in sequence])
    y.append(char_to_ix[label])

X = np.array(X)
y = np.array(y)

# Splitting the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Converting data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.long)
y_train = torch.tensor(y_train, dtype=torch.long)
X_val = torch.tensor(X_val, dtype=torch.long)
y_val = torch.tensor(y_val, dtype=torch.long)

X_train = torch.tensor(X_train, dtype=torch.long).to(device)
y_train = torch.tensor(y_train, dtype=torch.long).to(device)
X_val = torch.tensor(X_val, dtype=torch.long).to(device)
y_val = torch.tensor(y_val, dtype=torch.long).to(device)

# Defining the RNN model
class CharRNNModel(nn.Module):
    def __init__(self, model_type, vocab_size, hidden_size, num_layers=1):
        super().__init__()

        self.model_type = model_type
        self.embedding = nn.Embedding(vocab_size, hidden_size)

        if model_type == "rnn":
            self.rnn = nn.RNN(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        elif model_type == "gru":
            self.rnn = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        elif model_type == "lstm":
            self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out
def train_rnn_model(model_type, seq_len, hidden_size=24, num_layers=1, epochs=100):
    X, y = [], []

    for i in range(len(text) - seq_len):
        sequence = text[i:i + seq_len]
        label = text[i + seq_len]
        X.append([char_to_ix[c] for c in sequence])
        y.append(char_to_ix[label])

    X = np.array(X)
    y = np.array(y)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train = torch.tensor(X_train, dtype=torch.long)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_val = torch.tensor(X_val, dtype=torch.long)
    y_val = torch.tensor(y_val, dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=64,
        shuffle=True
    )

    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=64,
        shuffle=False
    )

    model = CharRNNModel(
        model_type,
        len(chars),
        hidden_size,
        num_layers
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    train_losses = []
    val_losses = []
    val_accs = []
    epoch_times = []

    best_val_loss = float("inf")
    best_model_state = None
    patience = 10
    patience_counter = 0

    start_time = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()

        model.train()
        running_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                val_output = model(X_batch)
                loss = criterion(val_output, y_batch)
                val_loss += loss.item()

                _, predicted = torch.max(val_output, 1)
                correct += (predicted == y_batch).sum().item()
                total += y_batch.size(0)

        val_loss /= len(val_loader)
        val_accuracy = correct / total

        val_losses.append(val_loss)
        val_accs.append(val_accuracy)

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        print(
            f"{model_type.upper()} seq={seq_len} "
            f"Epoch {epoch+1}: "
            f"loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}, "
            f"val_acc={val_accuracy:.4f}, "
            f"time={epoch_time:.2f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    total_time = time.time() - start_time

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    model_size = param_size / 1024**2

    return {
        "model": model_type,
        "seq_len": seq_len,
        "hidden_size": hidden_size,
        "layers": num_layers,
        "train_loss": train_losses,
        "val_loss": val_losses,
        "val_acc": val_accs,
        "time": total_time,
        "avg_epoch_time": np.mean(epoch_times),
        "trained_model": model,
        "params": total_params,
        "model_size_mb": model_size
    }


rnn_results = []

for model_type in ["rnn", "gru", "lstm"]:
    for seq_len in [10, 20, 30]:
        rnn_results.append(
            train_rnn_model(
                model_type,
                seq_len,
                hidden_size=24,
                num_layers=1,
                epochs=100
            )
        )
def save_result_plots(result):
    model_name = result["model"]
    seq_len = result["seq_len"]

    plt.figure(figsize=(8, 5))
    plt.plot(result["train_loss"], label="Training Loss")
    plt.plot(result["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name.upper()} Loss Curves | Sequence Length {seq_len}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_seq{seq_len}_loss.png"))
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(result["val_acc"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{model_name.upper()} Validation Accuracy | Sequence Length {seq_len}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_seq{seq_len}_accuracy.png"))
    plt.close()
for r in rnn_results:
    save_result_plots(r)

    summary = []

for r in rnn_results:
    summary.append({
        "model": r["model"],
        "seq_len": r["seq_len"],
        "hidden_size": r["hidden_size"],
        "layers": r["layers"],
        "final_train_loss": r["train_loss"][-1],
        "final_val_loss": r["val_loss"][-1],
        "final_val_acc": r["val_acc"][-1],
        "time_sec": r["time"],
        "avg_epoch_time": r["avg_epoch_time"],
        "trainable_params": r["params"],
        "model_size_MB": r["model_size_mb"]
    })

df_rnn = pd.DataFrame(summary)
df_rnn.to_csv(os.path.join(output_dir, "rnn_gru_lstm_comparison.csv"), index=False)

print(df_rnn)


