import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import pandas as pd

from torch.utils.data import DataLoader, TensorDataset
import os

os.makedirs("plots", exist_ok=True)

training_start = time.time()
epoch_start = time.time()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# txt import
with open("tiny-shakespeare.txt") as f:
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





# Defining the RNN model
class CharRNN(nn.Module):
    def __init__(self, model_type, vocab_size, hidden_size, num_layers=1, fc_size=128):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, hidden_size)

        self.model_type = model_type
        #fully connected layer
        if model_type == "gru":
            self.rnn = nn.GRU(
                hidden_size,
                hidden_size,
                num_layers=num_layers,
                batch_first=True
            )
        elif model_type == "lstm":
            self.rnn = nn.LSTM(
                hidden_size,
                hidden_size,
                num_layers=num_layers,
                batch_first=True
            )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, fc_size),
            nn.ReLU(),
            nn.Linear(fc_size, vocab_size)
        )

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out
        

def count_parameters(model):
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def model_size_mb(model):
    param_size = 0

    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    buffer_size = 0

    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    return (param_size + buffer_size) / 1024**2


def estimate_rnn_complexity(model_type, seq_len, vocab_size, hidden_size, num_layers, fc_size):
    fc_ops = hidden_size * vocab_size

    if model_type == "gru":
        gates = 3
    elif model_type == "lstm":
        gates = 4
    else:
        gates = 1

    ops_per_timestep_layer = gates * (
        hidden_size * hidden_size +
        hidden_size * hidden_size
    )

    rnn_ops = seq_len * num_layers * ops_per_timestep_layer

    return rnn_ops + fc_ops


epoch_times = []
# Training the model
def train_model(model_type, seq_len, hidden_size=128, num_layers=1,fc_size=128, epochs=100):
    best_val_loss = float("inf")
    patience = 10
    patience_counter = 0
    best_model_state = None

    print(f"\nTraining {model_type.upper()} | seq_len={seq_len}")

    # rebuild dataset for sequence length
    X, y = [], []
    for i in range(len(text) - seq_len):
        seq = text[i:i+seq_len]
        label = text[i+seq_len]
        X.append([char_to_ix[c] for c in seq])
        y.append(char_to_ix[label])

    X = np.array(X)
    y = np.array(y)

     # Split dataset
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

     # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.long)
    y_train = torch.tensor(y_train, dtype=torch.long)

    X_val = torch.tensor(X_val, dtype=torch.long)
    y_val = torch.tensor(y_val, dtype=torch.long)

     # Create loaders
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

    model = CharRNN(
    model_type,
    len(chars),
    hidden_size,
    num_layers,
    fc_size
).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=10,
        gamma=0.5
)

    train_losses, val_losses, val_accs = [], [], []

    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            total_loss += loss.item()
            

        

        train_losses.append(total_loss / len(train_loader))

        # validation
        model.eval()

        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():

            for X_batch, y_batch in val_loader:

                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
            
                outputs = model(X_batch)

                loss = criterion(outputs, y_batch)

                val_loss += loss.item()

                _, preds = torch.max(outputs, 1)

                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)

        val_loss /= len(val_loader)
        acc = correct / total
        

        val_losses.append(val_loss)
        val_accs.append(acc)
        scheduler.step()
       
        print(f"Epoch {epoch+1}: loss={train_losses[-1]:.4f}, val_acc={acc:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
    total_time = time.time() - start_time

    return {
    "model": model_type,
    "seq_len": seq_len,
    "hidden_size": hidden_size,
    "layers": num_layers,
    "train_loss": train_losses,
    "val_loss": val_losses,
    "val_acc": val_accs,
    "time": total_time,
    "trained_model": model,
    "params": count_parameters(model),
    "model_size_mb": model_size_mb(model),
    "estimated_ops": estimate_rnn_complexity(
        model_type,
        seq_len,
        len(chars),
        hidden_size,
        num_layers,
        fc_size
    ),
      "fc_size": fc_size
}

experiments = [
  #  {"hidden_size": 64, "num_layers": 1, "fc_size": 64},
    {"hidden_size": 128, "num_layers": 1, "fc_size": 128},
    {"hidden_size": 256, "num_layers": 1, "fc_size": 128},
    {"hidden_size": 128, "num_layers": 2, "fc_size": 128},
   # {"hidden_size": 128, "num_layers": 1, "fc_size": 256},
]

results = []

for model_type in ["gru", "lstm"]:
    for exp in experiments:
        results.append(
            train_model(
                model_type=model_type,
                seq_len=50,
                hidden_size=exp["hidden_size"],
                num_layers=exp["num_layers"],
                fc_size=exp["fc_size"],
                epochs=100,
            )
        )
summary = []

for r in results:
    summary.append({
        "model": r["model"],
        "seq_len": r["seq_len"],
        "hidden_size": r["hidden_size"],
        "layers": r["layers"],
        "final_train_loss": r["train_loss"][-1],
        "final_val_loss": r["val_loss"][-1],
        "final_val_acc": r["val_acc"][-1],
        "time_sec": r["time"],
        "trainable_params": r["params"],
        "model_size_MB": r["model_size_mb"],
        "estimated_ops": r["estimated_ops"],
        "fc_size": r["fc_size"]
    })

df = pd.DataFrame(summary)
df.to_csv("rnn_experiment_results.csv", index=False)

print(df)




total_training_time = time.time() - training_start

print(f"Training Time: {total_training_time:.2f} sec")


best_result = max(results, key=lambda r: r["val_acc"][-1])

best_model = best_result["trained_model"]
best_seq_len = best_result["seq_len"]

test_str = "To be or not to b"

def predict_next_char(model, char_to_ix, ix_to_char, initial_str, seq_len):
    model.eval()

    with torch.no_grad():
        input_text = initial_str[-seq_len:]

        initial_input = torch.tensor(
            [char_to_ix[c] for c in input_text],
            dtype=torch.long
        ).unsqueeze(0).to(device)

        prediction = model(initial_input)

        predicted_index = torch.argmax(prediction, dim=1).item()

        return ix_to_char[predicted_index]
    
predicted_char = predict_next_char(
    best_model,
    char_to_ix,
    ix_to_char,
    test_str,
    best_seq_len
)

print(test_str[-best_seq_len:])
print(f"Predicted next character: '{predicted_char}'")


def plot_result(result):
    name = (
        f"{result['model']}_"
        f"seq{result['seq_len']}_"
        f"h{result['hidden_size']}_"
        f"layers{result['layers']}_"
        f"fc{result['fc_size']}"
    )

    plt.figure()
    plt.plot(result["train_loss"], label="train_loss")
    plt.plot(result["val_loss"], label="val_loss")
    plt.title(name + " loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"plots/{name}_loss.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(result["val_acc"], label="val_accuracy")
    plt.title(name + " accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Validation accuracy")
    plt.legend()
    plt.savefig(f"plots/{name}_accuracy.png", dpi=300, bbox_inches="tight")
    plt.close()

for r in results:
    plot_result(r)

    df.to_csv("rnn_experiment_results.csv", index=False)

