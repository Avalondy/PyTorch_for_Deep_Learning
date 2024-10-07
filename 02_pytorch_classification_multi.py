# Putting all together with a multi-class classification problem

# %% Create a toy multi-class dataset
import matplotlib.pyplot as plt
import torch
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from torch import nn
from torchmetrics import Accuracy

from helper_functions import plot_decision_boundary

# Set the hyperparameters for data creation
NUM_SAMPLES = 1000
NUM_CLASSES = 4
NUM_FEATURES = 2
RANDOM_SEED = 42

X_blob, y_blob = make_blobs(
    n_samples=NUM_SAMPLES,
    n_features=NUM_FEATURES,
    centers=NUM_CLASSES,
    cluster_std=1.5,
    random_state=RANDOM_SEED,
)

# Turn data into tensors
X_blob = torch.from_numpy(X_blob).type(torch.float32)
y_blob = torch.from_numpy(y_blob).type(torch.LongTensor)

# Split into train and test
X_blob_train, X_blob_test, y_blob_train, y_blob_test = train_test_split(
    X_blob, y_blob, train_size=0.8, random_state=RANDOM_SEED
)

# Plot data
plt.figure(figsize=(8, 6))
plt.scatter(X_blob[:, 0], X_blob[:, 1], c=y_blob, cmap=plt.cm.RdYlBu)


# %% Building a multi-class classification model

# Create device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"


# Build the class
class BlobModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_units=8):
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer_stack(x)


# Create an instance model and send it to the target device
model_0 = BlobModel(
    input_features=NUM_FEATURES, output_features=NUM_CLASSES, hidden_units=8
).to(device)

# Create a loss function and an optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.1)


def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    return correct / len(y_pred)


# %% Training loop and testing
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
epochs = 100

X_blob_train, X_blob_test = X_blob_train.to(device), X_blob_test.to(device)
y_blob_train, y_blob_test = y_blob_train.to(device), y_blob_test.to(device)

for epoch in range(epochs):
    model_0.train()

    # Forward pass
    y_logits = model_0(X_blob_train)
    y_pred = torch.argmax(torch.softmax(y_logits, dim=1), dim=1)

    # Calculate loss/accuracy
    loss = loss_fn(y_logits, y_blob_train)
    acc = accuracy_fn(y_pred=y_pred, y_true=y_blob_train)

    # Optimizer zero gradients
    optimizer.zero_grad()

    # Back propgation
    loss.backward()

    # Step optimizer
    optimizer.step()

    # Testing
    model_0.eval()
    with torch.inference_mode():
        # Forward pass
        y_logits_test = model_0(X_blob_test)
        y_pred_test = torch.argmax(torch.softmax(y_logits_test, dim=1), dim=1)

        # Calculate loss/accuracy
        loss_test = loss_fn(y_logits_test, y_blob_test)
        acc_test = accuracy_fn(y_pred=y_pred_test, y_true=y_blob_test)

    # Print out the results
    if epoch % 10 == 0:
        print(
            f"Epoch: {epoch} | Loss: {loss:.4f}, Acc: {acc:.4f} | Test loss: {loss_test:.4f}, Test acc: {acc_test:.4f}"
        )


# %% Plot results
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_0, X_blob_train, y_blob_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_0, X_blob_test, y_blob_test)


# %% More classification metrics
"""
-- Accuracy = (tp + tn) / (tp + tn + fp + fn)
-- Precision = tp / (tp + fp)
-- Recall = tp / (tp + fn)
-- F1-score = 2 * (precision * recall) / (precision + recall)
-- Confusion matrix
-- Classification report

tp = True Positive
tn = True Negative
fp = False Positive
fn = False Negative
"""

torchmetrics_accuracy = Accuracy(task="multiclass", num_classes=NUM_CLASSES).to(
    device
)

# Calculate accuracy
torchmetrics_accuracy(y_pred_test, y_blob_test)
