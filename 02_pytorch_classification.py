# %% Make classification dataset
from sklearn.datasets import make_circles
# Make 1000 samples
n_samples = 1000

# Create circles
X, y = make_circles(n_samples, noise=0.03, random_state=42)
print(f"First 5 samples of X:\n {X[:5]}")
print(f"First 5 samples of y:\n {y[:5]}")

# Make DataFrame of circle data
import pandas as pd
circles = pd.DataFrame({"X1": X[:, 0], "X2": X[:, 1], "label": y})
circles.head(10)

# Plot data
import matplotlib.pyplot as plt
plt.scatter(x=X[:, 0], y=X[:, 1], c=y, cmap=plt.cm.RdYlBu)
plt.show()

# Check input and output shapes
print(X.shape, y.shape)

# Turn data into tensors and create train and test splits
import torch
X = torch.from_numpy(X).type(torch.float32)
y = torch.from_numpy(y).type(torch.float32)
print(X[:5], y[:5])
print(X.dtype, y.dtype)
print(X.shape, y.shape)

# Split data into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# %% Building a model
'''
 -- 1. Setup device agonistic code so our code will run on an accelerator (GPU) if there is one
 -- 2. Construct a model (by subclassing nn.Module)
 -- 3. Define a loss function and optimizer
 -- 4. Create a training and test loop
'''

import torch
from torch import nn

# Set up device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

# Construct a model class
class CircleModelV0(nn.Module):
    def __init__(self):
        super().__init__()
        # Create 2 nn.Linear() layers capable of handling the shapes of the data
        # 1 hidden layer wit 5 neurons
        self.layer_1 = nn.Linear(in_features=2, out_features=5)
        self.layer_2 = nn.Linear(in_features=5, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_2(self.layer_1(x)) # x -> layer_1 -> layer_2

# Instantiate the model and sned it to the target device
model_0 = CircleModelV0().to(device)
print(model_0)

# Replicate the model using nn.Sequential()
model_0 = nn.Sequential(
    nn.Linear(in_features=2, out_features=5),
    nn.Linear(in_features=5, out_features=1)
).to(device)
print(model_0)
print(model_0.state_dict())

# Make predictions
with torch.inference_mode():
    untrained_predictions = model_0(X_test.to(device))
print(f"Length of untrained predictions: {len(untrained_predictions)}, Shape: {untrained_predictions.shape}")
print(f"First 10 untrained predictions:\n{untrained_predictions[:10]}")
print(f"First 10 labels:\n{y_test[:10]}")

# %% Setup loss function and optimizer
# For classification problems, better to use binary cross entropy loss or categorical cross entropy loss
# Here for loss function, we use binary cross entropy loss with logits
# For optimizers, two of the most popular and useful optimizers are SGD and Adam

# Setup loss function
# BCEWithLogitsLoss has a sigmoid activation function built in
# On the other hand, BCELoss requires inputs to have gong through a sigmoid activation function prior to being fed into BCELoss
loss_fn = nn.BCEWithLogitsLoss()

# Setup optimizer
optimizer = torch.optim.SGD(model_0.parameters(), lr=0.1)

# Calculate accuracy
def accuracy_fn(y_true, y_pred):
    return (y_pred > 0.5).eq(y_true).sum().item() / len(y_true)



# %% Train the model
'''
 -- 1. Forward pass
 -- 2. Calculate loss
 -- 3. Optimizer zero gradients
 -- 4. Backward propagation
 -- 5. Optimizer step
'''

# Raw logits -> probabilities -> predicted labels
# The model ouputs are `raw logits`, which can be converted to `probabilities` using activation functions
# such as sigmoid for binary classification or softmax for multi-class classification
# The `probabilities` are then converted to `predicted labels` by either rounding for binary classification
# or argmax() for multi-class classification

model_0.eval()
with torch.inference_mode():
    # Raw logits -> probabilities -> predicted labels
    y_logits = model_0(X_test.to(device)).sigmoid().round().squeeze()

print(y_logits[:5], y_test[:5])


# %% Building a training and testing loop
torch.manual_seed(42)
torch.cuda.manual_seed(42)
epochs = 100

# Put data to target device
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

for epoch in range(epochs):
    model_0.train()

    # Forward pass
    y_logits = model_0(X_train).squeeze()
    y_pred = y_logits.sigmoid().round() # raw logits -> probabilities -> predicted labels

    # Calculate loss/accuracy
    loss = loss_fn(y_logits, y_train) # nn.BCEWithLogitsLoss() expects raw logits as input
    acc = accuracy_fn(y_true=y_train, y_pred=y_pred)

    # Optimizer zero gradients
    optimizer.zero_grad()

    # Backward propagation
    loss.backward()

    # Optimizer step (gradient descent)
    optimizer.step()

    # Test the model
    model_0.eval()
    with torch.inference_mode():
        y_logits_test = model_0(X_test).squeeze()
        y_pred_test = y_logits_test.sigmoid().round()

        # Calculate test loss/accuracy
        test_loss = loss_fn(y_logits_test, y_test)
        test_acc = accuracy_fn(y_true=y_test, y_pred=y_pred_test)

    # Print training and test loss/accuracy
    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.4f} | Test loss: {test_loss:.5f}, Test acc: {test_acc:.4f}")


# %% Make some predictions and visualize
import requests
from pathlib import Path

# Download helper functions from Learn PyTorch repo
if Path("helper_functions.py").is_file():
    print("helper_functions.py already exists, skipping download")
else:
    print("Download helper_function.py")
    request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
    with open("helper_functions.py", "wb") as f:
        f.write(request.content)

from helper_functions import plot_predictions, plot_decision_boundary

# Plot decision boundary of the model
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_0, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_0, X_test, y_test)


# %% Improve the model (from a model perspective)
class CircleModelV1(nn.Module):
    def __init__(self):
        super().__init__()
        # 2 hidden layer wit 10 neurons each
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_3(self.layer_2(self.layer_1(x))) # x -> layer_1 -> layer_2 -> layer_3

model_1 = CircleModelV1().to(device)

# Create a loss function and an optimizer
loss_fn_1 = nn.BCEWithLogitsLoss()
optimizer_1 = torch.optim.SGD(params=model_1.parameters(), lr=0.01)

# Setup the hyper parameters
epochs = 1000
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Put data on the target device
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

for epoch in range(epochs):
    # Use training mode
    model_1.train()

    # Forward pass
    y_logits = model_1(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))

    # Calculate loss/accuracy
    loss = loss_fn_1(y_logits, y_train)
    acc = accuracy_fn(y_pred=y_pred, y_true=y_train)

    # Optimizer zero gradients
    optimizer_1.zero_grad()

    # Back propgation
    loss.backward()

    # Step the optimizier
    optimizer_1.step()

    # Testing
    model_1.eval()
    with torch.inference_mode():
        y_logits_test = model_1(X_test).squeeze()
        y_pred_test = torch.round(torch.sigmoid(y_logits_test))
        test_loss = loss_fn_1(y_logits_test, y_test)
        test_acc = accuracy_fn(y_pred=y_pred_test, y_true=y_test)

    # Print out the results
    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.4f} | Test loss: {test_loss:.5f}, Test acc: {test_acc:.4f}")


# Plot decision boundary of the model
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_1, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_1, X_test, y_test)



# %% The missing piece: non-linearity

# Recreate non-linear data
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles

n_samples = 1000

X, y = make_circles(n_samples=n_samples, noise=0.03, random_state=42)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)

# Convert data to tensors and then to train and test splits
import torch
from sklearn.model_selection import train_test_split

X = torch.from_numpy(X).type(torch.float32)
y = torch.from_numpy(y).type(torch.float32)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a model with non-linear activation functions
from torch import nn
class CircleModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)
        self.relu = nn.ReLU() # a non-linear activation function

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))

model_3 = CircleModelV2().to(device)

# Loss function and optimizer
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model_3.parameters(), lr=0.1)

# %% Training a model with non-linearity
torch.manual_seed(42)
torch.cuda.manual_seed(42)
epochs = 1000

X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

for epoch in range(epochs):
    model_3.train()
    # Forward pass
    y_logits = model_3(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))
    # Calculate loss/accuracy
    loss = loss_fn(y_logits, y_train)
    acc = accuracy_fn(y_pred=y_pred, y_true=y_train)
    # Optimizer zero gradients
    optimizer.zero_grad()
    # Back propgation
    loss.backward()
    # Step optimizer
    optimizer.step()

    # Test
    model_3.eval()
    with torch.inference_mode():
        # Forward pass
        y_logits_test = model_3(X_test).squeeze()
        y_pred_test = torch.round(torch.sigmoid(y_logits_test))
        # Calculate loss/accuracy
        loss_test = loss_fn(y_logits_test, y_test)
        acc_test = accuracy_fn(y_pred=y_pred_test, y_true=y_test)

    # Print out the results
    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.4f} | Test loss: {loss_test:.5f}, Test acc: {acc_test:.4f}")

# %% Plot decision boundaries
plt.figure(figsize=(12, 3.5))
plt.subplot(1, 3, 1)
plt.title("Train")
plot_decision_boundary(model_3, X_train, y_train)
plt.subplot(1, 3, 2)
plt.title("Test")
plot_decision_boundary(model_3, X_test, y_test)
plt.subplot(1, 3, 3)
plt.title("Train - Linear")
plot_decision_boundary(model_1, X_train, y_train)
