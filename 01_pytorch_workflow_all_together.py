# Putting everything learned in `01` together

import torch
from torch import nn
import matplotlib.pyplot as plt
torch.__version__

# Create device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Create some data using the linear regression formula of y = weight * X + bias
weight = 0.7
bias = 0.3

# Create range values
start = 0
end = 1
step = 0.02

# Create X and y (features and labels)
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = X * weight + bias

# Split data
train_split = int(0.8*len(X))
X_train, X_test = X[:train_split], X[train_split:]
y_train, y_test = y[:train_split], y[train_split:]

# Plot data
def plot_prediction(train_data=X_train,
                    train_labels=y_train,
                    test_data=X_test,
                    test_labels=y_test,
                    prediction=None):

    plt.figure(figsize=(10, 7))

    # Plot training data
    plt.scatter(train_data, train_labels, color="blue", s=4, label="Training data")

    # Plot test data
    plt.scatter(test_data, test_labels, color="green", s=4, label="Testing data")

    # Plot prediction
    if prediction is not None:
        plt.scatter(test_data, prediction, color="red", s=4, label="Predictions")

    # Add labels
    plt.legend(prop={"size": 14})
    plt.show()

# Call the plot function
# plot_prediction()

# Build a PyTorch linear model
class LinearRegressionV2(nn.Module):
    def __init__(self):
        super().__init__()
        # Use nn.Linear() for creating model parameters
        self.linear_layer = nn.Linear(in_features=1, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)

# Set manual seed
torch.manual_seed(42)
model_1 = LinearRegressionV2()

# Set the model to use the target device
model_1 = model_1.to(device)

# Training
''''
-- Set up a loss function
-- Set up an optimizer
-- Training loop
-- Testing loop
'''
# Loss function
loss_fn = nn.L1Loss()
# Optimizer
optimizer = torch.optim.SGD(params=model_1.parameters(), lr=0.01)

# Training loop
epochs = 200
torch.manual_seed(42)

# Put data on the target device
X_train = X_train.to(device)
y_train = y_train.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)

for epoch in range(epochs):
    # `train` mode
    model_1.train()

    # 1. Forward pass
    y_pred = model_1(X_train)

    # 2. Calculate loss
    loss = loss_fn(y_pred, y_train)

    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Back propagation
    loss.backward()

    # 5. Step optimizer
    optimizer.step()

    # Testing
    model_1.eval()
    with torch.inference_mode():
        # 1. Forward pass
        test_pred = model_1(X_test)
        # 2. Calculate loss
        test_loss = loss_fn(test_pred, y_test)

    # Print out the progress
    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Loss: {loss} | Test Loss: {test_loss}")


# Make predictions
model_1.eval()
with torch.inference_mode():
    y_preds = model_1(X_test)

plot_prediction(prediction=y_preds.cpu())


# Save the model
from pathlib import Path
SAVE_PATH = Path("models")
SAVE_PATH.mkdir(parents=True, exist_ok=True)
MODEL_NAME = "01_pytorch_workflow_model_1.pt"
MODEL_SAVE_PATH = SAVE_PATH / MODEL_NAME
torch.save(obj=model_1.state_dict(), f=MODEL_SAVE_PATH)

# Load the model
loaded_model_1 = LinearRegressionV2().to(device)
loaded_model_1.load_state_dict(state_dict=torch.load(f=MODEL_SAVE_PATH, weights_only=True))

# Make predictions using the loaded model
loaded_model_1.eval()
with torch.inference_mode():
    y_preds_loaded = loaded_model_1(X_test)

plot_prediction(prediction=y_preds_loaded.cpu())
