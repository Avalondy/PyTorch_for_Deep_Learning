# An example of a PyTorch end-to-end workflow

# %% Steps to build a model
what_were_covering = {
    1: "data (prepare and load)",
    2: "build model",
    3: "fitting the model to data (training)",
    4: "making predictions and evaluating a model (inference)",
    5: "saving and loading a model",
    6: "putting it all together",
}

import torch
from torch import (
    nn,
)  # nn contains all of PyTorch's building blocks for neural networks
import matplotlib.pyplot as plt

torch.__version__

# %% Data (preparation and loading)
# use a linear regression formula to make a straight line with known parameters
weight = 0.7
bias = 0.3

# create data points
start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias
print(X.shape, y.shape)

# %% Split data into training and test sets
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]
len(X_train), len(y_train), len(X_test), len(y_test)


# %% Visualize the data
def plot_predictions(
    train_data=X_train,
    train_labels=y_train,
    test_data=X_test,
    test_labels=y_test,
    prediction=None,
):
    """
    Plots traning, test data and compare predictions
    """
    plt.figure(figsize=(10, 7))
    plt.scatter(train_data, train_labels, color="b", s=4, label="Training data")
    plt.scatter(test_data, test_labels, color="g", s=4, label="Test data")

    if prediction is not None:
        plt.scatter(test_data, prediction, color="r", s=4, label="Predictions")

    plt.legend(prop={"size": 14})


plot_predictions()


# %% Build a model
# create a linear regression model class
class LienarRegressionModel(
    nn.Module
):  # nn.Module is a base class for all neural network modules
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(
            torch.randn(1, requires_grad=True, dtype=torch.float32)
        )
        self.bias = nn.Parameter(
            torch.randn(1, requires_grad=True, dtype=torch.float32)
        )

    # forward method to define the computaion in the model
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias


"""
PyTorch model building esstentials:
-- torch.nn - contains all of the buildings for computational graphs (a neural network can be considered a computational graph)
-- torch.nn.Parameter - what parameters should our model try and learn, often a PyTorch layer from torch.nn will set these for us
-- torch.nn.Module - The base class for all neural network modules, if you subclass it, you should overwrite forward()
-- torch.optim - this where the optimizers in PyTorch live, they will help with gradient descent
-- def forward() - All nn.Module subclasses require you to overwrite forward(), this method defines what happens in the forward computation
"""

# %% Check the contents of the model
torch.manual_seed(42)
# create an instance of the class
model_0 = LienarRegressionModel()
# check out the parameters
print(list(model_0.parameters()))
# list named parameters
print(model_0.state_dict())

# %% Making predictions using `torch.inference_mode()`
with torch.inference_mode():
    y_preds = model_0(X_test)

plot_predictions(prediction=y_preds)


# %% Training the model
"""
Loss function: A function to measure how wrong your model's predictions are to the ideal outputs, lower is better.
Optimizer: Takes into account the loss of a model and adjusts the model's parameters (e.g. weight & bias in our case) to improve the loss function.
Inside the optimizer you'll often have to set two parameters:
-- params - the model parameters you'd like to optimize, for example params=model_0.parameters()
-- lr (learning rate) - the learning rate is a hyperparameter that defines how big/small the optimizer changes the parameters with each step (a small lr results in small changes, a large lr results in large changes)
"""

# set up a loss function
loss_fn = nn.L1Loss()
# set up an optimizer (stochastic gradient descent)
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01)

# %% Build a training loop
"""
Steps:
-- 0. Loop through the data
-- 1. Forward pass - (data moves through `forward()` to make predictions)
-- 2. Calculate loss
-- 3. Optimizer zero grad
-- 4. Loss backward (back propagation) - move backwards through the network to calculate gradients of each of the parameters with respect to the loss
-- 5. Optimizer step (gradient descent) - use the optimizer to adjust the parameters of the model to minimize the loss
"""
# an epoch is a single pass through the data
epochs = 200
torch.manual_seed(42)

# Track epoch, loss and test loss
epoch_count = []
train_loss_values = []
test_loss_values = []

# 0. Loop through the data
for epoch in range(epochs):
    # Set the model to training mode
    # train mode in PyTorch sets all parameters that require gradients to require gradients
    model_0.train()

    # 1. Forward pass
    y_pred = model_0(X_train)

    # 2. Calculate loss
    loss = loss_fn(y_pred, y_train)
    # print(f"Loss: {loss}")

    # 3. Optimizer zero grad
    optimizer.zero_grad()  # need to zero out due to accumulation

    # 4. Back propagation
    loss.backward()

    # 5. Step the optimizer
    optimizer.step()  # by default how the optimizer changes will accumulate through the loop

    # Testing
    model_0.eval()  # turns off different settings in the model not needed for evaluation/testing (dropout/batch norm layers...)
    with torch.inference_mode():  # turns off gradient tracking and extra things behind the scenes
        # 1. Forward pass
        test_pred = model_0(X_test)
        # 2. Calculate loss
        test_loss = loss_fn(test_pred, y_test)

    if epoch % 10 == 0:
        epoch_count.append(epoch)
        train_loss_values.append(loss.item())
        test_loss_values.append(test_loss.item())
        print(f"Epoch {epoch} | Loss: {loss} | Test Loss: {test_loss}")
        # Print the model's state_dict
        print(model_0.state_dict())

# make predictions using `torch.inference_mode()`
with torch.inference_mode():
    y_preds_new = model_0(X_test)

plot_predictions(prediction=y_preds_new)

# %% Plotting the training and test loss
plt.plot(epoch_count, train_loss_values, label="Train Loss")
plt.plot(epoch_count, test_loss_values, label="Test Loss")
plt.legend(prop={"size": 14})
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Test Loss")
plt.show()

# %% Saving a model
"""'
-- torch.save()
-- torch.load()
-- torch.nn.Module.load_state_dict() - load a model's saved dictionary
"""
from pathlib import Path

# 1. Create a directory to save the model
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# 2. Create model save path
MODEL_NAME = "01_pytorch_workflow_model_0.pt"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# 3. Save the model state dictionary
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model_0.state_dict(), f=MODEL_SAVE_PATH)


# %% Loading a PyTorch model
# Since we saved our model's state_dict() rather the entire model, we'll create a
# new instance of our model class and load the saved state_dict() into that.

# 1. To load in a saved state_dict, we need to create a new instance of the model class
loaded_model_0 = LienarRegressionModel()

# 2. Load the saved state_dict() into the new model
loaded_model_0.load_state_dict(
    state_dict=torch.load(f=MODEL_SAVE_PATH, weights_only=True)
)
loaded_model_0.state_dict()

# 3. Make predictions using the loaded model
loaded_model_0.eval()
with torch.inference_mode():
    y_preds_loaded = loaded_model_0(X_test)

plot_predictions(prediction=y_preds_loaded)
