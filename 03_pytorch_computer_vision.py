# PyTorch computer vision

# Computer vision libraries in PyTorch
"""
** `torchvision` - base domain library for PyTorch computer vision
** `torchvision.datasets` - get datasets and data loading functions
** `torchvision.models` - get pre-trained computer vision models
** `torchvision.transforms` - fuinctions for manipulating vision data (images) to be suitable for use with ML models
** `torchvision.utils.data.Dataset` - base dataset class for PyTorch computer vision
** `torchvision.utils.data.DataLoader` - creates a Python iterable over a dataset
"""

from timeit import default_timer as timer

import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

from helper_functions import accuracy_fn

# Check versions
print(torch.__version__)
print(torchvision.__version__)

# %% Getting a dataset - FashionMNIST
train_data = torchvision.datasets.FashionMNIST(
    root="data",
    train=True,  # get the training data set
    download=True,
    transform=torchvision.transforms.ToTensor(),  # transform image data to tensors
    target_transform=None,  # how do we want to transform the labels/targets?
)

test_data = torchvision.datasets.FashionMNIST(
    root="data",
    train=False,  # get the testing data set
    download=True,
    transform=torchvision.transforms.ToTensor(),  # transform image data to tensors
    target_transform=None,  # how do we want to transform the labels/targets?
)

print(len(train_data), len(test_data))

class_names = train_data.classes
class_to_idx = train_data.class_to_idx
class_to_idx

# %% See the first train data
image, label = train_data[0]
print(f"Image shape: {image.shape} -> [color_channels, height, width]")
print(f"Image label: {class_names[label]}")
plt.imshow(image.squeeze(), cmap="gray")
plt.title(class_names[label])
plt.axis(False)

# %% Plot more images
torch.manual_seed(42)
fig = plt.figure(figsize=(9, 9))
rows, cols = 4, 4
for i in range(rows * cols):
    image, label = train_data[i]
    ax = fig.add_subplot(rows, cols, i + 1)
    ax.imshow(image.squeeze(), cmap="gray")
    ax.set_title(class_names[label])
    ax.axis(False)
plt.tight_layout()

# %% See the first test data
image, label = test_data[0]
print(f"Image shape: {image.shape} -> [color_channels, height, width]")
print(f"Image label: {class_names[label]}")
plt.imshow(image.squeeze(), cmap="gray")
plt.title(class_names[label])
plt.axis(False)

# %% Plot more images
torch.manual_seed(42)
fig = plt.figure(figsize=(9, 9))
rows, cols = 4, 4
for i in range(1, rows * cols + 1):
    random_idx = torch.randint(0, len(train_data), size=[1]).item()
    # print(f"Random index: {random_idx}")
    image, label = train_data[random_idx]
    fig.add_subplot(rows, cols, i)
    plt.imshow(image.squeeze(), cmap="gray")
    plt.title(class_names[label])
    plt.axis(False)

# %% Prepare Dataloader

# Setup batch size hyperparameter
BATCH_SIZE = 32
# Turn datasets into iterables (batches)
train_dataloader = DataLoader(
    dataset=train_data, batch_size=BATCH_SIZE, shuffle=True
)
test_dataloader = DataLoader(
    dataset=test_data, batch_size=BATCH_SIZE, shuffle=False
)
print(f"Dataloaders: {train_dataloader, test_dataloader}")
print(
    f"Length of train dataloader: {len(train_dataloader)} batches of size {BATCH_SIZE}"
)
print(
    f"Length of test dataloader: {len(test_dataloader)} batches of size {BATCH_SIZE}"
)

# Check out what's inside the training dataloader, and get the first batch
train_features_batch, train_labels_batch = next(iter(train_dataloader))
print(train_features_batch.shape, train_labels_batch.shape)

# Show a sample image from the first batch
torch.manual_seed(42)
random_idx = torch.randint(0, len(train_features_batch), size=[1]).item()
image, label = train_features_batch[random_idx], train_labels_batch[random_idx]
plt.imshow(image.squeeze(), cmap="gray")
plt.title(class_names[label])
plt.axis(False)
print(f"Image shape: {image.shape}")


# %% Build a baseline model
# Start simply and add complexity when nessecary

# # Create a flatten layer
# flatten_model = nn.Flatten()
# # Get a single sample
# x = train_features_batch[0]
# # Flatten the sample
# output = flatten_model(x)
# print(output.shape)


class FashionMNISTModelV0(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            # nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_shape),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_stack(x)


# Setup model with input parameters
model_0 = FashionMNISTModelV0(
    input_shape=28 * 28,
    hidden_units=10,
    output_shape=len(class_names),
).to("cpu")
model_0

# dummy_x = torch.rand([1, 1, 28, 28])
# model_0(dummy_x)

# Setup loss, optimizer and evaluation metrics
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.1)
accuracy_fn


# %% Two metrics two track for machine learning
# 1. Model's performance (loss and accuracy etc.)
# 2. How fast it runs
def print_train_time(start: float, end: float, device: torch.device = None):
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time


start_time = timer()
end_time = timer()
print_train_time(start=start_time, end=end_time, device="cpu")


# %% Create training loop and train model on batches of data
# 1. Loop through epochs
# 2. Loop through training batches, perform training steps, calculate the train loss per batch
# 3. Loop through testing batches, perform testing steps, calculate the test loss per batch
# 4. Print out results
# 5. Time all

# Import tqdm for progress bar
from tqdm.auto import tqdm

# Set the seed and start the timer
torch.manual_seed(42)
train_time_start_on_cpu = timer()

# Set the number of epochs
epochs = 3

# Create training and test loop
for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n------")
    # Initialize training loss per batch
    train_loss = 0
    # Loop through training batches
    for batch, (X, y) in enumerate(train_dataloader):
        model_0.train()
        y_pred = model_0(X)

        # Calculate loss (per batch)
        loss = loss_fn(y_pred, y)
        train_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print the results
        if batch % 400 == 0:
            print(
                f"Looked at {batch * len(X)}/{len(train_dataloader.dataset)} samples."
            )

    # Divide total training loss by length of train dataloader
    train_loss /= len(train_dataloader)

    # Testing
    test_loss, test_acc = 0, 0
    model_0.eval()
    with torch.inference_mode():
        for X_test, y_test in test_dataloader:
            y_pred_test = model_0(X_test)

            # Calculate loss accumulatively
            test_loss += loss_fn(y_pred_test, y_test)

            # Calculate accuracy
            test_acc += accuracy_fn(
                y_pred=y_pred_test.argmax(dim=1), y_true=y_test
            )

        # Calculate the test loss/accuracy per batch
        test_loss /= len(test_dataloader)
        test_acc /= len(test_dataloader)

    # Print out the results
    print(
        f"\nTrain loss: {train_loss:.4f} | Test loss: {test_loss:.4f}, Test acc: {test_acc:.4f}"
    )

# Calculate the total training time
train_time_end_on_cpu = timer()
total_train_time_model_0 = print_train_time(
    start=train_time_start_on_cpu,
    end=train_time_end_on_cpu,
    device=str(next(model_0.parameters()).device),
)


# %% Make predictions and get model_0 results
def eval_model(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    accuracy_fn,
    device: torch.device = "cpu",
):
    """Returns a dictionary containing the results of model predicting on data_loader."""
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in tqdm(data_loader):
            X, y = X.to(device), y.to(device)
            y_pred = model(X)

            # Calculate loss accumulatively
            loss += loss_fn(y_pred, y)

            # Calculate accuracy
            acc += accuracy_fn(y_pred=y_pred.argmax(dim=1), y_true=y)

        # Calculate loss/accuracy per batch
        loss /= len(data_loader)
        acc /= len(data_loader)

    return {
        "model_name": model.__class__.__name__,  # only works when model was created with a class
        "model_loss": loss.item(),
        "model_acc": acc,
    }


# Calculate model_0 results on test dataset
model_0_results = eval_model(
    model=model_0,
    data_loader=test_dataloader,
    loss_fn=loss_fn,
    accuracy_fn=accuracy_fn,
)
model_0_results


# %% Setup device agnostic code for using GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
device


# %% Build a better model with non-linearity
class FashionMNISTModelV1(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),  # flatten inputs into a single vector
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_shape),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_stack(x)


torch.manual_seed(42)
model_1 = FashionMNISTModelV1(
    input_shape=28 * 28,
    hidden_units=10,
    output_shape=len(class_names),
).to(device)

# Setup loss, optimizer and evaluation metrics
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_1.parameters(), lr=0.1)


# %% Functionizing training and testing loops
def train_step(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    accuracy_fn,
    device: torch.device = device,
):
    """Perform a training with model trying to learn on data_loader"""
    # Initialize training loss per batch
    train_loss, train_acc = 0, 0
    model.train()
    # Loop through training batches
    for batch, (X, y) in enumerate(data_loader):
        # Put data on target device
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        # Calculate loss/accuracy (per batch)
        loss = loss_fn(y_pred, y)
        train_loss += loss
        train_acc += accuracy_fn(y_pred=y_pred.argmax(dim=1), y_true=y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Divide total training loss/accuracy by length of train dataloader
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)

    # Print out the results
    print(f"Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f}")


def test_step(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    accuracy_fn,
    device: torch.device = device,
):
    """Evaluate the trained model on the test dataset"""
    # Initialize testing loss/accuracy per batch
    test_loss, test_acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            # Put data on target device
            X, y = X.to(device), y.to(device)
            test_pred = model(X)
            # Calculate loss/accuracy (per batch)
            test_loss += loss_fn(test_pred, y)
            test_acc += accuracy_fn(y_pred=test_pred.argmax(dim=1), y_true=y)

        # Divide total testing loss/accuracy by length of test dataloader
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)

    # Print out the results
    print(f"Test loss: {test_loss:.4f} | Test acc: {test_acc:.4f}\n")


# %% Train model_1 using train_step() and test_step()
torch.manual_seed(42)
epochs = 3

# Measure the time
train_time_start_on_gpu = timer()

for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n------")
    train_step(
        model=model_1,
        data_loader=train_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        accuracy_fn=accuracy_fn,
        device=device,
    )

    test_step(
        model=model_1,
        data_loader=test_dataloader,
        loss_fn=loss_fn,
        accuracy_fn=accuracy_fn,
        device=device,
    )

train_time_end_on_gpu = timer()
total_train_time_model_1 = print_train_time(
    start=train_time_start_on_gpu,
    end=train_time_end_on_gpu,
    device=str(next(model_1.parameters()).device),
)


# %% Get model_1 results dictionary
model_1_results = eval_model(
    model=model_1,
    data_loader=test_dataloader,
    loss_fn=loss_fn,
    accuracy_fn=accuracy_fn,
    device=device,
)
model_0_results, model_1_results
