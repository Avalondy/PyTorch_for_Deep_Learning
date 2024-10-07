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
train_features_batch.shape, train_labels_batch.shape

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

# Create a flatten layer
flatten_model = nn.Flatten()

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
