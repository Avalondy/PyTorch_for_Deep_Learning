# TinyVGG without data augmentation
import os
from pathlib import Path
from timeit import default_timer

import matplotlib.pyplot as plt
import pandas as pd
import requests
import torch
import torchinfo
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.auto import tqdm

# %% Load the data

# Create simple transform
simple_transform = transforms.Compose(
    transforms=[
        transforms.Resize(size=(64, 64)),
        transforms.ToTensor(),
    ]
)

# Load and transform data
data_path = Path("data/")
image_path = data_path / "pizza_steak_sushi"
train_dir = image_path / "train"
test_dir = image_path / "test"

train_data_simple = datasets.ImageFolder(
    root=train_dir,
    transform=simple_transform,
)
test_data_simple = datasets.ImageFolder(
    root=test_dir,
    transform=simple_transform,
)

# Turn dataset into DataLoader
BATCH_SIZE = 32
NUM_WORKERS = os.cpu_count()

train_dataloader_simple = DataLoader(
    dataset=train_data_simple,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    shuffle=True,
)
test_dataloader_simple = DataLoader(
    dataset=test_data_simple,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    shuffle=False,
)


# %% Create the model
class TinyVGG(nn.Module):
    def __init__(
        self,
        input_shape: int,
        hidden_units: int,
        output_shape: int,
    ):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_shape,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=hidden_units * 13 * 13,
                out_features=output_shape,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block_1(x)
        # print(x.shape)
        x = self.conv_block_2(x)
        # print(x.shape)
        x = self.classifier(x)
        # print(x.shape)

        return x


device = "cuda" if torch.cuda.is_available() else "cpu"
MANUAL_SEED = 42
torch.manual_seed(MANUAL_SEED)
model_0 = TinyVGG(
    input_shape=train_data_simple[0][0].shape[0],
    hidden_units=10,
    output_shape=len(train_data_simple.classes),
).to(device)


# %% Try a forward pass on a single random batch
test_random_batch = torch.rand(size=(32, 3, 64, 64), device=device)
print(model_0(test_random_batch).shape)
torchinfo.summary(model=model_0, input_size=(1, 3, 64, 64))


# %% Create train_step and test_step functions
def train_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device=device,
):
    model.train()

    # Setup train loss/accuracy initial values
    train_loss, train_acc = 0, 0

    # Loop through data loader batches
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Forward pass
        y_pred = model(X)
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)

        # Calculate loss/accuracy
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        train_acc += (y_pred_class == y).sum().item() / len(y)

        # Optimizer zero gradient
        optimizer.zero_grad()

        # Back propgation
        loss.backward()

        # Step optimizer
        optimizer.step()

    # Get average loss/accuracy per batch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc


def test_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device=device,
):
    model.eval()
    with torch.inference_mode():
        # Setup train loss/accuracy initial values
        test_loss, test_acc = 0, 0

        # Loop through data loader batches
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            # Forward pass
            y_pred = model(X)
            y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)

            # Calculate loss/accuracy
            loss = loss_fn(y_pred, y)
            test_loss += loss.item()
            test_acc += (y_pred_class == y).sum().item() / len(y)

        # Get average loss/accuracy per batch
        test_loss = test_loss / len(dataloader)
        test_acc = test_acc / len(dataloader)
        return test_loss, test_acc


# %% Create a train() function to combine train_step and test_step
def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module = torch.nn.CrossEntropyLoss(),
    epochs: int = 5,
    device: torch.device = device,
):
    # Create empty results dictionary to keep track of the results
    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }
    # Loop through the training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        print(f"Epoch: {epoch}\n------")
        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )
        test_loss, test_acc = test_step(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device,
        )
        # Print out the results
        print(
            f"Epoch {epoch + 1} | Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f} | Test loss: {test_loss:.4f} | Test acc: {test_acc:.4f}"
        )
        # Keep track of the results in the results dict
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    return results


# %% Train and evaluate model_0 using train()
torch.manual_seed(42)
torch.cuda.manual_seed(42)
NUM_EPOCHS = 50

model_0 = TinyVGG(
    input_shape=train_data_simple[0][0].shape[0],
    hidden_units=10,
    output_shape=len(train_data_simple.classes),
).to(device)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_0.parameters(), lr=0.001)

# Start the timer
start_time = default_timer()

# Train model_0
model_0_results = train(
    model=model_0,
    train_dataloader=train_dataloader_simple,
    test_dataloader=test_dataloader_simple,
    optimizer=optimizer,
    loss_fn=loss_fn,
    epochs=NUM_EPOCHS,
    device=device,
)

end_time = default_timer()
print(f"Total training time: {end_time - start_time: .3f} seconds")


# %% Plot the loss curve of model_0
def plot_loss_curves(results_dict):
    """
    Plot training curves of a results dictionary.
    """
    train_loss = results_dict["train_loss"]
    train_acc = results_dict["train_acc"]
    test_loss = results_dict["test_loss"]
    test_acc = results_dict["test_acc"]

    epochs = range(len(train_loss))

    plt.figure(figsize=(12, 5))

    # Plot the loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot the accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, label="train_accuracy")
    plt.plot(epochs, test_acc, label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()


plot_loss_curves(results_dict=model_0_results)


# %% Use data augmentation to the training dataset

# Create transform with data augmentation
train_transform_trivial = transforms.Compose(
    [
        transforms.Resize(size=(64, 64)),
        transforms.TrivialAugmentWide(num_magnitude_bins=31),
        transforms.ToTensor(),
    ]
)
test_transform_simple = transforms.Compose(
    [
        transforms.Resize(size=(64, 64)),
        transforms.ToTensor(),
    ]
)

# Create train and test datasets and DataLoaders with data augmentation
# Load in datasets
train_data_augmented = datasets.ImageFolder(
    root=train_dir,
    transform=train_transform_trivial,
)
test_data_simple = datasets.ImageFolder(
    root=test_dir,
    transform=test_transform_simple,
)

# Get dataloaders
BATCH_SIZE = 32
NUM_WORKERS = os.cpu_count()
torch.manual_seed(42)

train_dataloader_augmented = DataLoader(
    dataset=train_data_augmented,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
)
test_dataloader_simple = DataLoader(
    dataset=test_data_simple,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
)

# %% Construct and train model_1
torch.manual_seed(42)
torch.cuda.manual_seed(42)
NUM_EPOCHS = 50

model_1 = TinyVGG(
    input_shape=train_data_augmented[0][0].shape[0],
    hidden_units=10,
    output_shape=len(train_data_augmented.classes),
).to(device)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_1.parameters(), lr=0.001)

# Start the timer

start_time = default_timer()

# Train model_1
model_1_results = train(
    model=model_1,
    train_dataloader=train_dataloader_augmented,
    test_dataloader=test_dataloader_simple,
    optimizer=optimizer,
    loss_fn=loss_fn,
    epochs=NUM_EPOCHS,
    device=device,
)

end_time = default_timer()
print(f"Total training time: {end_time - start_time: .3f} seconds")
plot_loss_curves(results_dict=model_1_results)


# %% Compare model results
"""
Option 1 - Hard coding
Option 2 - PyTorch + Tensorboard
Option 3 - Weights & Biases
Option 4 - MLFlow
"""

model_0_results_df = pd.DataFrame(model_0_results)
model_1_results_df = pd.DataFrame(model_1_results)

# Plot and comapre the results
plt.figure(figsize=(10, 8))

epochs = range(len(model_0_results_df))

# Pot the train loss
plt.subplot(2, 2, 1)
plt.plot(epochs, model_0_results_df["train_loss"], label="Model 0")
plt.plot(epochs, model_1_results_df["train_loss"], label="Model 1")
plt.title("Train Loss")
plt.xlabel("Epochs")
plt.legend()
# Plot the test loss
plt.subplot(2, 2, 2)
plt.plot(epochs, model_0_results_df["test_loss"], label="Model 0")
plt.plot(epochs, model_1_results_df["test_loss"], label="Model 1")
plt.title("Test Loss")
plt.xlabel("Epochs")
plt.legend()
# Plot the train accuracy
plt.subplot(2, 2, 3)
plt.plot(epochs, model_0_results_df["train_acc"], label="Model 0")
plt.plot(epochs, model_1_results_df["train_acc"], label="Model 1")
plt.title("Train Accuracy")
plt.xlabel("Epochs")
plt.legend()
# Plot the test accuracy
plt.subplot(2, 2, 4)
plt.plot(epochs, model_0_results_df["test_acc"], label="Model 0")
plt.plot(epochs, model_1_results_df["test_acc"], label="Model 1")
plt.title("Test Accuracy")
plt.xlabel("Epochs")
plt.legend()

plt.tight_layout()
plt.show()


# %% Make prediction on a custom image - Download custom image

# Download the image if it doesn't exist
custom_image_path = data_path / "04-pizza.jpeg"
if not custom_image_path.is_file():
    with open(custom_image_path, "wb") as f:
        request = requests.get(
            "https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/04-pizza-dad.jpeg"
        )
        print(f"Downloading {custom_image_path}...")
        f.write(request.content)
else:
    print(f"{custom_image_path} already exists, skipping download.")


# Build a function to make predictions on custom images
def pred_and_plot_image(
    model: torch.nn.Module,
    image_path: Path,
    class_names: list = None,
    transform=None,
    device: torch.device = device,
):
    # Load the image into PyTorch
    target_image = torchvision.io.read_image(path=image_path)
    # Convert to float32 Tensor and add a batch size of 1 to dim=0
    target_image = target_image.type(dtype=torch.float32).unsqueeze(dim=0)
    # Normalize with respect to max=255
    target_image /= 255.0

    if transform:
        target_image = transform(target_image)

    # Put model and image on target device
    model = model.to(device)
    target_image = target_image.to(device)

    model.eval()
    with torch.inference_mode():
        target_image_pred = model(target_image)

    # Convert logits to prediction probabilities and labels
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)
    target_image_pred_label = torch.argmax(input=target_image_pred_probs, dim=1)

    # Plot the image alongside the prediction and prediction probability
    plt.imshow(target_image.squeeze().permute(1, 2, 0).cpu())
    if class_names:
        title = f"Pred: {class_names[target_image_pred_label.cpu()]} | Prob: {target_image_pred_probs.max().cpu(): .3f}"
    else:
        title = f"Pred: {target_image_pred_label.cpu()} | Prob: {target_image_pred_probs.max().cpu(): .3f}"
    plt.title(title)
    plt.axis(False)
    plt.show()


# Create transform pipeline to resize image
custom_image_transformer = transforms.Compose([transforms.Resize((64, 64))])

# Call the function `pred_and_plot_image` on the custom image
pred_and_plot_image(
    model=model_1,
    image_path=custom_image_path,
    class_names=train_data_augmented.classes,
    transform=custom_image_transformer,
    device=device,
)
