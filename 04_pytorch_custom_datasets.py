# Learn how to import and use custom datasets

# Depending on what you're working on, vision, text, audio, recommendation, you'll want to
# look into each of the PyTorch domain libraries for existing data loading functions and
# customizable data loading functions.


# %% Imports and device-agnostic code
import zipfile
from pathlib import Path

import requests
import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"
device

# %% Get dataset - a subset of the Food101 dataset: 3 food classes with 100 examples each
data_path = Path("data/")
image_path = data_path / "pizza_steak_sushi"

if image_path.is_dir():
    print(f"{image_path} already exists. Skipping download.")
else:
    print(f"{image_path} does not exist. Creating it.")
    image_path.mkdir(parents=True, exist_ok=True)

# Download the reduced Food101 dataset
with open(data_path / "pizza_steak_sushi.zip", "wb") as f:
    request = requests.get(
        "https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip"
    )
    print("Downloading data...")
    f.write(request.content)

# Unzip the dataset
with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip", "r") as zip_ref:
    print("Unzipping data...")
    zip_ref.extractall(image_path)


# %% Prepare the data
import os


def walk_through_dir(dir_path):
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(
            f"There are {len(dirnames)} folders and {len(filenames)} files in '{dirpath}'"
        )


# walk_through_dir(image_path)

# Setup train and test paths
train_dir = image_path / "train"
test_dir = image_path / "test"

train_dir, test_dir


# %% Visualize images
import random
from PIL import Image

random.seed(42)

# Get all images' path
image_path_list = list(image_path.glob("*/*/*.jpg"))
image_path_list

# Pick a random image
random_image_path = random.choice(image_path_list)

# Get the image class
image_class = random_image_path.parent.stem

# Open the image
img = Image.open(random_image_path)

print(f"Random image path: {random_image_path}")
print(f"Image class: {image_class}")
print(f"Image height: {img.height}")
print(f"Image width: {img.width}")
img

# %% Visualize the image using matplotlib
import numpy as np
import matplotlib.pyplot as plt

img_as_array = np.asarray(img)
plt.figure(figsize=(10, 8))
plt.imshow(img_as_array)
plt.title(f"Image class: {image_class} | Image shape: {img_as_array.shape}")
plt.axis("off")


# %% Transform data for PyTorch
"""
  1. Turn data into tensors
  2. Turn tensors into `torch.utils.data.Dataset` objects and subsequently into `torch.utils.data.DataLoader` objects
"""

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


data_transform = transforms.Compose(
    [
        # Resize to 64*64
        transforms.Resize(size=(64, 64)),
        # Flip the images randomly on the horizontal
        transforms.RandomHorizontalFlip(p=0.5),
        # Turn the image into a torch.Tensor
        transforms.ToTensor(),
    ]
)
print(data_transform(img).shape)


def plot_transformed_images(image_paths: list, transform, n=3, seed=None):
    if seed:
        random.seed(seed)
    for i in range(n):
        with Image.open(random.choice(image_paths)) as f:
            fig, ax = plt.subplots(nrows=1, ncols=2)
            ax[0].imshow(f)
            ax[0].set_title(f"Original\nSize: {f.size}")
            ax[0].axis(False)

            transformed_image = transform(f).permute(1, 2, 0)
            ax[1].imshow(transformed_image)
            ax[1].set_title(f"Transformed\nShape: {transformed_image.shape}")
            ax[1].axis(False)

            fig.suptitle(f"Class: {image_path.parent.stem}", fontsize=16)


plot_transformed_images(
    image_paths=image_path_list,
    transform=data_transform,
    n=3,
    seed=42,
)


# %% Load image data using torchvision.datasets.ImageFolder
from torchvision import datasets

train_data = datasets.ImageFolder(
    root=train_dir,
    transform=data_transform,  # a transform for the data(image)
    target_transform=None,  # a transform for the target(label)
)
test_data = datasets.ImageFolder(
    root=test_dir,
    transform=data_transform,  # a transform for the data(image)
    target_transform=None,  # a transform for the target(label)
)

train_data, test_data

# Get class names
class_names = train_data.classes
print(class_names)

# Plot the first image from the train_data dataset
plt.imshow(train_data[0][0].permute(1, 2, 0))
plt.title(class_names[train_data[0][1]])
plt.axis(False)

# Turn loaded images into DataLoaders
# DataLoader turns the dataset into iterables and we can customize the batch size
from torch.utils.data import DataLoader

BATCH_SIZE = 1

train_dataloader = DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    num_workers=os.cpu_count(),
    shuffle=True,
)
test_dataloader = DataLoader(
    dataset=test_data,
    batch_size=BATCH_SIZE,
    num_workers=os.cpu_count(),
    shuffle=False,
)
len(train_dataloader), len(test_dataloader)


# %% Create a function to display random images
from typing import List


def display_random_images(
    dataset: torch.utils.data.Dataset,
    classes: List[str] = None,
    n: int = 10,
    display_shape: bool = True,
    seed: int = None,
):
    # Adjust display if n is too high
    if n > 10:
        n = 10
        display_shape = False
        print(
            f"For display, purposes, n shouldn't be larger than 10, setting to 10 and removing shape display."
        )

    # Set the seed
    if seed:
        random.seed(seed)

    # Get random sampel indexes
    random_samples_idx = random.sample(range(len(dataset)), k=n)

    # Setup matplotlib
    plt.figure(figsize=(16, 8))

    # Plot the sampled images using matplotlib
    for i, targ_sample in enumerate(random_samples_idx):
        targ_image, targ_label = (
            dataset[targ_sample][0],
            dataset[targ_sample][1],
        )

        # Adjust tensor dimensions to match with matplotlib format
        targ_image_adjust = targ_image.permute(1, 2, 0)

        # Plot the adjusted image
        plt.subplot(1, n, i + 1)
        plt.imshow(targ_image_adjust)
        plt.axis(False)
        if classes:
            title = f"Class: {classes[targ_label]}"
            if display_shape:
                title = title + f"\nShape: {targ_image_adjust.shape}"
            plt.title(title)


display_random_images(
    dataset=train_data,
    classes=class_names,
    n=5,
    display_shape=True,
    seed=42,
)


# % Data augmentation - artificially adding diversity to the training data
