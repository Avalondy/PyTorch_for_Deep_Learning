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
