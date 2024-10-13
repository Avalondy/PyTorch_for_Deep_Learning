# TinyVGG without data augmentation
import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torchinfo
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

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
