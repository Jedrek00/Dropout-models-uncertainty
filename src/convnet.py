import torch
from torch import nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self,
                 image_channels: int = 3,
                 image_size: int = 32,
                 filters: list = [32, 64, 128],
                 kernel_sizes: list = [(3, 3), (3, 3), (3, 3)],
                 use_standard_dropout: bool = False,
                 use_spatial_dropout: bool = False,
                 use_cutout_dropout: bool = False,
                 dropout_rate: float = 0.1):
        if len(filters) != 3:
            raise Exception('Use 3 filters!')
        if len(kernel_sizes) != 3:
            raise Exception('Use 3 kernel sizes!')

        if use_standard_dropout + use_spatial_dropout + use_cutout_dropout > 1:
            raise Exception("Use only one dropout")
        elif use_standard_dropout + use_spatial_dropout + use_cutout_dropout == 1:
            self.any_dropout = True
            self.dropout_rate = dropout_rate
        else:
            self.any_dropout = False

        super().__init__()

        if use_standard_dropout:
            self.dropout_layer = nn.Dropout(dropout_rate)

        self.image_size = image_size

        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(image_channels, filters[0], kernel_sizes[0], padding='same')
        self.conv2 = nn.Conv2d(filters[0], filters[1], kernel_sizes[1], padding='same')
        self.conv3 = nn.Conv2d(filters[1], filters[2], kernel_sizes[2], padding='same')
        self.fc1 = nn.Linear(filters[-1] * (self.image_size//8) ** 2, 10)

    def forward(self, x):
        # conv -> ReLU -> MaxPool
        x = self.pool(F.relu(self.conv1(x)))

        # First dropout
        if self.any_dropout:
            x = self.dropout_layer(x)

        # conv -> ReLU -> MaxPool
        x = self.pool(F.relu(self.conv2(x)))

        # Second dropout
        if self.any_dropout:
            x = self.dropout_layer(x)

        # conv -> ReLU -> MaxPool
        x = self.pool(F.relu(self.conv3(x)))

        # Third dropout
        if self.any_dropout:
            x = self.dropout_layer(x)

        x = torch.flatten(x, 1)  # flatten all dimensions except batch

        # Linear -> Softmax
        x = F.softmax(self.fc1(x), dim=1)
        return x
