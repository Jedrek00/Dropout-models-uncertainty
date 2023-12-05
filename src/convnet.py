import torch
from torch import nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self,
                 use_standard_dropout: bool = False,
                 use_spatial_dropout: bool = False,
                 use_cutout_dropout: bool = False,
                 dropout_rate: float = 0.1):

        if use_standard_dropout + use_spatial_dropout + use_cutout_dropout > 1:
            raise Exception("Use only one dropout")
        elif use_standard_dropout + use_spatial_dropout + use_cutout_dropout == 1:
            self.any_dropout = True
            self.dropout_rate = dropout_rate
        else:
            self.any_dropout = False

        super().__init__()

        if use_standard_dropout:
            self.dropout1 = nn.Dropout(dropout_rate)
            self.dropout2 = nn.Dropout(dropout_rate)
            self.dropout3 = nn.Dropout(dropout_rate)

        #  Filters -> 16, 32, 64
        #  Kernels -> all 3x3
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 2 * 2, 64*2*2)
        self.fc2 = nn.Linear(64 * 2 * 2, 10)

    def forward(self, x):
        # conv -> ReLU -> MaxPool
        x = self.pool(F.relu(self.conv1(x)))

        # First dropout
        if self.any_dropout:
            x = self.dropout1(x)

        # conv -> ReLU -> MaxPool
        x = self.pool(F.relu(self.conv2(x)))

        # Second dropout
        if self.any_dropout:
            x = self.dropout2(x)

        # conv -> ReLU -> MaxPool
        x = self.pool(F.relu(self.conv3(x)))

        # Third dropout
        if self.any_dropout:
            x = self.dropout3(x)

        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        # Linear -> Softmax
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x
