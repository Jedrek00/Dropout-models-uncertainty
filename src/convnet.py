from typing import Optional
import torch
from torch import nn
import torch.nn.functional as F

class DropoutTypeException(Exception):
    pass


class ConvNet(nn.Module):
    def __init__(self,
                 image_channels: int = 3,
                 image_size: int = 32,
                 filters: list = [32, 64, 128],
                 kernel_sizes: list = [(3, 3), (3, 3), (3, 3)],
                 dropout_type: Optional[str] = "standard",
                 dropout_rate: float = 0.1):
        if len(filters) != 3:
            raise Exception('Use 3 filters!')
        if len(kernel_sizes) != 3:
            raise Exception('Use 3 kernel sizes!')

        self.valid = True
        if dropout_type is None:
            self.any_dropout = False
        else:
            self.any_dropout = True
            if dropout_type not in ["standard", "spatial"]:
                print(f'''Can't use "{dropout_type}" as dropout type for convnet!''')
                self.valid = False
            self.dropout_type = dropout_type

        super().__init__()

        if self.dropout_type == "standard":
            self.dropout_layer = nn.Dropout(dropout_rate)
        elif self.dropout_type == 'spatial':
            self.dropout_layer = nn.Dropout2d(dropout_rate)
        
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
