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
        """
        Simple convolutional network for image classification.
        
        :param image_channels: number of channels of images from dataset.
        :param image_size: size of the images from the dataset (images should be a square).
        :param filters: number of filters in each conv layer.
        :param kernel_size: size of the kernel in each conv layer.
        :param drop_rate: probability of dropout.
        :param dropout_type: pass 'spatial' to use Dropout2d, pass 'standard' to use Dropout (default option).
        """
        super().__init__()
        
        self.dropout_type = dropout_type
        self.dropout_rate = dropout_rate
        if self.dropout_type is not None:
            if self.dropout_type not in ["standard", "spatial"]:
                raise DropoutTypeException(f"Can't use '{dropout_type}' as dropout type for convnet!")
            
        if self.dropout_type == "standard":
            self.dropout_layer = nn.Dropout(dropout_rate)
        elif self.dropout_type == 'spatial':
            self.dropout_layer = nn.Dropout2d(dropout_rate)

        if len(filters) != 3:
            raise Exception('Use 3 filters!')
        if len(kernel_sizes) != 3:
            raise Exception('Use 3 kernel sizes!')
        
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
        if self.dropout_type is not None:
            x = self.dropout_layer(x)

        # conv -> ReLU -> MaxPool
        x = self.pool(F.relu(self.conv2(x)))

        # Second dropout
        if self.dropout_type is not None:
            x = self.dropout_layer(x)

        # conv -> ReLU -> MaxPool
        x = self.pool(F.relu(self.conv3(x)))

        # Third dropout
        if self.dropout_type is not None:
            x = self.dropout_layer(x)

        x = torch.flatten(x, 1)  # flatten all dimensions except batch

        # Linear
        out = self.fc1(x)
        
        return out
