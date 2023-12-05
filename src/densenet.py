import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseNet(nn.Module):

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list[int], drop_rate: float = 0.1, use_dropconnect: bool = False):
        """
        Simple neural network with fully connected layers for image classification.
        :param input_dim: size of the input, calculated as img_size * img_size * number_of_channels.
        :param output_dim: number of classes.
        :param hidden_dims: list of integers with number of neurons in hidden layers.
        :param drop_rate: probability of dropout.
        :param use_dropoconnect: pass True to use Drop Connect, pass False to use Standard Dropout (default option).
        """
        super().__init__()
        self.drop_rate = drop_rate
        self.dropconnect = use_dropconnect
        
        current_dim = input_dim
        self.layers = nn.ModuleList()
        for hdim in hidden_dims:
            self.layers.append(nn.Linear(current_dim, hdim))
            current_dim = hdim
        self.layers.append(nn.Linear(current_dim, output_dim))
        self.dropout = nn.Dropout(p=self.drop_rate)
        self.softmax = nn.Softmax(dim=1)
        

    def forward(self, x):
        x = torch.flatten(x, 1)
        for layer in self.layers[:-1]:
            if self.dropconnect:
                mask = torch.empty_like(layer.weight).bernoulli_(1 - self.drop_rate)
                masked_weights = layer.weight * mask
                x = F.relu(F.linear(x, masked_weights, layer.bias))
            else:
                x = self.dropout(x)
                x = F.relu(layer(x))
        out = self.softmax(self.layers[-1](x))
        return out