from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class DropoutTypeException(Exception):
    pass


class DenseNet(nn.Module):

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list[int], dropout_rate: float = 0.1, dropout_type: str = "standard"):
        """
        Simple neural network with fully connected layers for image classification.
        :param input_dim: size of the input, calculated as img_size * img_size * number_of_channels.
        :param output_dim: number of classes.
        :param hidden_dims: list of integers with number of neurons in hidden layers.
        :param drop_rate: probability of dropout.
        :param dropout_type: pass 'drop_connect' to use Drop Connect, pass 'standard' to use Standard Dropout (default option).
        """
        super().__init__()
        self.dropout_rate = dropout_rate
        if dropout_type not in ["standard", "drop_connect"]:
            raise DropoutTypeException("Dropout should be equal to 'standard' or 'drop_connect'")
        self.dropout_type = dropout_type
        
        current_dim = input_dim
        self.layers = nn.ModuleList()
        for hdim in hidden_dims:
            self.layers.append(nn.Linear(current_dim, hdim))
            current_dim = hdim
        self.layers.append(nn.Linear(current_dim, output_dim))
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.softmax = nn.Softmax(dim=1)
        

    def forward(self, x, test_dropout_rate: Optional[float] = None):
        x = torch.flatten(x, 1)
        for layer in self.layers[:-1]:
            if self.dropout_type == "drop_connect":
                if test_dropout_rate is None:
                    mask = torch.empty_like(layer.weight).bernoulli_(1 - self.dropout_rate)
                else:
                    mask = torch.empty_like(layer.weight).bernoulli_(1 - test_dropout_rate)
                masked_weights = layer.weight * mask
                x = F.relu(F.linear(x, masked_weights, layer.bias))
            else:
                if test_dropout_rate is not None:
                    self.dropout = nn.Dropout(p=test_dropout_rate)
                x = self.dropout(x)
                x = F.relu(layer(x))
        out = self.softmax(self.layers[-1](x))
        return out
