from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class DropoutTypeException(Exception):
    pass


class DenseNet(nn.Module):

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list[int], dropout_type: Optional[str] = "standard", dropout_rate: float = 0.1):
        """
        Simple neural network with fully connected layers for image classification.
        
        :param input_dim: size of the input, calculated as img_size * img_size * number_of_channels.
        :param output_dim: number of classes.
        :param hidden_dims: list of integers with number of neurons in hidden layers.
        :param dropout_type: pass 'drop_connect' to use Drop Connect, pass 'standard' to use Standard Dropout (default option).
        :param drop_rate: probability of dropout.
        """
        super().__init__()

        self.dropout_type = dropout_type
        self.dropout_rate = dropout_rate
        if self.dropout_type is not None:
            if self.dropout_type not in ["standard", "drop_connect"]:
                raise DropoutTypeException(f"Can't use '{dropout_type}' as dropout type for densenet!")
            
        if self.dropout_type == "standard":
            self.dropout = nn.Dropout(p=self.dropout_rate)
        
        current_dim = input_dim
        self.layers = nn.ModuleList()
        for hdim in hidden_dims:
            self.layers.append(nn.Linear(current_dim, hdim))
            current_dim = hdim
        self.layers.append(nn.Linear(current_dim, output_dim))        

    def forward(self, x):
        x = torch.flatten(x, 1)
        for layer in self.layers[:-1]:
            if self.dropout_type == "drop_connect":
                mask = torch.empty_like(layer.weight).bernoulli_(1 - self.dropout_rate)
                masked_weights = layer.weight * mask
                x = F.relu(F.linear(x, masked_weights, layer.bias))
            elif self.dropout_type == "standard":
                x = self.dropout(x)
                x = F.relu(layer(x))
            else:
                x = F.relu(layer(x))
        out = self.layers[-1](x)
        return out
