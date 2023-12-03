import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list[int], p: float = 0.1, use_dropconnect: bool = False):
        super().__init__()
        
        self.p = p
        self.dropconnect = use_dropconnect
        
        current_dim = input_dim
        self.layers = nn.ModuleList()
        for hdim in hidden_dims:
            self.layers.append(nn.Linear(current_dim, hdim))
            current_dim = hdim
        self.layers.append(nn.Linear(current_dim, output_dim))
        self.dropout = nn.Dropout(p=p)
        self.softmax = nn.Softmax(dim=1)
        

    def forward(self, x):
        x = torch.flatten(x, 1)
        for layer in self.layers[:-1]:
            if self.dropconnect:
                mask = torch.empty_like(layer.weight).bernoulli_(1 - self.p)
                masked_weights = layer.weight * mask
                x = F.relu(F.linear(x, masked_weights, layer.bias))
            else:
                x = self.dropout(x)
                x = F.relu(layer(x))
        out = self.softmax(self.layers[-1](x))
        return out
