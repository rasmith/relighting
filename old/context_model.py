#!/usr/bin/env python3
from torch import nn

class ContextModel(nn.Module):
    def __init__(self, input_dimension, hidden_dimension, output_dimension):
        super(ContextModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dimension, hidden_dimension),
            nn.ReLU(),
            nn.Linear(hidden_dimension, output_dimension),
        )

    def forward(self, x):
        x = self.model(x)
        return x
