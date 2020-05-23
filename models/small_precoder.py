import numpy as np
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
class SmallPrecoder(nn.Module):
        def __init__(self, input_dims):
            super(SmallPrecoder, self).__init__()
            self.input_dims = input_dims
            self.fc0 = nn.Linear(self.input_dims, 49152)
            self.sig0 = nn.Sigmoid()
        def forward(self, x):
            # print(f'x.shape = {x.shape}')
            x = x.view(-1, 1, 1, self.input_dims)
            x = self.sig0(self.fc0(x))
            x = x.view(-1, 3, 128, 128)
            return x
        def estimate_size(self):
            model_parameters = filter(lambda p: p.requires_grad, self.parameters())
            params = sum([np.prod(p.size()) for p in model_parameters])
            return params / (1024*1024)
