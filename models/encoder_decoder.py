import models.resnet as r
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, activation):
      super(EncoderDecoder, self).__init__()
      self.encoder = encoder
      self.decoder = decoder
      self.input_width = encoder.input_width
      self.input_height = encoder.input_height
      self.input_channels = encoder.input_channels
      self.activation = activation
      self.fc0 = nn.Linear(512 * 8 * 8, 512 * 4 * 4)

    def decode(self, x):
      x=self.decoder(x)
      return x

    def encode(self, x):
      x = self.encoder(x)
      x = x.view(-1, 512 * 8 * 8)
      x = self.fc0(x)
      x = x.view(-1, 512, 4, 4)
      return x

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        x = self.activation(x)
        return x
  
