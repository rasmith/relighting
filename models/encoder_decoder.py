import models.resnet as r
import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, activation):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.input_width = encoder.input_width
        self.input_height = encoder.input_height
        self.input_channels = encoder.input_channels
        self.activation = activation
        self.fc0 = torch.nn.Linear(encoder.output_size, decoder.input_size)

    def decode(self, x):
        x=self.decoder(x)
        return x

    def encode(self, x):
        x = self.encoder(x)
        x = self.fc0(x.view(-1, self.encoder.output_size))
        x = x.view(-1, self.decoder.input_channels, self.decoder.input_width, \
                       self.decoder.input_height)
        return x

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        x = self.activation(x)
        return x
  
