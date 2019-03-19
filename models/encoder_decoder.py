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
      # self.fc_channels = int(self.decoder.input_channels)
      # self.fc_input_size = \
        # self.fc_channels * decoder.input_width * decoder.input_height
      self.activation = activation
      self.fc0 = nn.Linear(512 * 8 * 8, 512 * 4 * 4)
      # self.conv0 = nn.Conv2d(512, 128, kernel_size = (5,5), padding = 0, stride = 1)
      # self.bn0 = nn.BatchNorm2d(128)
      # self.deconv1 = nn.ConvTranspose2d(256, 512, kernel_size=(5,5), padding = 2)
      # self.bn1 = nn.BatchNorm2d(512)
      # self.fc0 = nn.Linear(encoder.output_size, decoder.input_size)
      # self.conv2 = nn.Conv2d(512, 512, kernel_size = (3, 3), padding = 1)
      # self.bn2 = nn.BatchNorm2d(512)

    def decode(self, x):
      x=self.decoder(x)
      return x

    def encode(self, x):
      x = self.encoder(x)
      x = x.view(-1, 512 * 8 * 8)
      x = self.fc0(x)
      x = x.view(-1, 512, 4, 4)
      # x = self.deconv1(x)
      # x = checkpoint(lambda y: self.bn1(y), x)
      # x = self.conv2(x)
      # x = checkpoint(lambda y: self.bn2(y), x)
      return x
#         x = self.encoder(x) # -> B x 512 x 8 x 8
        # x = self.conv0(x) # -> B x 256 x 4 x 4
        # x = checkpoint(lambda y: self.bn0(y), x) # -> B x 256 x 4 x 4
        # x = x.view(-1, 128 * 4 * 4) # -> B x 256  4 * 4
        # x = checkpoint(lambda y: self.fc0(y), x) # -> B x 128 * 4 * 4
        # x = x.view(-1, 128, 4, 4)  # -> B x 128 x 4 x 4
        # x = self.deconv1(x) # -> B x 512 x 4 x 4
        # x = checkpoint(lambda y: self.bn1(y), x) # -> B x 512 x 4 x 4
#         return x

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        x = self.activation(x)
        return x
  
