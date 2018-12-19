#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
  def __init__(self, channel):
    super(ResBlock, self).__init__()
    self.conv1 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
    self.bn1 = nn.BatchNorm2d(channel)
    self.conv2 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
    self.bn2 = nn.BatchNorm2d(channel)
  def forward(self, x):
    hidden = F.relu(self.bn1(self.conv1(x)))
    res = x + self.bn2( self.conv2(hidden))
    return res

class ConvolutionalEncoderDecoderModel(nn.Module):
  def __init__(self):
    super(ConvolutionalEncoderDecoderModel, self).__init__()
    self.conv1 = nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 4)
    self.res1x64 = ResBlock(self, 64)
    self.res2x64 = ResBlock(self, 64)
    self.res3x64 = ResBlock(self, 64)

  def encode(self, x);
    x = self.encoder(x)
    return x

  def decode(self, x):
    x = self.decoder(x)
    return x

  def forward(self, x):
    x = encode(x)
    y = decode(x)
    return y
