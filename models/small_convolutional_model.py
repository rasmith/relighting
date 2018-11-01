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

class SmallConvolutionalModel(nn.Module):
  def __init__(self):
    super(SmallConvolutionalModel, self).__init__()
    self.conv1 = nn.Conv2d(3, 4, kernel_size=9, stride=1, padding=4)
    self.bn1 = nn.BatchNorm2d(4)

    self.res1 = ResBlock(4)

    self.bn1 = nn.BatchNorm2d(4)
    self.deconv1 = nn.Conv2d(4, 3, kernel_size=9, stride=1, padding=4)

  def forward(self, X):
    h = F.relu(self.bn1(self.conv1(X)))
    h = self.res1(h)
    h = F.relu(self.bn1(h))
    y = self.deconv1(h)
    return y
