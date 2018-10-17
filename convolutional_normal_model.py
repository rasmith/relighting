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

def make_down_layer(inputs, outputs):
    return nn.Sequential(
      nn.Conv2d(inputs, outputs, kernel_size = 3, stride=1, padding=1), 
      nn.Conv2d(outputs, outputs, kernel_size = 3, stride=1, padding=1),  
      nn.ReLU(),
      nn.MaxPool2d(2, stride=2)
    )

def make_up_layer(inputs, outputs):
    return nn.Sequential(
      nn.Upsample(2, 2),
      nn.ConvTranspose2d(inputs, outputs, kernel_size = 3, stride=1, padding=1), 
      nn.ConvTranspose2d(outputs, outputs, kernel_size = 3, stride=1, padding=1), 
      nn.ReLU()
    )

class ConvolutionalNormalModel(nn.Module):
  def __init__(self):
    super(ConvolutionalNormalModel, self).__init__()
    self.conv1 = nn.Conv2d(3, 8, kernel_size=9, stride=1, padding=4)
    self.bn1 = nn.BatchNorm2d(8)
    self.conv2 = nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1)
    self.bn2 = nn.BatchNorm2d(16) 
    self.conv3 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1)
    self.bn3 = nn.BatchNorm2d(32)       
    self.res1 = ResBlock(32)
    self.res2 = ResBlock(32)
    self.deconv1 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
    self.bn4 = nn.BatchNorm2d(16)
    self.deconv2 = nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1)
    self.bn5 = nn.BatchNorm2d(8)
    self.deconv3 = nn.Conv2d(8, 3, kernel_size=9, stride=1, padding=4)

  def forward(self, X):
    h = F.relu(self.bn1(self.conv1(X)))
    h = F.relu(self.bn2(self.conv2(h)))
    h = F.relu(self.bn3(self.conv3(h)))
    h = self.res1(h)
    h = self.res2(h)
    h = F.relu(self.bn4(self.deconv1(h)))
    h = F.relu(self.bn5(self.deconv2(h)))
    y = self.deconv3(h)
    return y
