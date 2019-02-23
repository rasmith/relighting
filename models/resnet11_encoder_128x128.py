import torch
import torch.nn as nn
import models.resnet as r

class Resnet11Encoder128x128(nn.Module):

  def __init__(self):
    super(Resnet11Encoder128x128, self).__init__()
    self.input_width  = 128
    self.input_height = 128
    self.input_channels = 3
    self.output_width = 8
    self.output_height = 8
    self.output_channels = 512
    self.output_size = self.output_width * self.output_height * self.output_channels
    self.resnet = r.ResNet(r.BasicBlock, [2, 2, 2, 2]) # 11 layer resnet
  
  def forward(self, x):
    x = self.resnet(x)
    return x
