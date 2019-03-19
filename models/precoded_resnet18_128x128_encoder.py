import torch
import torch.nn as nn
import models.resnet_cp as r
import models.precoder  as p

class PrecodedResnet18Encoder128x128(nn.Module):

  def __init__(self):
    super(PrecodedResnet18Encoder128x128, self).__init__()
    self.input_width = 1
    self.input_height = 19
    self.input_channels = 1
    self.output_width = 8
    self.output_height = 8
    self.output_channels = 512
    self.output_size = self.output_width * self.output_height * self.output_channels
    self.resnet = r.ResNet(r.BasicBlock, [2, 2, 2, 2]) # 11 layer resnet
    self.precoder = p.Precoder(19)
  
  def precode(self, x):
    x = self.precoder(x)
    return x

  def forward(self, x):
    x = self.precoder(x)
    x = self.resnet(x)
    return x
