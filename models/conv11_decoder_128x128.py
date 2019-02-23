import torch
import torch.nn as nn
import models.resnet as r

class Conv11Decoder128x128(nn.Module):

  def __init__(self, activation = None):
    super(Conv11Decoder128x128, self).__init__()
    self.input_width  = 4
    self.input_height = 4
    self.input_channels = 512
    self.input_size = self.input_width * self.input_height * self.input_channels
    self.output_width = 128
    self.output_height = 128
    self.output_channels = 3
    self.output_size = self.output_width * self.output_height * self.output_channels
    self.deconv0 = nn.ConvTranspose2d(512, 512, kernel_size = 3, stride = 3,\
                                      padding = 2)
    self.bn0 = nn.BatchNorm2d(512)
    self.conv0 = nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1)
    self.bn1 = nn.BatchNorm2d(512)
    self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size = 3, stride = 2,\
                                      padding= 1, output_padding=1)
    self.bn2 = nn.BatchNorm2d(256)
    self.conv1 = nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding = 1)
    self.bn3 = nn.BatchNorm2d(256)
    self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size = 5, stride = 2,\
                                      padding= 2, output_padding=1)
    self.bn4 = nn.BatchNorm2d(128)
    self.conv2 = nn.Conv2d(128, 128, kernel_size = 3, stride = 1, padding = 1)
    self.bn5 = nn.BatchNorm2d(128)
    self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size = 5, stride = 2,\
                                      padding= 2, output_padding=1) 
    self.bn6 = nn.BatchNorm2d(64)
    self.conv3 = nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1)
    self.bn7 = nn.BatchNorm2d(64)
    self.deconv4 = nn.ConvTranspose2d(64, 32, kernel_size = 5, stride = 2,\
                                      padding= 2, output_padding=1)
    self.bn8 = nn.BatchNorm2d(32)
    self.conv4 = nn.Conv2d(32, 3, kernel_size = 3, stride = 1, padding = 1)
    self.activation = activation
  
  def forward(self, x):
    x = self.deconv0(x)
    x = self.bn0(x)
    x = self.conv0(x)
    x = self.bn1(x)
    x = self.deconv1(x)
    x = self.bn2(x)
    x = self.conv1(x)
    x = self.bn3(x)
    x = self.deconv2(x)
    x = self.bn4(x)
    x = self.conv2(x)
    x = self.bn5(x)
    x = self.deconv3(x)
    x = self.bn6(x)
    x = self.conv3(x)
    x = self.bn7(x)
    x = self.deconv4(x)
    x = self.bn8(x)
    x = self.conv4(x)
    if self.activation is not None:
      x = self.activation(x)
    return x
