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

    # layer 0
    self.deconv0 = nn.ConvTranspose2d(512, 512, kernel_size = 3, stride = 3,\
                                      padding = 2)
    self.bn0 = nn.BatchNorm2d(512)

    # layer 1
    self.conv1 = nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1)
    self.bn1 = nn.BatchNorm2d(512)
 
    # layer 2 
    self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size = 3, stride = 2,\
                                      padding= 1, output_padding=1)
    self.bn2 = nn.BatchNorm2d(256)

    # layer 3
    self.conv3 = nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding = 1)
    self.bn3 = nn.BatchNorm2d(256)

    # layer 4
    self.deconv4 = nn.ConvTranspose2d(256, 128, kernel_size = 5, stride = 2,\
                                      padding= 2, output_padding=1)
    self.bn4 = nn.BatchNorm2d(128)

    # layer 5
    self.conv5 = nn.Conv2d(128, 128, kernel_size = 3, stride = 1, padding = 1)
    self.bn5 = nn.BatchNorm2d(128)

    # layer 6
    self.deconv6 = nn.ConvTranspose2d(128, 64, kernel_size = 5, stride = 2,\
                                      padding= 2, output_padding=1) 
    self.bn6 = nn.BatchNorm2d(64)

    # layer 7
    self.conv7 = nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1)
    self.bn7 = nn.BatchNorm2d(64)

    # layer 8
    self.deconv8 = nn.ConvTranspose2d(64, 32, kernel_size = 5, stride = 2,\
                                      padding= 2, output_padding=1)
    self.bn8 = nn.BatchNorm2d(32)

    # layer 9
    self.conv9 = nn.Conv2d(32, 3, kernel_size = 3, stride = 1, padding = 1)

    # output
    self.activation = activation
  
  def forward(self, x):
    x = self.bn0(self.deconv0(x))
    x = self.bn1(self.conv1(x))
    x = self.bn2(self.deconv2(x))
    x = self.bn3(self.conv3(x))
    x = self.bn4(self.deconv4(x))
    x = self.bn5(self.conv5(x))
    x = self.bn6(self.deconv6(x))
    x = self.bn7(self.conv7(x))
    x = self.bn8(self.deconv8(x))
    x = self.conv9(x)
    if self.activation is not None:
      x = self.activation(x)
    return x
