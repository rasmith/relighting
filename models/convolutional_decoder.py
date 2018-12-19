import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvolutionalDecoder(nn.Module):
  def __init__(self):
    super(ConvolutionalDecoder, self).__init__()
    self.channels = 3 
    self.conv0 = nn.Conv2d(64, 128, kernel_size=3, stride=1,padding=1)
    self.deconv0 = nn.ConvTranspose2d(128,128,kernel_size=3,stride=3,padding=2)
    self.conv1 = nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1)
    self.deconv1 = nn.ConvTranspose2d(128,128,kernel_size=2,stride=2,padding=0)
    self.conv2 = nn.Conv2d(128, 128, kernel_size = 3, stride = 1, padding = 1)
    self.deconv2 = nn.ConvTranspose2d(128, 128, kernel_size = 2, stride = 2, padding = 0)
    self.conv3 = nn.Conv2d(128, 128, kernel_size = 3, stride = 1, padding = 1)
    self.deconv3 = nn.ConvTranspose2d(128, 128, kernel_size = 2, stride = 2, padding = 0)
    self.conv4 = nn.Conv2d(128, 128, kernel_size = 3, stride = 1, padding = 1)
    self.deconv4 = nn.ConvTranspose2d(128, 128, kernel_size = 2, stride = 2, padding = 0)
    self.conv5 = nn.Conv2d(128, 128, kernel_size = 3, stride = 1, padding = 1)
    self.deconv5 = nn.ConvTranspose2d(128, 64, kernel_size = 2, stride = 2, padding = 0)
    self.conv6 = nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1)
    self.deconv6 = nn.ConvTranspose2d(64, 32, kernel_size = 3, stride = 1, padding = 1)
    self.conv7 = nn.Conv2d(32, self.channels, kernel_size = 3, stride = 1, padding = 1)

  def forward(self, x):
    x=self.conv0(x)
    x=self.deconv0(x)
    x=self.conv1(x)
    x=self.deconv1(x)
    x=self.conv2(x)
    x=self.deconv2(x)
    x=self.conv3(x)
    x=self.deconv3(x)
    x=self.conv4(x)
    x=self.deconv4(x)
    x=self.conv5(x)
    x=self.deconv5(x)
    x=self.conv6(x)
    x=self.deconv6(x)
    x=self.conv7(x)
    return x
