import models.resnet as r
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResnetEncoderConvolutionalDecoder(nn.Module):
    def __init__(self):
        super(ResnetEncoderConvolutionalDecoder, self).__init__()
        self.width = 256
        self.height = 256
        self.channels = 3
        # encoder
        self.resnet = r.ResNet(r.BasicBlock, [1, 1, 1, 1])
        self.fc0 = torch.nn.Linear(512 * 16 *16, 64*4*4)
        # decoder
        # 9 convolutional layers and 7 deconvolutional layers
        # output sould be [B, C, W, H]
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


    def encode(self, x):
        x=self.resnet(x)
        x=x.view(-1, 512 * 16 *16)
        x=F.relu(self.fc0(x))
        x=x.view(-1, 64, 4, 4)
        return x

    def decode(self, x):
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

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x
  
