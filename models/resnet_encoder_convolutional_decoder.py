
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
        self.resnet = r.resnet18()
        self.fc0 = torch.nn.Linear(512 * 16 *16, 1024*4*4)
        # decoder
        self.conv0 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1,padding=1)
        self.conv1 = nn.Conv2d(1024,1024,kernel_size=3, stride=1,padding=1)
        self.deconv0 = nn.ConvTranspose2d(1024,1024,kernel_size=3,stride=3,padding=2)
        self.conv2 = nn.Conv2d(1024,1024,kernel_size=3,stride=1,padding=1)
        self.deconv1 = nn.ConvTranspose2d(1024,512,kernel_size=2,stride=2,padding=0)
        self.conv3 = nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1)
        self.deconv2 = nn.ConvTranspose2d(512, 512, kernel_size = 2, stride = 2, padding = 0)
        self.conv4 = nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1)
        self.deconv3 = nn.ConvTranspose2d(512, 256, kernel_size = 2, stride = 2, padding = 0)
        self.conv5 = nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding = 1)
        self.deconv4 = nn.ConvTranspose2d(256, 128, kernel_size = 2, stride = 2, padding = 0)
        self.conv6 = nn.Conv2d(128, 128, kernel_size = 3, stride = 1, padding = 1)
        self.deconv5 = nn.ConvTranspose2d(128, 64, kernel_size = 2, stride = 2, padding = 0)
        self.conv7 = nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1)
        self.deconv6 = nn.ConvTranspose2d(64, 32, kernel_size = 3, stride = 1, padding = 1)
        self.conv8 = nn.Conv2d(32, self.channels, kernel_size = 3, stride = 1, padding = 1)
    def encode(self, X):
        Y = self.resnet(X)
        Y=Y.view(-1, 512 * 16 *16)
        Y=F.relu(self.fc0(Y))
        Y=Y.view(-1, 1024, 4, 4)
        return Y

    def decode(self, X):
        Y=self.conv0(X)
        Y=self.conv1(Y)
        Y=self.deconv0(Y)
        Y=self.conv2(Y)
        Y=self.deconv1(Y)
        Y=self.conv3(Y)
        Y=self.deconv2(Y)
        Y=self.conv4(Y)
        Y=self.deconv3(Y)
        Y=self.conv5(Y)
        Y=self.deconv4(Y)
        Y=self.conv6(Y)
        Y=self.deconv5(Y)
        Y=self.conv7(Y)
        Y=self.deconv6(Y)
        Y=self.conv8(Y)
        return Y

    def forward(self, X):
        Y = self.encode(X)
        Y = self.decode(Y)
        return Y
  
