import torch
import torch.nn as nn

class Discriminator(nn.Module):
  def __init__(self):
    super(Discriminator, self).__init__()
    self.input_height = 128
    self.input_width = 128
    # layer 0
    self.conv0 = nn.Conv2d(6, 64, kernel_size = 4, stride = 2, padding = 1)
    self.relu0 = nn.LeakyReLU(0.2, inplace=True)
    # layer 1
    self.conv1 = nn.Conv2d(64, 128, kernel_size = 4, stride = 2, padding = 1)
    self.bn1 = nn.BatchNorm2d(128)
    self.relu1 = nn.LeakyReLU(0.2, inplace=True)
    # layer 2
    self.conv2 = nn.Conv2d(128,256, kernel_size = 4, stride = 2, padding = 1)
    self.bn2 = nn.BatchNorm2d(256)
    self.relu2 = nn.LeakyReLU(0.2, inplace=True)
    # layer 3
    self.conv3 = nn.Conv2d(256,512, kernel_size = 4, stride = 2, padding = 1)
    self.bn3 = nn.BatchNorm2d(512)
    self.relu3 = nn.LeakyReLU(0.2, inplace=True)
    # layer 4
    self.pad4 = nn.ZeroPad2d((1,0,1,0))
    self.conv4 =  nn.Conv2d(512, 512, 4, padding=1)
    self.bn4=nn.BatchNorm2d(512)
    self.relu4 = nn.LeakyReLU(0.2, inplace=True)
    # layer 5
    self.pad5 = nn.ZeroPad2d((1,0,1,0))
    self.conv5 =  nn.Conv2d(512, 1, 4, padding=1,bias=False)
    # output
    self.sig_out = torch.nn.Sigmoid()
    self.avg_out = nn.AvgPool2d(kernel_size=8)

  def forward(self, x, y):
    z = torch.cat((x, y), 1)
    z = self.relu0(self.conv0(z))
    z = self.relu1(self.bn1(self.conv1(z)))
    z = self.relu2(self.bn2(self.conv2(z)))
    z = self.relu3(self.bn3(self.conv3(z)))
    z = self.relu4(self.bn4(self.conv4(self.pad4(z))))
    z = self.conv5(self.pad5(z))
    z = self.avg_out(self.sig_out(z))
    return z

