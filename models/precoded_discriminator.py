import torch
import torch.nn as nn
# from inplace_abn.modules.bn import InPlaceABN
# from inplace_abn import InPlaceABN
import models.small_precoder as p

class Identity(nn.Module):
    def forward(self, x):
            return x

class PrecodedDiscriminator(nn.Module):
  def __init__(self, precoder_size, inplace_bn = False, use_bn = False):
    super(PrecodedDiscriminator, self).__init__()
    self.precoder_size = precoder_size
    self.precoder = p.SmallPrecoder(self.precoder_size)
    self.input_height = 128
    self.input_width = 128
    self.inplace_bn = inplace_bn
    self.use_bn = use_bn
    # layer 0    [6 -> 32]
    self.conv0 = nn.Conv2d(6, 32, kernel_size = 1, stride = 1, padding = 0)
    self.relu0 = nn.LeakyReLU(0.2, inplace=True)
    # layer 1    [32 -> 64]
    self.conv1 = nn.Conv2d(32, 64, kernel_size = 1, stride = 1, padding = 0)
    if self.inplace_bn:
      self.abn1 = InPlaceABN(64, activation = 'leaky_relu', activation_param = 0.2)
    else:
      if self.use_bn:
        self.bn1 = nn.BatchNorm2d(64)
      else:
        self.bn1 = Identity()
      self.relu1 = nn.LeakyReLU(0.2, inplace=True)

    # layer 3 [64 -> 128]
    self.conv2 = nn.Conv2d(64, 128, kernel_size = 1, stride = 1, padding = 0)
    if self.inplace_bn:
      self.abn2 = InPlaceABN(128, activation = 'leaky_relu', activation_param = 0.2)
    else:
      if self.use_bn:
        self.bn2 = nn.BatchNorm2d(128)
      else:
        self.bn2 = Identity()
      self.relu2 = nn.LeakyReLU(0.2, inplace=True)

    # layer3 [128 -> 1]
    self.conv3 = nn.Conv2d(128, 1, kernel_size = 1, stride = 1, padding = 0)
    # layer 3
    # self.conv3 = nn.Conv2d(256,512, kernel_size = 4, stride = 2, padding = 1)


    # self.conv3 = nn.Conv2d(256, 1, kernel_size = 4, stride = 2, padding = 1)
    # if self.inplace_bn:
      # self.abn3 = InPlaceABN(512, activation = 'leaky_relu', activation_param = 0.2)
    # else:
      # if self.use_bn:
        # self.bn3 = nn.BatchNorm2d(512)
      # else:
        # self.bn3 = Identity()
      # self.relu3 = nn.LeakyReLU(0.2, inplace=True)

    # # layer 4
    # self.pad4 = nn.ZeroPad2d((1,0,1,0))
    # self.conv4 =  nn.Conv2d(512, 512, 4, padding=1)
    # if self.inplace_bn:
      # self.abn4 = InPlaceABN(512, activation = 'leaky_relu', activation_param = 0.2)
    # else:
      # if self.use_bn:
        # self.bn4 = nn.BatchNorm2d(512)
      # else:
        # self.bn4 = Identity()
      # self.relu4 = nn.LeakyReLU(0.2, inplace=True)

    # # layer 5
    # self.pad5 = nn.ZeroPad2d((1,0,1,0))
    # self.conv5 =  nn.Conv2d(512, 1, 4, padding=1,bias=False)
    # # output
    # self.sig_out = torch.nn.Sigmoid()
    # self.avg_out = nn.AvgPool2d(kernel_size=8)
    # self.softmax_out = nn.Softmax()

  def forward(self, x, y):
    w = self.precoder(y)
    z = torch.cat((x, w), 1)
    z = self.relu0(self.conv0(z))
    z = self.relu1(self.conv1(z))
    z = self.relu2(self.conv2(z))
    z = self.conv3(z)
    # w = self.precoder(y)
    # z = torch.cat((x, w), 1)
    # z = self.relu0(self.conv0(z))
    # if self.inplace_bn:
      # z = self.abn1(self.conv1(z))
      # z = self.abn2(self.conv2(z))
      # z = self.abn3(self.conv3(z))
      # z = self.abn4(self.conv4(self.pad4(z)))
    # else:
      # z = self.relu1(self.bn1(self.conv1(z)))
      # z = self.relu2(self.bn2(self.conv2(z)))
      # z = self.relu3(self.bn3(self.conv3(z)))
      # # z = self.relu4(self.bn4(self.conv4(self.pad4(z))))
      # # z = self.relu4(self.bn4(self.conv4(z)))
    # # z = self.conv5(self.pad5(z))
    # z = self.avg_out(self.sig_out(z))
    # z = self.softmax_out(z)
    # z = self.sig_out(z)
    return z

