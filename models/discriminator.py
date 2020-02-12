import torch
import torch.nn as nn

class Discriminator(nn.Module):
  def __init__(self, inplace_bn = False, checkpointed_bn = True):
    super(Discriminator, self).__init__()
    self.input_height = 128
    self.input_width = 128
    self.inplace_bn = inplace_bn
    self.checkpointed_bn = checkpointed_bn
    if self.checkpointed_bn:
      self.inplace_bn = False
      from torch.utils.checkpoint import checkpoint
    if self.inplace_bn:
      from inplace_abn.modules.bn import InPlaceABN
      
    # layer 0
    self.conv0 = nn.Conv2d(6, 64, kernel_size = 4, stride = 2, padding = 1)
    self.relu0 = nn.LeakyReLU(0.2, inplace=True)
    # layer 1
    self.conv1 = nn.Conv2d(64, 128, kernel_size = 4, stride = 2, padding = 1)
    if self.inplace_bn:
      self.abn1 = InPlaceABN(128, activation = 'leakyrelu', slope = 0.2)
    else:
      self.bn1 = nn.BatchNorm2d(128)
      self.relu1 = nn.LeakyReLU(0.2, inplace=True)

    # layer 2
    self.conv2 = nn.Conv2d(128,256, kernel_size = 4, stride = 2, padding = 1)
    if self.inplace_bn:
      self.abn2 = InPlaceABN(256, activation = 'leakyrelu', slope = 0.2)
    else:
      self.bn2 = nn.BatchNorm2d(256)
      self.relu2 = nn.LeakyReLU(0.2, inplace=True)
    # layer 3
    self.conv3 = nn.Conv2d(256,512, kernel_size = 4, stride = 2, padding = 1)
    if self.inplace_bn:
      self.abn3 = InPlaceABN(512, activation = 'leakyrelu', slope = 0.2)
    else:
      self.bn3 = nn.BatchNorm2d(512)
      self.relu3 = nn.LeakyReLU(0.2, inplace=True)

    # layer 4
    self.pad4 = nn.ZeroPad2d((1,0,1,0))
    self.conv4 =  nn.Conv2d(512, 512, 4, padding=1)
    if self.inplace_bn:
      self.abn4 = InPlaceABN(512, activation = 'leakyrelu', slope = 0.2)
    else:
      self.bn4 = nn.BatchNorm2d(512)
      self.relu4 = nn.LeakyReLU(0.2, inplace=True)

    # layer 5
    self.pad5 = nn.ZeroPad2d((1,0,1,0))
    self.conv5 =  nn.Conv2d(512, 1, 4, padding=1,bias=False)
    # output
    self.sig_out = torch.nn.Sigmoid()
    self.avg_out = nn.AvgPool2d(kernel_size=8)

  def forward(self, x, y):
    if self.checkpointed_bn:
      from torch.utils.checkpoint import checkpoint
    if self.inplace_bn:
      from inplace_abn.modules.bn import InPlaceABN
    print(f"\n\nx.shape = {x.shape} y.shape = {y.shape}\n\n")
    z = torch.cat((x, y), 1)
    # layer 0
    z = self.relu0(self.conv0(z))
    if self.inplace_bn:
      z = self.abn1(self.conv1(z))
      z = self.abn2(self.conv2(z))
      z = self.abn3(self.conv3(z))
      z = self.abn4(self.conv4(self.pad4(z)))
    else:
      if self.checkpointed_bn:
        # layer 1
        z = self.conv1(z) 
        z = checkpoint(lambda y: self.bn1(y), z)
        z = self.relu1(z)
        # layer 2
        z = self.conv2(z)
        z = checkpoint(lambda y: self.bn2(y), z)
        z = self.relu2(z)
        # layer 3
        z = self.conv3(z)
        z = checkpoint(lambda y: self.bn3(y), z)
        z = self.relu3(z)
        # layer 4
        z = self.conv4(self.pad4(z))
        z = checkpoint(lambda y: self.bn4(y), z)
        z = self.relu4(z)
      else:
        z = self.relu1(self.bn1(self.conv1(z)))
        z = self.relu2(self.bn2(self.conv2(z)))
        z = self.relu3(self.bn3(self.conv3(z)))
        z = self.relu4(self.bn4(self.conv4(self.pad4(z))))
    z = self.conv5(self.pad5(z))
    z = self.avg_out(self.sig_out(z))
    return z

