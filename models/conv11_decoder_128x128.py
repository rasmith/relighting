import torch
import torch.nn as nn
import models.resnet_abn as r
from inplace_abn.modules.bn import InPlaceABN

class Conv11Decoder128x128(nn.Module):

  def __init__(self, activation = None, use_inplace_bn = True,\
              use_dropout = True, use_checkpointed_bn = True):
    super(Conv11Decoder128x128, self).__init__()
    self.input_width  = 4
    self.input_height = 4
    self.input_channels = 512
    self.input_size = self.input_width * self.input_height * self.input_channels
    self.output_width = 128
    self.output_height = 128
    self.output_channels = 3
    self.output_size = self.output_width * self.output_height * self.output_channels
    self.use_inplace_bn = use_inplace_bn
    self.use_checkpointed_bn = use_checkpointed_bn
    self.use_dropout = use_dropout 
    if self.use_checkpointed_bn:
      self.use_inplace_bn = False

    # layer 0
    self.deconv0 = nn.ConvTranspose2d(512, 512, kernel_size = 3, stride = 3,\
                                      padding = 2)
    if self.use_inplace_bn: 
      self.abn0  = InPlaceABN(512, activation = 'none')
    else:
      self.bn0 = nn.BatchNorm2d(512)

    # layer 1
    self.conv1 = nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1)
    if self.use_inplace_bn: 
      self.abn1 = InPlaceABN(512, activation = 'none')
    else:
      self.bn1 = nn.BatchNorm2d(512)

    # layer 2 
    self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size = 3, stride = 2,\
                                      padding= 1, output_padding=1)
    if self.use_inplace_bn: 
      self.abn2 = InPlaceABN(256, activation = 'none')
    else:
      self.bn2 = nn.BatchNorm2d(256)

    if use_dropout:
      self.dropout2 = nn.Dropout(0.5)

    # layer 3
    self.conv3 = nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding = 1)
    if self.use_inplace_bn: 
      self.abn3 = InPlaceABN(256, activation = 'none')
    else:
      self.bn3 = nn.BatchNorm2d(256)

    # layer 4
    self.deconv4 = nn.ConvTranspose2d(256, 128, kernel_size = 5, stride = 2,\
                                      padding= 2, output_padding=1)
    if self.use_inplace_bn: 
      self.abn4 = InPlaceABN(128, activation = 'none')
    else:
      self.bn4 = nn.BatchNorm2d(128)
    if use_dropout:
      self.dropout4 = nn.Dropout(0.5)

    # layer 5
    self.conv5 = nn.Conv2d(128, 128, kernel_size = 3, stride = 1, padding = 1)
    if self.use_inplace_bn: 
      self.abn5 = InPlaceABN(128, activation = 'none')
    else:
      self.bn5 = nn.BatchNorm2d(128)

    # layer 6
    self.deconv6 = nn.ConvTranspose2d(128, 64, kernel_size = 5, stride = 2,\
                                      padding= 2, output_padding=1) 
    if self.use_inplace_bn: 
      self.abn6 = InPlaceABN(64, activation = 'none')
    else:
      self.bn6 = nn.BatchNorm2d(64)

    # layer 7
    self.conv7 = nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1)
    if self.use_inplace_bn: 
      self.abn7 = InPlaceABN(64, activation = 'none')
    else:
      self.bn7 = nn.BatchNorm2d(64)

    # layer 8
    self.deconv8 = nn.ConvTranspose2d(64, 32, kernel_size = 5, stride = 2,\
                                      padding= 2, output_padding=1)
    if self.use_inplace_bn: 
      self.abn8 = InPlaceABN(32, activation = 'none')
    else:
      self.bn8 = nn.BatchNorm2d(32)

    # layer 9
    self.conv9 = nn.Conv2d(32, 3, kernel_size = 3, stride = 1, padding = 1)
    # if self.use_inplace_bn: 
      # self.abn9 = InPlaceABN(3, activation = 'none')
    # else:
      # self.bn9 = nn.BatchNorm2d(3)

    # output
    self.activation = activation
  
  def forward(self, x):
    if self.use_inplace_bn:
      x = self.abn0(self.deconv0(x))
      x = self.abn1(self.conv1(x))
      if self.use_dropout:
        x =  self.dropout2(self.abn2(self.deconv2(x)))
      else:
        x =  self.abn2(self.deconv(x))
      x =  self.abn3(self.conv3(x))
      if self.use_dropout:
        x =  self.dropout4(self.abn4(self.deconv4(x)))
      else:
        x =  self.abn4(self.deconv4(x))
        
      x =  self.abn5(self.conv5(x))
      x =  self.abn6(self.deconv6(x))
      x =  self.abn7(self.conv7(x))
      x =  self.abn8(self.deconv8(x))
    else:
      if self.use_checkpointed_bn:
        from torch.utils.checkpoint import checkpoint
        x = self.deconv0(x)
        x = checkpoint(lambda y: self.bn0(y), x)

        x = self.conv1(x)
        x = checkpoint(lambda y: self.bn1(y), x)

        x = self.deconv2(x)
        x = checkpoint(lambda y: self.bn2(y), x)
        if self.use_dropout:
          x = self.dropout2(x)

        x = self.conv3(x)
        x = checkpoint(lambda y: self.bn3(y), x)

        x = self.deconv4(x)
        x = checkpoint(lambda y: self.bn4(y), x)
        if self.use_dropout:
          x = self.dropout4(x)

        x = self.conv5(x)
        x = checkpoint(lambda y: self.bn5(y), x)

        x = self.deconv6(x)
        x = checkpoint(lambda y: self.bn6(y), x)

        x = self.conv7(x)
        x = checkpoint(lambda y: self.bn7(y), x)

        x = self.deconv8(x)
        x = checkpoint(lambda y: self.bn8(y), x)
      else:
        x = self.bn0(self.deconv0(x)) 
        x = self.bn1(self.conv1(x))
        x = self.bn2(self.deconv2(x))
        if self.use_dropout:
          x = self.dropout2(x)
        x = self.bn3(self.conv3(x))
        x = self.bn4(self.deconv4(x))
        if self.use_dropout:
          x = self.dropout4(x)
        x = self.bn5(self.conv5(x))
        x = self.bn6(self.deconv6(x))
        x = self.bn7(self.conv7(x))
        x = self.bn8(self.deconv8(x))

    x = self.conv9(x)
    if self.activation is not None:
      x = self.activation(x)
    return x
