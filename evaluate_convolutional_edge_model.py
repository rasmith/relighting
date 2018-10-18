#!/usr/bin/env python3
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import os
import edge_dataset
import convolutional_edge_model

if not os.path.exists('./eval_img'):
    os.mkdir('./eval_img')

def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, 256, 256)
    return x

cfg_file = 'config.cfg'
input_dir = '.'
image_dir = 'out'
batch_size = 1
trans = transforms.Compose([transforms.ToTensor(), \
                            transforms.Normalize((0.5,), (1.0,))])
dataset = edge_dataset.NormalDataset(input_dir, cfg_file, image_dir,\
                                        trans, trans)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = convolutional_edge_model.ConvolutionalNormalModel().cpu()
model.load_state_dict(torch.load('./conv_normnal.pth'))
model.eval()

i = 0
for data in dataloader:
  img, target = data
  img = Variable(img).cpu()
  target = Variable(target).cpu()
  output = model(img)
  pic = to_img(output.cpu().data)
  print ("save_image ./eval_img/image_%04d.png" % (i))
  save_image(pic, './eval_img/image_%04d.png' % (i))
  i = i + 1

