#!/usr/bin/env python3
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision import transforms
from torchvision.utils import save_image
import os
import edge_dataset
import convolutional_edge_model
import concurrent.futures

if not os.path.exists('./eval_img_edges'):
    os.mkdir('./eval_img_edges')

def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, 256, 256)
    return x

cfg_file = 'config.cfg'
input_dir = '.'
image_dir = 'out'
edges_dir = 'edges'
batch_size = 1
trans = transforms.Compose([transforms.ToTensor(), \
                            transforms.Normalize((0.5,), (1.0,))])
dataset = edge_dataset.EdgeDataset(input_dir, cfg_file, image_dir,\
                                        edges_dir, trans, trans)
dataset = Subset(dataset, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

model = convolutional_edge_model.ConvolutionalEdgeModel().cpu()
model.load_state_dict(torch.load('./conv_edges.pth'))
model.eval()
i = 0
for data in dataloader:
  img, target = data
  img = Variable(img).cpu()
  target = Variable(target).cpu()
  output = model(img)
  pic = to_img(output.cpu().data)
  print ("save_image ./eval_img_edges/image_%04d.png" % (i))
  save_image(pic, './eval_img_edges/image_%04d.png' % (i))
  i = i + 1
  


