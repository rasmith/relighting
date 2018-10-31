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
import depth_dataset
import convolutional_depth_model
import concurrent.futures

def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, 256, 256)
    return x
  
class Evaluator(object):
  def __init__(self, cfg_loader):
  self.cfg_loader = cfg_loader 
  self.task_cfg = {}

  def init(self, which_device):
    self.device = torch.device(which_device)\
        if "cuda" in which_device and torch.cuda.is_available() else "cpu")
    self.selected_device = torch.cuda.get_device()\
        if "cuda" in which_device and torch.cuda.is_available() else "cpu"
    print(f"selected device = %{selected_device}")
    self.task_cfg = self.cfg_loader.get_cfg(selected_device)

  def evaluate(self):
    batch_size = self.task_cfg['batch_size']
    criterion = self.task_cfg['criterion']
    dataset = self.task_cfg['dataset']
    dc_img = self.task_cfg['dc_img']
    eval_dir = self.task_cfg['eval_dir']
    loss = self.task_cfg['loss']
    model = self.task_cfg['model']
    num_epochs = self.task_cfg['num_epochs']
    optimizer = self.task_cfg['optimizer']
    shuffle = self.task_cfg['shuffle']
    weights_file = self.task_cfg['weights_file']

    dataset = Subset(dataset, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.load_state_dict(torch.load('./f{weights_file}')
    model.eval()

    os.mkdir(eval_dir) if not os.path.exists(eval_dir)

    i = 0
    for data in dataloader:
      img, target = data
      img = Variable(img).cpu()
      target = Variable(target).cpu()
      output = model(img)
      pic = to_img(output.cpu().data)
      print ("save_image ./eval_img_depths/image_%04d.png" % (i))
      save_image(pic, './eval_img_depths/image_%04d.png' % (i))
      i = i + 1
