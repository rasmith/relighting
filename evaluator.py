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
import concurrent.futures

def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, 256, 256)
    return x
  
class Evaluator(object):
  def __init__(self, cfg_loader, dataset_wrapper = None):
    self.cfg_loader = cfg_loader 
    self.task_cfg = {}
    self.dataset_wrapper = dataset_wrapper

  def init(self, which_device):
    self.device = torch.device(which_device\
        if "cuda" in which_device and torch.cuda.is_available() else "cpu")
    self.selected_device = torch.cuda.get_device()\
        if "cuda" in which_device and torch.cuda.is_available() else "cpu"
    print(f"selected device = {self.selected_device}")
    self.task_cfg = self.cfg_loader.get_cfg(self.selected_device)

  def evaluate(self):
    criterion = self.task_cfg['criterion']
    dataset = self.task_cfg['dataset']
    eval_dir = self.task_cfg['eval_dir']
    model = self.task_cfg['model']
    weights_file = self.task_cfg['weights_file']
    if self.dataset_wrapper is not None:
      dataset = self.dataset_wrapper(dataset)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model.load_state_dict(torch.load(f'./{weights_file}'))
    model.eval()
    if not os.path.exists(eval_dir):
      os.mkdir(eval_dir)
    i = 0
    for data in dataloader:
      img, target = data
      img = Variable(img).cpu()
      target = Variable(target).cpu()
      output = model(img)
      pic = to_img(output.cpu().data)
      print (f"save_image ./{eval_dir}/image_%04d.png" % (i))
      save_image(pic, f'./{eval_dir}/image_%04d.png' % (i))
      i = i + 1
