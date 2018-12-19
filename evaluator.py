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
  def __init__(self, cfg_loader, dataset_wrapper = None, writer = None):
    self.cfg_loader = cfg_loader 
    self.task_cfg = {}
    self.dataset_wrapper = dataset_wrapper
    self.writer = writer

  def init(self, which_device):
    if which_device is None:
      which_device = "cpu"
    self.use_cuda = True\
        if "cuda" in which_device and torch.cuda.is_available() else False
    self.device = torch.device(which_device\
        if "cuda" in which_device and torch.cuda.is_available() else "cpu")
    if not self.use_cuda:
      which_device = "cpu"
    print(f"selected device = {which_device}")
    self.selected_device = which_device
    self.task_cfg = self.cfg_loader.get_cfg(which_device)
    self.initialized = True

  def encode(self, dest_dir):
    dataset = self.task_cfg['dataset']
    model = self.task_cfg['model']
    weights_file = self.task_cfg['weights_file']
    print(f'weights_file={weights_file}')
    if self.dataset_wrapper is not None:
      dataset = self.dataset_wrapper(dataset)
    dataloader = DataLoader(dataset, batch_size=1)

    model.load_state_dict(torch.load(f'./{weights_file}'))
    model.eval()
    if not os.path.exists(dest_dir):
      os.mkdir(dest_dir)
    i = 0
    for data in dataloader:
      img, target = data
      img = Variable(img).cuda() if self.use_cuda else  Variable(img).cpu()
      target = Variable(target).cuda() if self.use_cuda else Variable(img).cpu()
      context_vector = model.encode(img)
      context_vector_file = f'./{dest_dir}/{i:08d}.pth'
      print (f'save {context_vector_file}')
      torch.save(torch.squeeze(context_vector), f'{context_vector_file}')
      i = i + 1

  def evaluate(self, sampler):
    dataset = self.task_cfg['dataset']
    eval_dir = self.task_cfg['eval_dir']
    model = self.task_cfg['model']
    if sampler == "validation":
      sampler = self.task_cfg['validation_sampler']
      eval_dir = f'{eval_dir}/validation'
    else:
      sampler = self.task_cfg['train_sampler']
      eval_dir = f'{eval_dir}/training'
    weights_file = self.task_cfg['weights_file']
    shuffle = self.task_cfg['shuffle']
    if self.dataset_wrapper is not None:
      dataset = self.dataset_wrapper(dataset)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=shuffle, sampler = sampler)

    model.load_state_dict(torch.load(f'./{weights_file}'))
    model.eval()
    if not os.path.exists(eval_dir):
      os.mkdir(eval_dir)
    i = 0
    grid_images = None
    for data in dataloader:
      img, target = data
      img = Variable(img).cuda() if self.use_cuda else  Variable(img).cpu()
      target = Variable(target).cuda() if self.use_cuda else Variable(target).cpu()
      output = model(img)
      pic = to_img(output.cpu().data)
      print (f"save_image ./{eval_dir}/image_%04d.png" % (i))
      print (f"pic.shape = {pic.shape} output.shape = {output.shape} \
          target.shape = {target.shape}")
      save_image(pic, f'./{eval_dir}/image_%04d.png' % (i))
      if i < 10:
          out = to_img(torch.cat((output, target)).cpu().data)
          self.writer.add_image(f'{self.task_cfg["task_name"]}-{i}-eval', out)
      i = i + 1
