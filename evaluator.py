#!/usr/bin/env python3
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision import transforms
from torchvision.utils import save_image
import samplers as s
import os
import concurrent.futures

def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, 128, 128)
    return x
  
class Evaluator(object):
  def __init__(self, cfg_loader, writer = None):
    self.cfg_loader = cfg_loader 
    self.task_cfg = {}
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

  def evaluate(self, sampler_type):
    eval_dir = self.task_cfg['eval_dir']
    model = self.task_cfg['model']
    eval_dir = f'{eval_dir}/{sampler_type}'
    weights_file = self.task_cfg['weights_file']
    model.load_state_dict(torch.load(f'./{weights_file}'))
    dataset = self.task_cfg['dataset']
    model.eval()
    if not os.path.exists(eval_dir):
      os.makedirs(eval_dir)
    for i, img_number in enumerate(self.task_cfg[f'{sampler_type}_indices']):
      value, target = dataset[img_number]
      print(f'value.shape = {value.shape}')
      value = value.view(1, *value.shape)
      target = target.view(1, *target.shape)
      value = Variable(value).cuda() if self.use_cuda else  Variable(value).cpu()
      target = Variable(target).cuda() if self.use_cuda else Variable(target).cpu()
      print (f'value.shape = {value.shape}')
      output = model(value)
      pic = to_img(output.cpu().data)
      print (f'save_image ./{eval_dir}/image_{img_number:04}.png')
      print (f'pic.shape = {pic.shape} output.shape = {output.shape}')
      print (f'target.shape = {target.shape}')
      save_image(pic, f'./{eval_dir}/image_{img_number:04}.png')
      if i < 10:
          out = to_img(torch.cat((output, target)).cpu().data)
          self.writer.add_image(f'{self.task_cfg["task_name"]}-{i}-eval-{sampler_type}', out)
