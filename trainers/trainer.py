#!/usr/bin/env python3
import gc
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os
import torch
import time


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, 128, 128)
    return x

class Trainer(object):
  def __init__(self, cfg_loader, writer = None):
    self.cfg_loader = cfg_loader 
    self.task_cfg = {}
    self.initialized = False
    self.writer = writer 

  def init(self, which_device = None):
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

  def check(self):
    if not os.path.isdir(self.task_cfg['target_dir']):
      print(f'target_dir "{target_dir}" does not exist.\n')
      return False
    return True

  def train(self):
    if not self.initialized:
      self.init()
    if not self.check():
      return
    batch_size = self.task_cfg['batch_size']
    criterion = self.task_cfg['criterion']
    dataset = self.task_cfg['dataset']
    dc_img = self.task_cfg['dc_img']
    criterion = self.task_cfg['criterion']
    model = self.task_cfg['model']
    num_epochs = self.task_cfg['num_epochs']
    optimizer = self.task_cfg['optimizer']
    training_sampler = self.task_cfg['training_sampler']
    validation_sampler = self.task_cfg['validation_sampler']
    shuffle = self.task_cfg['shuffle']
    weights_file = self.task_cfg['weights_file']
    best_loss = 10000000.0

    if not os.path.exists(dc_img):
      os.mkdir(dc_img)
    print(f'pytorch version = {torch.__version__}')
    print(f'num_epochs = {num_epochs} len(dataset) = {len(dataset)}')
    current_step = 0
    for epoch in range(num_epochs):
        running_loss = 0.0
        for phase in ['training', 'validation']:
          dataloader = DataLoader(self.task_cfg[f'{phase}_dataset'],\
                                  batch_size = batch_size, shuffle = shuffle,
                                  sampler = self.task_cfg[f'{phase}_sampler'],
                                  pin_memory = True)
          for data in dataloader: 
              img, target = data
              img = Variable(img).cuda()\
                  if 'cuda' in self.selected_device else Variable(img).cpu()
              target = Variable(target).cuda()\
                  if 'cuda' in self.selected_device else Variable(target).cpu()
              # ===================forward=====================
              output = model(img)
              loss = criterion(output, target)
              # ===================backward====================
              optimizer.zero_grad()
              # print(f'memory_allocated={torch.cuda.memory_allocated()/(1024.0*1024.0)}')
              # print(f'memory_cached={torch.cuda.memory_cached()/(1024.0*1024.0)}')
              if phase == 'training':
                loss.backward()
                optimizer.step()
                current_step += 1
              running_loss += loss.item()
          # ===================log========================
          epoch_loss = running_loss / len(dataloader) 
          print(f'{phase}-epoch [{epoch+1:d}/{num_epochs:d}], loss:{epoch_loss:.4f}')
          self.writer.add_scalar(f"{self.task_cfg['task_name']}-{phase}-loss", \
                                    epoch_loss, epoch, time.time())
          if phase == 'validation':
            if loss < best_loss:
              # best_model.load_state_dict(model.state_dict())
              torch.save(model.state_dict(), f'./{weights_file}')

          if epoch % 10 == 0:
              pic = to_img(output.cpu().data)
              save_image(pic, f'{dc_img}/image_{epoch:d}.png')
              self.writer.add_image(f"{self.task_cfg['task_name']}-dc", pic, epoch, time.time())
    print('Saving...')
    # torch.save(best_model.state_dict(), f'./{weights_file}')
