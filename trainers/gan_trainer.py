#!/usr/bin/env python3
import gc
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os
import torch
import time
import numpy as np
import sys
# from tqdm import tqdm
from collections import OrderedDict
from torch import Tensor

def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, 128, 128)
    return x

def format_time(seconds):
    hours, seconds = divmod(seconds, 3600) 
    minutes, seconds = divmod(seconds, 60)
    return f'{int(hours)}:{int(minutes)}:{seconds:2.2f}'

def perform_validation_step(img, target, generator, discriminator,\
    optimizer_generator, optimizer_discriminator,\
    criterion_pixel_l1, criterion_pixel_l2, criterion_gan, optimization_choice,\
    stats, cfg):
    if cfg['device'] in 'cuda':
        torch.cuda.empty_cache()
    
    with torch.no_grad():
      lambda_pixel = cfg['lambda_pixel']
      ones = Variable(Tensor(np.ones((img.size(0), 1, 1, 1))), requires_grad=False)
      zeros= Variable(Tensor(np.zeros((img.size(0), 1, 1, 1))), requires_grad=False)
      if cfg['device'] in 'cuda':
        ones = ones.cuda()
        zeros = zeros.cuda()

      output = generator(img) # prediction with generator

      loss_pixel_l1 = criterion_pixel_l1(output, target) # get pixel-wise l1 loss
      loss_pixel_l2 = criterion_pixel_l2(output, target) # get pixel-wise l2 loss

      # print(f'output.shape = {output.shape} img.shape = {img.shape}')
      fakeness = discriminator(output, img) # ask discriminator how fake this is
      loss_gan =  criterion_gan(fakeness, ones) # get discriminator loss
      loss_generator = loss_gan + lambda_pixel * loss_pixel_l1    

      realness = discriminator(target, img)
      loss_gan_real = criterion_gan(realness, ones)

      fakeness = discriminator(output.detach(), img)
      loss_gan_fake = criterion_gan(fakeness, zeros)

      loss_discriminator = 0.5 * (loss_gan_real + loss_gan_fake)

    del fakeness
    del realness
    del ones
    del zeros
    del loss_gan_real
    del loss_gan_fake
    if cfg['device'] in 'cuda':
        torch.cuda.empty_cache()

    return loss_generator, loss_discriminator, loss_pixel_l1, loss_pixel_l2, output

def perform_gan_step(img, target, generator, discriminator,\
    optimizer_generator, optimizer_discriminator,\
    criterion_pixel_l1, criterion_pixel_l2, criterion_gan, optimization_choice,\
    stats, cfg):
    if cfg['device'] in 'cuda':
        torch.cuda.empty_cache()
    
    lambda_pixel = cfg['lambda_pixel']
    ones = Variable(Tensor(np.ones((img.size(0), 1, 1, 1))), requires_grad=False)
    zeros= Variable(Tensor(np.zeros((img.size(0), 1, 1, 1))), requires_grad=False)

    if cfg['device'] in 'cuda':
      ones = ones.cuda()
      zeros = zeros.cuda()

    # =============train generator===================
    optimizer_generator.zero_grad()

    output = generator(img) # prediction with generator

    loss_pixel_l1 = criterion_pixel_l1(output, target) # get pixel-wise l1 loss
    loss_pixel_l2 = criterion_pixel_l2(output, target) # get pixel-wise l2 loss

    fakeness = discriminator(output, img) # ask discriminator how fake this is
    loss_gan =  criterion_gan(fakeness, ones) # get discriminator loss
    loss_generator = loss_gan + lambda_pixel * loss_pixel_l1    

    loss_generator.backward()
    optimizer_generator.step()
    # =============train discriminator===================
    optimizer_discriminator.zero_grad()

    # real loss
    realness = discriminator(target, img)
    loss_gan_real = criterion_gan(realness, ones)

    # fake loss
    fakeness = discriminator(output.detach(), img)
    loss_gan_fake = criterion_gan(fakeness, zeros)

    loss_discriminator = 0.5 * (loss_gan_real + loss_gan_fake)

    loss_discriminator.backward()
    optimizer_discriminator.step()

    del fakeness
    del realness
    del ones
    del zeros
    del loss_gan_real
    del loss_gan_fake
    if cfg['device'] in 'cuda':
        torch.cuda.empty_cache()

    return loss_generator, loss_discriminator, loss_pixel_l1, loss_pixel_l2, output

def perform_base_step(img, target, generator, discriminator,\
    optimizer_generator, optimizer_discriminator,\
    criterion_pixel_l1, criterion_pixel_l2, criterion_gan, optimization_choice,\
    stats, cfg):
    if cfg['device'] in 'cuda':
        torch.cuda.empty_cache()
    
    optimizer_generator.zero_grad()

    output = generator(img) # prediction with generator

    loss_pixel_l2 = criterion_pixel_l2(output, target) # get pixel-wise l2 loss


    loss_pixel_l2.backward()
    optimizer_generator.step()

    if cfg['device'] in 'cuda':
        torch.cuda.empty_cache()

    return 0.0, 0.0, 0.0, loss_pixel_l2, output

def update_stat_average(stats, value_name, step_name, value):
  step = stats[step_name]
  stats[value_name] = (stats[value_name] * step + value) / (step + 1)

def add_stat(stats, value_name, value):
  stats[value_name] += value

def reset_stats(phase, data_size, stats):
  reset_data = {'phase': phase, 'loss_discriminator':0.0,\
    'loss_generator':0.0, 'l1':0.0, 'l2':0.0, 'tp':0.0, 'etap':0.0,\
    'validation_step': 0, 'training_step':0, 'data_size': data_size}
  for k in reset_data.keys():
    stats[k] = reset_data[k]

def initialize_stats(num_epochs, task_cfg):
  return {'phase':'', 'loss_discriminator':0.0, 'loss_generator':0.0, 'l1':0.0,\
       'l2':0.0, 'tp':0.0, 'etap':0.0, 't':0.0, 'eta':0.0,\
       'data_size': 0, 'training_step' : 0, \
       'validation_step': 0, 'global_training_step': 0,\
       'global_validation_step': 0, 'epoch': 0, 'num_epochs':num_epochs,\
       'avg_training_time': 0.0, 'avg_validation_time': 0.0, 'cfg' : task_cfg,\
       'avg_epoch_time':0.0, 'optimization_choice':'none', 'annealed':False}

def update_etas(phase, stats):
  avg_phase_time = stats[f'avg_{phase}_time']
  stats['etap'] = avg_phase_time * \
                  (stats['data_size'] - stats[f'{phase}_step'])
  stats['eta'] = stats['avg_epoch_time'] *\
                 (stats['num_epochs'] - stats['epoch'])

def update_training_stats(phase, stats, optimization_choice,\
  loss_discriminator,loss_generator, loss_pixel_l1, loss_pixel_l2, phase_start):
  step = f'{phase}_step'
  stats['optimization_choice'] = optimization_choice
  update_stat_average(stats, 'loss_generator', step, loss_generator)
  update_stat_average(stats, 'loss_discriminator', step, loss_discriminator)
  update_stat_average(stats, 'l1', step, loss_pixel_l1)
  update_stat_average(stats, 'l2', step, loss_pixel_l2)
  current_time = time.time()
  add_stat(stats, 'tp', current_time  - phase_start)
  add_stat(stats, 't', current_time - phase_start)
  update_etas(phase, stats)

def update_progress_bar(stats):
  tp = format_time(stats['tp'])
  etap = format_time(stats['etap'])
  t = format_time(stats['t'])
  eta = format_time(stats['eta'])
  # desc = f"[{stats['epoch']:04d}] {stats['phase']:#>10}\
    # {stats['optimization_choice']: >4} lg:{stats['loss_generator']:7.4f}\
    # ld:{stats['loss_discriminator']:7.4f} l1:{stats['l1']:7.4f}\
    # l2:{stats['l2']:7.4f} tp:{tp} etap:{etap} t:{t} eta:{eta}"

  phase = stats['phase'][0]
  optimization_choice = stats['optimization_choice'][0]
  desc = (f"[{stats['epoch']:03d}] {phase} "
          f"{optimization_choice} l2:{stats['l2']:6.4f} "
          f"l1:{stats['l1']:6.4f} "
          f" etap:{etap} eta:{eta}"
          f" {stats['global_training_step']}"
          f" {stats['global_validation_step']}"
          # f" {torch.cuda.memory_allocated()/(1024.0*1024.0):6.2f}"
  )
  sys.stdout.write(" \r")
  sys.stdout.write(desc)

class GanTrainer(object):
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
    criterion_pixel_l2 = self.task_cfg['criterion']
    criterion_pixel_l1 = self.task_cfg['criterion_pixel_l1']
    criterion_gan = self.task_cfg['criterion_gan']
    dataset = self.task_cfg['dataset']
    training_dataset = self.task_cfg['training_dataset']
    validation_dataset = self.task_cfg['validation_dataset']
    dc_img = self.task_cfg['dc_img']
    generator = self.task_cfg['model']
    discriminator = self.task_cfg['discriminator']
    num_epochs = self.task_cfg['num_epochs']
    base_steps = self.task_cfg['base_steps']
    optimizer_generator = self.task_cfg['optimizer']
    optimizer_discriminator = self.task_cfg['optimizer_discriminator']
    lambda_pixel = self.task_cfg['lambda_pixel']
    training_sampler = self.task_cfg['training_sampler']
    validation_sampler = self.task_cfg['validation_sampler']
    shuffle = self.task_cfg['shuffle']
    weights_file = self.task_cfg['weights_file']
    best_loss = 10000000.0

    if not os.path.exists(dc_img):
      os.mkdir(dc_img)
    print(f'pytorch version = {torch.__version__}')
    print(f'num_epochs = {num_epochs} len(dataset) = {len(dataset)}')
    training_stats = initialize_stats(num_epochs, self.task_cfg)
    global_start = time.time()
    for epoch in range(num_epochs):
        epoch_start = time.time()
        for phase in self.task_cfg['phases']:
          phase_start = time.time()
          dataloader = DataLoader(self.task_cfg[f'{phase}_dataset'],\
                                  batch_size = batch_size, shuffle = shuffle,
                                  sampler = self.task_cfg[f'{phase}_sampler'],
                                  pin_memory = True)
          if phase == 'training':
            torch.enable_grad()
            generator.train()
            discriminator.train()
          else:
            torch.no_grad()
            generator.eval()
            discriminator.eval()
            criterion_gan.eval()
            criterion_gan.eval()
            criterion_pixel_l1.eval()
            criterion_pixel_l2.eval()

          data_size = len(dataloader)
          reset_stats(phase, data_size, training_stats)
          logged_scalars = {}
          for data in dataloader: 
              img, target = data
              img = Variable(img, requires_grad = False).cuda()\
                  if 'cuda' in self.selected_device else \
                    Variable(img, requires_grad = False).cpu()
              target = Variable(target, requires_grad = False).cuda()\
                  if 'cuda' in self.selected_device else \
                    Variable(target, requires_grad = False).cpu()
              optimization_choice = 'none'
              if phase == 'training':
                optimization_choice = 'base'
                if training_stats['global_training_step'] > base_steps:
                  optimization_choice = 'gan'
                  loss_generator, loss_discriminator, loss_pixel_l1, loss_pixel_l2, output  =\
                    perform_gan_step(img, target, generator, discriminator,\
                                 optimizer_generator, optimizer_discriminator,\
                                 criterion_pixel_l1, criterion_pixel_l2,\
                                 criterion_gan, optimization_choice, training_stats,\
                                 self.task_cfg)
                else:
                  loss_generator, loss_discriminator, loss_pixel_l1, loss_pixel_l2, output  =\
                    perform_base_step(img, target, generator, discriminator,\
                                 optimizer_generator, optimizer_discriminator,\
                                 criterion_pixel_l1, criterion_pixel_l2,\
                                 criterion_gan, optimization_choice, training_stats,\
                                 self.task_cfg)
              else:
                  loss_generator, loss_discriminator, loss_pixel_l1, loss_pixel_l2, output  =\
                    perform_validation_step(img, target, generator, discriminator,\
                                 optimizer_generator, optimizer_discriminator,\
                                 criterion_pixel_l1, criterion_pixel_l2,\
                                 criterion_gan, optimization_choice, training_stats,\
                                 self.task_cfg)
              #================update stats and progress bar===================
              update_training_stats(phase, training_stats, optimization_choice,\
                loss_generator, loss_discriminator, loss_pixel_l1,\
                loss_pixel_l2, phase_start)
              update_progress_bar(training_stats)
              training_stats[f'global_{phase}_step'] += 1
              training_stats[f'{phase}_step'] += 1
              logged_scalars = {'l1': loss_pixel_l1.item() \
                                        if isinstance(loss_pixel_l1, torch.Tensor) \
                                        else loss_pixel_l1,
                                'l2': loss_pixel_l2.item() \
                                        if isinstance(loss_pixel_l2, torch.Tensor) \
                                        else loss_pixel_l2,
                                 'g': loss_generator.item() \
                                        if isinstance(loss_generator, torch.Tensor)
                                        else loss_generator,
                                 'd': loss_discriminator.item() \
                                        if isinstance(loss_discriminator, torch.Tensor) \
                                        else loss_discriminator}
              del img
              del target
              del loss_generator
              del loss_discriminator
              del loss_pixel_l1
              del loss_pixel_l2
          del dataloader
          phase_end = time.time()
          update_stat_average(training_stats, f'avg_{phase}_time',\
                              f'global_{phase}_step', phase_end - phase_start)
          # ===================log========================
          if self.task_cfg['log_to_tensorboard']: 
            for k in logged_scalars.keys():
              self.writer.add_scalar(f"{self.task_cfg['task_name']}-{phase}-{k}",\
                          logged_scalars[k], epoch, time.time())
          if phase == 'validation':
            if training_stats['l2'] < best_loss:
              best_model_state_dict = \
                {k:v.to('cpu') for k, v in generator.state_dict().items()}
              best_model_state_dict = OrderedDict(best_model_state_dict)
          sys.stdout.write("\n")

          epoch_end = time.time()
          #=========update stats and progress bar===================
          update_stat_average(training_stats, 'avg_epoch_time', 'epoch',\
                              epoch_end - epoch_start)
          update_progress_bar(training_stats)
          #========tensorboard==============
          if epoch % 10 == 0:
              if self.task_cfg['log_to_tensorboard']: 
                pic = to_img(output.cpu().data)
                save_image(pic, f'{dc_img}/image_{epoch:d}.png')
                self.writer.add_images(f"{self.task_cfg['task_name']}-dc", pic,\
                                      epoch, time.time())
        training_stats['epoch'] += 1
    print('Saving...')
    torch.save(best_model_state_dict, f'./{weights_file}')
