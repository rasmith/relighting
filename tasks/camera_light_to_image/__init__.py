from datasets.camera_light_dataset import CameraLightDataset
import models as r
import samplers as s
from torch import nn
from torch.optim import Adam
from torch.optim import lr_scheduler
from torch.utils.data import Subset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.transforms import Compose
from torchvision.transforms import Normalize
from torchvision.transforms import ToTensor
import math
import numpy as np
import torch

task_name = 'camera_light_to_image'

cfg = {
    'annealing_step' : 1000,
    # 'batch_size' : 32,
    'batch_size' : 32,
    'base_steps' : int(250),
    'cfg_file' : 'config.cfg',
    'criterion' : nn.MSELoss(),
    'criterion_gan': nn.BCEWithLogitsLoss(),
    'criterion_pixel_l1': nn.L1Loss(),
    # 'data_wrapper' : None,
    'data_wrapper' : (lambda x : Subset(x, range(32))),
    # 'data_wrapper' : None,
    'dc_img' : f'dc_img/{task_name}',
    'discriminator_layers' : 3,
    # 'enabled' : False,
    'enabled' : True,  # to run this task
    'eval_dir':f'eval/{task_name}',
    'evaluation_enabled': True, # to evaluate this task
    'image_dir' : 'out', # source "images"
    'input_dir' : '.', # base input for all data needed
    'lambda_pixel': 100,
    'learning_rate' : 1e-4,
    'learning_rate_discriminator' : 1e-4,
    'log_dir': '',
    'log_to_tensorboard': True,
    # 'num_epochs' : 200,
    'num_epochs' : 10,
    # 'num_epochs' : 16000,
    'phases': ['training', 'validation'],
    'required_command_line_arguments': ['eval_dir',
                                        'discriminator_layers',
                                        'log_dir',
                                        'target_dir',
                                        'weights_dir'],
    'seed': (lambda : 42),
    'shuffle': True,
    'target_dir': f'targets/{task_name}',
    'target_transform' : Compose([ToTensor(), Normalize((0.5,), (1.0,))]),
    'task_name': f'{task_name}',
    'trainer' : 'gan',
    'training_enabled': True, # to train this task
    'transform' : None,
    'use_sampler': False, 
    'validation_split': .2,
    'weight_decay': 2e-6,
    'weight_decay_discriminator': 2e-5,
    'weights_dir' : 'weights',
    'weights_file' : f'weights/{task_name}.pth'
}

def cyclical_lr(step_sz, min_lr=0.001, max_lr=1, mode='triangular', scale_func=None, scale_md='cycles', gamma=1.):
    if scale_func == None:
        if mode == 'triangular':
            scale_fn = lambda x: 1.
            scale_mode = 'cycles'
        elif mode == 'triangular2':
            scale_fn = lambda x: 1 / (2.**(x - 1))
            scale_mode = 'cycles'
        elif mode == 'exp_range':
            scale_fn = lambda x: gamma**(x)
            scale_mode = 'iterations'
        else:
            raise ValueError(f'The {mode} is not valid value!')
    else:
        scale_fn = scale_func
        scale_mode = scale_md
    lr_lambda = lambda iters: min_lr + (max_lr - min_lr) * rel_val(iters, step_sz, scale_mode)
    def rel_val(iteration, stepsize, mode):
        cycle = math.floor(1 + iteration / (2 * stepsize))
        x = abs(iteration / stepsize - 2 * cycle + 1)
        if mode == 'cycles':
            return max(0, (1 - x)) * scale_fn(cycle)
        elif mode == 'iterations':
            return max(0, (1 - x)) * scale_fn(iteration)
        else:
            raise ValueError(f'The {scale_mode} is not valid value!')
    return lr_lambda

class CfgLoader(object):
  def __init__(self):
    self.cfg = cfg

  def get_cfg(self, device):
    self.cfg['device'] = device
    self.cfg['dataset'] = CameraLightDataset(cfg['target_dir'], cfg['task_name'],
                                        cfg['transform'],
                                        cfg['target_transform'])
    if self.cfg['data_wrapper'] is not None:
      self.cfg['wrapped_dataset'] = self.cfg['data_wrapper'](self.cfg['dataset'])
    else :
      self.cfg['wrapped_dataset'] = self.cfg['dataset']
    self.cfg['encoder'] = r.PrecodedResnet18Encoder128x128(9)
    self.cfg['decoder'] = r.Conv11Decoder128x128()
    self.cfg['activation'] = nn.Tanh()
    self.cfg['model'] = r.EncoderDecoder(self.cfg['encoder'],\
                                         self.cfg['decoder'],\
                                         self.cfg['activation'])
    self.cfg['discriminator'] = r.PrecodedDiscriminator(num_layers = int(self.cfg['discriminator_layers']), num_input_channels = 9)
    if 'cuda' in device:
      self.cfg['model'] = self.cfg['model'].cuda()
      self.cfg['discriminator'] = self.cfg['discriminator'].cuda()
      self.cfg['criterion'] = self.cfg['criterion'].cuda()
      self.cfg['criterion_pixel_l1'] = self.cfg['criterion_pixel_l1'].cuda()
      self.cfg['criterion_gan'] = self.cfg['criterion_gan'].cuda()
    # self.cfg['clr'] = cyclical_lr(2000, min_lr=0.001, max_lr=1, mode='triangular2')
    self.cfg['optimizer'] = Adam(cfg['model'].parameters(),
                                 lr=cfg['learning_rate'],
                                 weight_decay=cfg['weight_decay'])
    # self.cfg['scheduler'] = lr_scheduler.LambdaLR(self.cfg['optimizer'], [self.cfg['clr']])
    self.cfg['scheduler'] = lr_scheduler.ReduceLROnPlateau(self.cfg['optimizer'])
    # self.cfg['clr_discriminator'] = cyclical_lr(2000, min_lr=0.001, max_lr=1, mode='triangular2')
    self.cfg['optimizer_discriminator'] = \
      Adam(cfg['discriminator'].parameters(), lr=cfg['learning_rate_discriminator'],
            weight_decay =  cfg['weight_decay_discriminator'])
    # self.cfg['scheduler_discriminator'] = lr_scheduler.LambdaLR(self.cfg['optimizer_discriminator'], [self.cfg['clr_discriminator']])
    self.cfg['scheduler_discriminator'] = lr_scheduler.ReduceLROnPlateau(self.cfg['optimizer_discriminator'])
    if self.cfg['use_sampler']:
      self.cfg['shuffle'] = False
    dataset_size = len(self.cfg['wrapped_dataset'])
    indices = list(range(dataset_size))
    split = int(np.floor(self.cfg['validation_split'] * dataset_size))
    np.random.seed(self.cfg['seed']())
    np.random.shuffle(indices)
    training_indices, validation_indices = indices[split:], indices[:split]
    self.cfg['indices'] = indices
    self.cfg['training_indices'] = training_indices
    self.cfg['validation_indices'] = validation_indices
    self.cfg['training_dataset'] = Subset(self.cfg['wrapped_dataset'], \
                                          training_indices)
    self.cfg['validation_dataset'] = Subset(self.cfg['wrapped_dataset'], \
                                          validation_indices)
    self.cfg['training_sampler'] = None
    self.cfg['validation_sampler'] = None
    self.cfg['weights_file'] = f'{self.cfg["weights_dir"]}/{task_name}.pth'
    return self.cfg

