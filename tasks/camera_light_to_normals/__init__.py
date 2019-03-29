from datasets.camera_light_dataset import CameraLightDataset
import models as r
import samplers as s
from torch import nn
from torch.optim import Adam
from torch.utils.data import Subset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.transforms import Compose
from torchvision.transforms import Normalize
from torchvision.transforms import ToTensor
import numpy as np
import torch

task_name = 'camera_light_to_normals'

cfg = {
    'annealing_step' : 1000,
    # 'batch_size' : 32,
    'batch_size' : 32,
    'base_steps' : int(128/32),
    'cfg_file' : 'config.cfg',
    'criterion' : nn.MSELoss(),
    'criterion_gan': nn.MSELoss(),
    'criterion_pixel_l1': nn.L1Loss(),
    # 'data_wrapper' : None,
    'data_wrapper' : (lambda x : Subset(x, range(128))),
    'dc_img' : f'dc_img/{task_name}',
    # 'enabled' : False,
    'enabled' : True,  # to run this task
    'eval_dir':f'eval/{task_name}',
    'evaluation_enabled': True, # to evaluate this task
    'image_dir' : 'out', # source "images"
    'input_dir' : '.', # base input for all data needed
    'lambda_pixel': 100,
    'learning_rate' : 1e-4,
    'learning_rate_discriminator' : 1e-4,
    'log_to_tensorboard': False,
    'num_epochs' : 100,
    # 'num_epochs' : 2,
    'phases': ['training', 'validation'],
    'seed': (lambda : 42),
    'shuffle': True,
    'target_dir': f'targets/normals',
    'target_transform' : Compose([ToTensor(), Normalize((0.5,), (1.0,))]),
    'task_name': f'{task_name}',
    'trainer' : 'gan',
    'training_enabled': False, # to train this task
    'transform' : None,
    'use_sampler': True, 
    'validation_split': .2,
    'weight_decay': 2e-6,
    'weight_decay_discriminator': 2e-5,
    'weights_file' : f'weights/{task_name}.pth'
}

class CfgLoader(object):
  def __init__(self):
    self.cfg = cfg

  def get_cfg(self, device):
    self.cfg['device'] = device
    self.cfg['dataset'] = CameraLightDataset(cfg['input_dir'], cfg['cfg_file'],
                                             cfg['target_dir'], cfg['task_name'],
                                             cfg['transform'],
                                             cfg['target_transform'])
    if self.cfg['data_wrapper'] is not None:
      self.cfg['wrapped_dataset'] = self.cfg['data_wrapper'](self.cfg['dataset'])
    else :
      self.cfg['wrapped_dataset'] = self.cfg['dataset']
    self.cfg['encoder'] = r.PrecodedResnet18Encoder128x128()
    self.cfg['decoder'] = r.Conv11Decoder128x128()
    self.cfg['activation'] = nn.Tanh()
    self.cfg['model'] = r.EncoderDecoder(self.cfg['encoder'],\
                                         self.cfg['decoder'],\
                                         self.cfg['activation'])
    self.cfg['discriminator'] = r.PrecodedDiscriminator()
    if device in 'cuda':
      self.cfg['model'] = self.cfg['model'].cuda()
      self.cfg['discriminator'] = self.cfg['discriminator'].cuda()
      self.cfg['criterion'] = self.cfg['criterion'].cuda()
      self.cfg['criterion_pixel_l1'] = self.cfg['criterion_pixel_l1'].cuda()
      self.cfg['criterion_gan'] = self.cfg['criterion_gan'].cuda()
    self.cfg['optimizer'] = Adam(cfg['model'].parameters(),
                                 lr=cfg['learning_rate'],
                                 weight_decay=cfg['weight_decay'])
    self.cfg['optimizer_discriminator'] = \
      Adam(cfg['discriminator'].parameters(), lr=cfg['learning_rate_discriminator'],
            weight_decay =  cfg['weight_decay_discriminator'])
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
    return self.cfg

