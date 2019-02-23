from datasets.file_system_dataset import FileSystemDataset
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

task_name = 'normals'

cfg = {
    'batch_size' : 32,
    'cfg_file' : 'config.cfg',
    'criterion' : nn.MSELoss(),
    'data_wrapper' : None,
    # 'data_wrapper' : (lambda x : Subset(x, range(32))),
    'dc_img' : f'dc_img/{task_name}',
    # 'enabled' : False,
    'enabled' : True,  # to run this task
    'eval_dir':f'eval/{task_name}',
    'evaluation_enabled': True, # to evaluate this task
    'image_dir' : 'out', # source "images"
    'input_dir' : '.', # base input for all data needed
    'learning_rate' : 1e-4,
    'num_epochs' : 200,
    # 'num_epochs' : 2,
    'seed': (lambda : 42),
    'shuffle': True,
    'target_dir': f'targets/{task_name}',
    'target_transform' : Compose([ToTensor(), Normalize((0.5,), (1.0,))]),
    'task_name': f'{task_name}',
    'training_enabled': True, # to train this task
    'transform' : Compose([ToTensor(), Normalize((0.5,), (1.0,))]),
    'use_sampler': True, 
    'validation_split': .2,
    'weight_decay': 2e-6,
    'weights_file' : f'weights/{task_name}.pth'
}

class CfgLoader(object):
  def __init__(self):
    self.cfg = cfg

  def get_cfg(self, device):
    self.cfg['encoder'] = r.Resnet11Encoder128x128()
    self.cfg['decoder'] = r.Conv11Decoder128x128()
    self.cfg['activation'] = nn.Tanh()
    self.cfg['model'] = r.EncoderDecoder(self.cfg['encoder'],\
                                         self.cfg['decoder'],\
                                         self.cfg['activation']).cuda(device)\
       if "cuda" in device else r.EncoderDecoder(self.cfg['encoder'],\
                                         self.cfg['decoder'],\
                                         self.cfg['activation']).cpu()
    self.cfg['dataset'] = FileSystemDataset(cfg['input_dir'], cfg['cfg_file'],
                                            cfg['image_dir'], cfg['target_dir'],
                                            cfg['task_name'], cfg['transform'],
                                            cfg['target_transform'])
    self.cfg['optimizer'] = Adam(cfg['model'].parameters(),
                                 lr=cfg['learning_rate'],
                                 weight_decay=cfg['weight_decay'])
    self.cfg['optimizer_d'] = Adam(cfg['model'].parameters(),
                                 lr=cfg['learning_rate'],
                                 weight_decay=cfg['weight_decay'])
    if self.cfg['use_sampler']:
      self.cfg['shuffle'] = False
      if self.cfg['data_wrapper'] is not None:
        dataset_size = len(self.cfg['data_wrapper'](self.cfg['dataset']))
      else:
        dataset_size = len(self.cfg['dataset'])

      indices = list(range(dataset_size))
      split = int(np.floor(self.cfg['validation_split'] * dataset_size))
      np.random.seed(self.cfg['seed']())
      np.random.shuffle(indices)
      training_indices, validation_indices = indices[split:], indices[:split]
      self.cfg['indices'] = indices
      self.cfg['training_indices'] = training_indices
      self.cfg['validation_indices'] = validation_indices
      self.cfg['training_sampler'] = s.MemorySampler(SubsetRandomSampler(training_indices))
      self.cfg['validation_sampler'] = s.MemorySampler(SubsetRandomSampler(validation_indices))
    return self.cfg

