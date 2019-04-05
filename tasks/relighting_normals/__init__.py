from datasets.context_vector_dataset import ContextVectorDataset
from torch import nn
from torch.optim import Adam
from torch.utils.data import Subset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.transforms import Compose
from torchvision.transforms import Normalize
from torchvision.transforms import ToTensor
import models as r
import numpy as np
import torch

task_name = 'relighting_normals'

cfg = {
    'batch_size' : 8,
    'cfg_file' : 'config.cfg',
    'criterion' : nn.MSELoss(),
    'data_wrapper' : (lambda x : Subset(x, range(32))),
    'data_wrapper' : None,
    'dc_img' : f'dc_img/{task_name}',
    # 'enabled' : True,
    'enabled' : False,
    'eval_dir':f'eval/{task_name}',
    'image_dir' : f'sources/{task_name}',
    'input_dir' : '.',
    'learning_rate' : 1e-3,
    # 'num_epochs' : 2,
    'num_epochs' : 200,
    'seed': (lambda : 42),
    'shuffle': True,
    'target_dir': f'out',
    'target_transform' : Compose([ToTensor(), Normalize((0.5,), (1.0,))]),
    'task_name': f'{task_name}',
    'transform' : None,
    'use_sampler': True, 
    'validation_split': .2,
    'weight_decay': 1e-5,
    'weights_file' : f'weights/{task_name}.pth'
}

class CfgLoader(object):
  def __init__(self):
    self.cfg = cfg

  def get_cfg(self, device):
    self.cfg['activation'] = nn.Tanh()
    self.cfg['model'] = r.Conv11Decoder128x128(self.cfg['activation']).cuda(device) \
      if "cuda" in device else r.Conv11Decoder128x128(self.cfg['activation']).cpu()
    self.cfg['dataset'] = ContextVectorDataset(cfg['input_dir'], cfg['cfg_file'],
                                            cfg['image_dir'], cfg['target_dir'],
                                            cfg['task_name'], cfg['transform'],
                                            cfg['target_transform'])
    self.cfg['optimizer'] = Adam(cfg['model'].parameters(),
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
      train_indices, validation_indices = indices[split:], indices[:split]
      self.cfg['train_sampler'] = SubsetRandomSampler(train_indices)
      self.cfg['validation_sampler'] = SubsetRandomSampler(validation_indices)
    return self.cfg

