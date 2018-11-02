from datasets.file_system_dataset import FileSystemDataset
from models.small_convolutional_model import SmallConvolutionalModel
from torch import nn
from torch.optim import Adam
from torchvision.transforms import Compose
from torchvision.transforms import Normalize
from torchvision.transforms import ToTensor
import torch

task_name = 'dummy2'

enabled = True

cfg = {
    'batch_size' : 2,
    'cfg_file' : 'config.cfg',
    'criterion' : nn.MSELoss(),
    'dc_img' : f'dc_img/{task_name}',
    'eval_dir':f'eval/{task_name}',
    'image_dir' : 'out',
    'input_dir' : '.',
    'learning_rate' : 1e-3,
    'num_epochs' : 20,
    'shuffle': True,
    'target_dir': f'target/{task_name}',
    'target_transform' : Compose([ToTensor(), Normalize((0.5,), (1.0,))]),
    'task_name': f'{task_name}',
    'transform' : Compose([ToTensor(), Normalize((0.5,), (1.0,))]),
    'weight_decay': 1e-5,
    'weights_file' : f'weights/{task_name}.pth',
}


class CfgLoader(object):
  def __init__(self):
    self.cfg = cfg

  def get_cfg(self, device):
    self.cfg['model'] = ConvolutionalModel().cuda(device)\
        if "cuda" in device else SmallConvolutionalModel().cpu()
    self.cfg['dataset'] = FileSystemDataset(cfg['input_dir'], cfg['cfg_file'],
                                            cfg['image_dir'], cfg['target_dir'],
                                            cfg['task_name'], cfg['transform'],
                                            cfg['target_transform'])
    self.cfg['optimizer'] = Adam(cfg['model'].parameters(),
                                 lr=cfg['learning_rate'],
                                 weight_decay=cfg['weight_decay'])
    return self.cfg

