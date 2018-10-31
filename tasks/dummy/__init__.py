from datasets.file_system_dataset import FileSystemDataset
from models.convolutional_model import ConvolutionalModel
from torch import nn
from torch.optim import Adam
from torchvision.transforms import Compose
from torchvision.transforms import Normalize
from torchvision.transforms import ToTensor
import torch

cfg = {
    'batch_size' : 8,
    'cfg_file' : 'config.cfg',
    'criterion' : nn.MSELoss(),
    'dc_img' : 'dc_img_dummys',
    'image_dir' : 'out',
    'input_dir' : '.',
    'learning_rate' : 1e-3,
    'num_epochs' : 200,
    'shuffle': True,
    'target_dir': 'dummy',
    'target_transform' : Compose([ToTensor(), Normalize((0.5,), (1.0,))]),
    'task_name': 'dummy',
    'transform' : Compose([ToTensor(), Normalize((0.5,), (1.0,))]),
    'weight_decay': 1e-5,
    'weights_file' : 'dummy.pth',
}

class CfgLoader(object):
  def __init__(self):
    self.cfg = cfg

  def get_cfg(self, device):
    self.cfg['model'] = ConvolutionalModel().cuda(device)\
        if "cuda" in device else ConvolutionalModel.cpu()
    self.cfg['dataset'] = FileSystemDataset(cfg['input_dir'], cfg['cfg_file'],
                                            cfg['image_dir'], cfg['target_dir'],
                                            cfg['task_name'], cfg['transform'],
                                            cfg['target_transform'])
    self.cfg['optimizer'] = Adam(cfg['model'].parameters(),
                                 lr=cfg['learning_rate'],
                                 weight_decay=cfg['weight_decay'])
    return self.cfg

