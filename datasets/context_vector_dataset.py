import cfg
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
import graphics_math
import re

class ContextVectorDataset(data.Dataset):
    def __init__(self, input_dir, cfg_file, source_dir, target_dir, task_name,\
                 source_transform = None, target_transform = None):
        self.input_dir = input_dir 
        self.cfg_file = cfg_file
        self.source_dir = source_dir
        self.target_dir = target_dir 
        self.task_name = task_name
        self.source_transform = source_transform
        self.target_transform = target_transform 
        self.cfg = cfg.read_cfg(input_dir, cfg_file)
        for j in self.cfg.keys():
          for k in self.cfg[j].keys():
            self.cfg[j][k].update({'context':'context_%04d.pth' % (int(j)),\
                                   'rendering':'out-%04d-%04d.png'\
                                    % (int(j), int(k)),\
                                    'i': j, 'j' : k})
        self.flat_cfg =  [self.cfg[j][k] for j in self.cfg.keys() \
                                         for k in self.cfg[j].keys()]

    def __len__(self):
        return len(self.flat_cfg)

    def __getitem__(self, idx):
        entry = self.flat_cfg[idx]
        source_path = "%s/%s" % (self.source_dir, entry['rendering'])
        matches  = re.search('out-(\d+)-(\d+).png', source_path)
        target_file_name = "%s-%s-%s.png" % (self.task_name, matches[1], matches[2])
        target_path = "%s/%s" % (self.target_dir, target_file_name)
        img = Image.open(source_path).convert('RGB')
        target = Image.open(target_path).convert('RGB')
        if self.source_transform is not None:
          img = self.source_transform(img)
        if self.target_transform is not None:
          target = self.target_transform(target)
        return img, target



