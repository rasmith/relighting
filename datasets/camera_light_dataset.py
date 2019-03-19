import cfg
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from graphics_math import lookat
import re

def normalize_numpy_array(values):
  max_vals = values[0]
  min_vals = values[0]
  for m in values:
    max_vals = np.maximum(max_vals, m)
    min_vals = np.minimum(min_vals, m)
  divisor = np.maximum(max_vals - min_vals, np.ones(max_vals.shape))
  normalized_values = (values - min_vals) / divisor - 0.5
  return normalized_values
  
class CameraLightDataset(data.Dataset):
    def __init__(self, input_dir, cfg_file, target_dir, task_name,\
                 transform = None, target_transform = None):
        self.input_dir = input_dir 
        self.cfg_file = cfg_file
        self.target_dir = target_dir 
        self.task_name = task_name
        self.transform = transform
        self.target_transform = target_transform 
        self.cfg = cfg.read_cfg(input_dir, cfg_file)
        for j in self.cfg.keys():
          for k in self.cfg[j].keys():
            self.cfg[j][k].update({'normals':'normal_%04d.png' % (int(j)),\
                                   'rendering':'out-%04d-%04d.png'\
                                    % (int(j), int(k)),\
                                    'i': j, 'j' : k})
        self.flat_cfg =  [self.cfg[j][k] for j in self.cfg.keys() \
                                         for k in self.cfg[j].keys()]
        self.last = None
        self.matrices = [lookat(np.array([entry['eye']]).transpose(),\
                                np.array([entry['at']]).transpose(),\
                                np.array([entry['up']]).transpose())\
                          for entry in self.flat_cfg]
        self.matrices = normalize_numpy_array(self.matrices)
        self.lights = [entry['light'] for entry in self.flat_cfg]
        self.lights = normalize_numpy_array(self.lights)

    def __len__(self):
        return len(self.flat_cfg)

    def __getitem__(self, idx):
        self.last = idx
        m = np.reshape(self.matrices[idx], (1, 16)).astype(np.float32)
        l = np.reshape(self.lights[idx], (1, 3)).astype(np.float32)
        ml = torch.from_numpy(np.concatenate((m, l), axis = 1)).view(1, 1, 19)
        entry = self.flat_cfg[idx]
        image_path = entry['rendering']
        matches  = re.search('out-(\d+)-(\d+).png', image_path)
        target_file_name = "%s-%s-%s.png" % ('normals', matches[1], matches[2])
        target_path = "%s/%s" % (self.target_dir, target_file_name)
        target = Image.open(target_path).convert('RGB')
        if self.transform is not None:
          ml = self.transform(ml)
        if self.target_transform is not None:
          target = self.target_transform(target)
        return ml, target
