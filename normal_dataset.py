import cfg
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
import graphics_math

class NormalDataset(data.Dataset):
    def __init__(self, input_dir, cfg_file, image_dir, transform = None,\
                 target_transform = None):
        self.cfg_file = cfg_file
        self.input_dir = input_dir 
        self.image_dir = image_dir
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

    def __len__(self):
        return len(self.flat_cfg)

    def __getitem__(self, idx):
        entry = self.flat_cfg[idx]
        image_path = "%s/%s" % (self.image_dir, entry['rendering'])
        target_path = "%s/%s" % (self.image_dir, entry['normals'])
        img = Image.open(image_path).convert('RGB')
        target = Image.open(target_path).convert('RGB')
        if self.transform is not None:
          img = self.transform(img)
        if self.transform is not None:
          target = self.target_transform(target)
        return img, target

    def get_view_matrix(self, idx):
      eye = np.array([self.flat_cfg[idx]['eye']])
      up = np.array([self.flat_cfg[idx]['up']])
      at = np.array([self.flat_cfg[idx]['at']])
      view_matrix = graphics_math.lookat(eye.transpose(), at.transpose(),\
                                         up.transpose())
      return view_matrix


