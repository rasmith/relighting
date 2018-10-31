#!/usr/bin/env python3
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os
import torch


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, 256, 256)
    return x

class Trainer(object):
  def __init__(self, cfg_loader):
  self.cfg_loader = cfg_loader 
  self.task_cfg = {}

  def init(self, which_device):
    self.device = torch.device(which_device\
        if "cuda" in which_device and torch.cuda.is_available() else "cpu")
    self.selected_device = torch.cuda.get_device()\
        if "cuda" in which_device and torch.cuda.is_available() else "cpu"
    print("selected device = %a" % (selected_device))
    self.task_cfg = self.cfg_loader.get_cfg(selected_device)
    if not os.path.exists('./dc_img_ambient_occlusions'):
        os.mkdir('./dc_img_ambient_occlusions')

  def train(self):
    batch_size = self.task_cfg['batch_size']
    criterion = self.task_cfg['criterion']
    dataset = self.task_cfg['dataset']
    dc_img = self.task_cfg['dc_img']
    loss = self.task_cfg['loss']
    model = self.task_cfg['model']
    num_epochs = self.task_cfg['num_epochs']
    optimizer = self.task_cfg['optimizer']
    shuffle = self.task_cfg['shuffle']
    weights_file = self.task_cfg['weights_file']

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffl)
    for epoch in range(num_epochs):
        for data in self.dataloader:
            img, target = data
            img = Variable(img).cuda()\
                if "cuda" in self.selected_device else Variable(img).cpu()
            target = Variable(target).cuda()\
                if "cuda" in self.selected_device else Variable(target).cpu()
            # ===================forward=====================
            output = model(img)
            loss = criterion(output, target)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # ===================log========================
        print(f'epoch [{epoch+1:d}/{num_epochs:d}], loss:{loss.data[0]:.4f}')
        if epoch % 10 == 0:
            pic = to_img(output.cpu().data)
            save_image(pic, f'{dc_img}/image_{}.png')
    torch.save(model.state_dict(), f'./{weights_file}')
