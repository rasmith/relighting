#!/usr/bin/env python 
from evaluator import Evaluator
from tensorboardX import SummaryWriter 
from trainers.trainer import Trainer
from trainers.gan_trainer import GanTrainer
import importlib
import pkgutil
import tasks
import time
import torch


def main():
  package = tasks
  device = 'cuda'
  log_dir='tensorboard/log'

  for importer, modname, ispkg in pkgutil.iter_modules(package.__path__):
      print('--------------------------------------------')
      print(f'Task:{modname}')
      target_task = importlib.import_module(f'{package.__name__}.{modname}')
      enabled = target_task.cfg['enabled'] \
                if 'enabled' in target_task.cfg else False
      training_enabled = target_task.cfg['training_enabled'] \
                if 'training_enabled' in target_task.cfg else True
      evaluation_enabled = target_task.cfg['evaluation_enabled'] \
                if 'evaluation_enabled' in target_task.cfg else True
      if enabled:
          writer = SummaryWriter(f'{log_dir}/{time.ctime()}')
          cfg_loader = target_task.CfgLoader()
          if training_enabled:
            print(f'{modname} is enabled for training ...')
            t = GanTrainer(cfg_loader, writer) \
              if target_task.cfg['trainer'] == 'gan' else \
                Trainer(cfg_loader, writer) 
            t.init(device)
            t.train()
          if evaluation_enabled:
            print(f'{modname} is enabled for evaluation ...')
            e = Evaluator(cfg_loader, writer)
            e.init(device)
            e.evaluate('validation')
            e.evaluate('training')
          writer.close()
      else:
          print(f'{modname} is disabled ... skipping.')

if __name__ == "__main__":
  main()
