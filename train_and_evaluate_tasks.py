#!/usr/bin/env python 
from evaluator import Evaluator
from tensorboardX import SummaryWriter 
from trainers.trainer import Trainer
from trainers.gan_trainer import GanTrainer
import argparse
import importlib
import pkgutil
import tasks
import time
import torch
import sys
from multiprocessing import Process

def run_trainer(package, modname, ispkg, device, log_dir, use_argv):
  print('--------------------------------------------')
  print(f'Task:{modname}')
  # if 'cuda' in device:
      # torch.cuda.set_device(device)
  target_task = importlib.import_module(f'{package.__name__}.{modname}')
  enabled = target_task.cfg['enabled'] \
            if 'enabled' in target_task.cfg else False
  training_enabled = target_task.cfg['training_enabled'] \
            if 'training_enabled' in target_task.cfg else True
  evaluation_enabled = target_task.cfg['evaluation_enabled'] \
            if 'evaluation_enabled' in target_task.cfg else True
  if enabled:
      cfg_loader = target_task.CfgLoader()
      if use_argv:
        if 'required_command_line_arguments' in target_task.cfg:
          parser = argparse.ArgumentParser()
          parser.add_argument(f'--task', help = "Task to run.")
          for keyword in target_task.cfg['required_command_line_arguments']:
            parser.add_argument(f'--{keyword}', help = "")
          args = parser.parse_args()
          for keyword in target_task.cfg['required_command_line_arguments']:
            target_task.cfg[keyword] = getattr(args, keyword)
          if 'log_dir' in target_task.cfg['required_command_line_arguments']:
            log_dir = target_task.cfg['log_dir']
      writer = SummaryWriter(f'{log_dir}/{time.ctime()}')
      if training_enabled:
        print(f'{modname} is enabled for training ...')
        t = GanTrainer(cfg_loader, writer) \
          if target_task.cfg['trainer'] == 'gan' else \
            Trainer(cfg_loader, writer) 
        t.init(device)
        t.train()
        del t
      if evaluation_enabled:
        print(f'{modname} is enabled for evaluation ...')
        e = Evaluator(cfg_loader, writer)
        e.init(device)
        e.evaluate('validation')
        e.evaluate('training')
        del e
      writer.close()
      del writer
      del target_task
  else:
    print(f'{modname} is disabled ... skipping.')

def main():
  package = tasks
  device = 'cuda:0'
  log_dir='tensorboard/log10'

  if len(sys.argv) <= 1:
    for importer, modname, ispkg in pkgutil.iter_modules(package.__path__):
      p = Process(target = run_trainer,\
                  args = (package, modname, ispkg, device, log_dir, False))
      p.start()
      p.join()
  else:
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', help='Task to process.')
    args, argv = parser.parse_known_args()  
    modname = args.task
    ispkg = True
    run_trainer(package, modname, ispkg, device, log_dir, True)

if __name__ == "__main__":
  main()
