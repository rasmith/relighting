#!/usr/bin/env python 
from evaluator import Evaluator
from tensorboardX import SummaryWriter 
from trainer import Trainer
import importlib
import pkgutil
import tasks
import time

package = tasks
device = 'cuda'
log_dir='tensorboard/log'

for importer, modname, ispkg in pkgutil.iter_modules(package.__path__):
    print('--------------------------------------------')
    print(f'Task:{modname}')
    target_task = importlib.import_module(f'{package.__name__}.{modname}')
    enabled = target_task.cfg['enabled']
    data_wrapper = target_task.cfg['data_wrapper']
    if enabled:
        print(f'{modname} is enabled ... training.')
        cfg_loader = target_task.CfgLoader()
        writer = SummaryWriter(f'{log_dir}/{time.ctime()}')
        t = Trainer(cfg_loader, data_wrapper, writer)
        t.init(device)
        t.train()
        print('Evaluating..')
        e = Evaluator(cfg_loader, data_wrapper, writer)

        e.init(device)
        e.evaluate()
        writer.close()
    else:
        print(f'{modname} is disabled ... skipping.')

