#!/usr/bin/env python 
from evaluator import Evaluator
from trainer import Trainer
import importlib
import pkgutil
import tasks

package = tasks
device = 'cuda'

for importer, modname, ispkg in pkgutil.iter_modules(package.__path__):
    print('--------------------------------------------')
    print(f'Task:{modname}')
    target_task = importlib.import_module(f'{package.__name__}.{modname}')
    enabled = target_task.cfg['enabled']
    data_wrapper = target_task.cfg['data_wrapper']
    if enabled:
        print(f'{modname} is enabled ... training.')
        cfg_loader = target_task.CfgLoader()
        t = Trainer(cfg_loader, data_wrapper)
        t.init(device)
        t.train()
        print('Evaluating..')
        e = Evaluator(cfg_loader, data_wrapper)
        e.init(device)
        e.evaluate()
    else:
        print(f'{modname} is disabled ... skipping.')

