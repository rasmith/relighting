from evaluator import Evaluator
from trainer import Trainer
import importlib
import pkgutil
import tasks

package = tasks
device = 'cpu'

for importer, modname, ispkg in pkgutil.iter_modules(package.__path__):
    print('--------------------------------------------')
    target_task = importlib.import_module(f'{package.__name__}.{modname}')
    enabled = target_task.cfg['enabled']
    data_wrapper = target_task.cfg['data_wrapper']
    if enabled:
        cfg_loader = target_task.CfgLoader()
        t = Trainer(cfg_loader, data_wrapper)
        t.init(device)
        t.train()
        e = Evaluator(cfg_loader, data_wrapper)
        e.init(device)
        e.evaluate()
