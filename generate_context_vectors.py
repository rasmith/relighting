#!/usr/bin/env python3
import tasks
import sys
import importlib
from evaluator import Evaluator
target = sys.argv[1]
package = tasks
target_task = importlib.import_module(f'{package.__name__}.{target}')
cfg_loader = target_task.CfgLoader()
cfg = cfg_loader.get_cfg('cpu')
e = Evaluator(cfg_loader, cfg['data_wrapper'])
if len (sys.argv) > 3:
  e.init(sys.argv[3])
else:
  e.init('cuda')
dest_dir = sys.argv[2]
e.encode(dest_dir)
