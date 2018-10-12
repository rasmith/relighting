import glob

def read_cfg(input_dir, cfg_file):
  cfg_path = "%s/%s" % (input_dir, cfg_file)
  with open(cfg_path,'r') as input_file:
    cfg_dict = eval(input_file.read())
  return cfg_dict
