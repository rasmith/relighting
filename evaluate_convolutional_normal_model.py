#!/usr/bin/env python3
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import os
import normal_dataset
import convolutional_normal_model
import concurrent.futures

if not os.path.exists('./eval_img'):
    os.mkdir('./eval_img')

def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, 256, 256)
    return x

def eval_thread(data):
  img, target, i, m= data
  img = Variable(img).cpu()
  target = Variable(target).cpu()
  output = m(img)
  pic = to_img(output.cpu().data)
  print ("save_image ./eval_img/image_%04d.png" % (i))
  save_image(pic, './eval_img/image_%04d.png' % (i))
  return './eval_img/image_%04d.png' % (i)

cfg_file = 'config.cfg'
input_dir = '.'
image_dir = 'out'
batch_size = 1
trans = transforms.Compose([transforms.ToTensor(), \
                            transforms.Normalize((0.5,), (1.0,))])
dataset = normal_dataset.NormalDataset(input_dir, cfg_file, image_dir,\
                                        trans, trans)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

model = convolutional_normal_model.ConvolutionalNormalModel().cpu()
model.load_state_dict(torch.load('./conv_normal.pth'))
model.eval()

print("Gathering data.")
model_data = [data for data in dataloader]
print("Preparing threads.")
thread_data = [(model_data[i][0], model_data[i][1], i, model) for i in range(len(dataloader))]
print("Running models.")
with concurrent.futures.ThreadPoolExecutor(max_workers = 24) as executor:
  future_to_result = {executor.submit(eval_thread, data):\
      data for data in thread_data}
  for future in concurrent.futures.as_completed(future_to_result):
    try:
      output = future.result()
    except Exception as exc:
      print('%s generated an exception: %s' % (output, exc))
    else:
      print('%s competed successfully.'  % (output))





