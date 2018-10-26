#!/usr/bin/env python3
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import os
import diffuse_albedo_dataset
import convolutional_diffuse_albedo_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists('./dc_img_diffuse_albedo'):
    os.mkdir('./dc_img_diffuse_albedo')


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, 256, 256)
    return x


num_epochs = 200
batch_size = 8
learning_rate = 1e-3

cfg_file = 'config.cfg'
input_dir = '.'
image_dir = 'out'
diffuse_albedo_dir = 'diffuse_albedo'
trans = transforms.Compose([transforms.ToTensor(), \
                            transforms.Normalize((0.5,), (1.0,))])
dataset = diffuse_albedo_dataset.DiffuseAlbedoDataset(input_dir, cfg_file, image_dir,\
                                        diffuse_albedo_dir, trans, trans)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = convolutional_diffuse_albedo_model.ConvolutionalDiffuseAlbedoModel().cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                             weight_decay=1e-5)
print("size = %d" % (len(dataset)))

for epoch in range(num_epochs):
    for data in dataloader:
        img, target = data
        img = Variable(img).cuda()
        target = Variable(target).cuda()
        # ===================forward=====================
        output = model(img)
        loss = criterion(output, target)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch+1, num_epochs, loss.data[0]))
    if epoch % 10 == 0:
        pic = to_img(output.cpu().data)
        save_image(pic, './dc_img_diffuse_albedo/image_{}.png'.format(epoch))

torch.save(model.state_dict(), './conv_diffuse_albedo.pth')
