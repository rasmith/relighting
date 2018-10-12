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
import normal_model
import context_dataset
import context_model 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_epochs = 5000
batch_size = 100
learning_rate = 1e-3

cfg_file = 'config.cfg'
input_dir = '.'
image_dir = 'out'
image_transform = transforms.Compose([transforms.ToTensor(), \
                            transforms.Normalize((0.5,), (1.0,))])
normal_dataset = normal_dataset.NormalDataset(input_dir, cfg_file, image_dir,\
                                        image_transform, image_transform)
normal_model = normal_model.NormalModel().cpu()
normal_model.load_state_dict(torch.load('./conv_autoencoder.pth'))
normal_model.eval()

dataset =  context_dataset.ContextDataset(normal_dataset, normal_model)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
hidden_size = 20
input_size = dataset.get_input_size()
output_size = dataset.get_output_size()
model = context_model.ContextModel(input_size, hidden_size, output_size).cuda()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                             weight_decay=1e-5)

for epoch in range(num_epochs):
    for data in dataloader:
        context_vector, camera_extrinsic = data
        context_vector = Variable(context_vector).cuda()
        camera_extrinsic = Variable(camera_extrinsic).cuda()
        # ===================forward=====================
        output = model(context_vector)
        loss = criterion(output, camera_extrinsic)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch+1, num_epochs, loss.data[0]))

torch.save(model.state_dict(), './context_vector_nn.pth')
