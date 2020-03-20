#!/usr/bin/env python
# coding: utf-8

# In[283]:


import models.resnet as r


# In[284]:


model = r.resnet18()


# In[285]:


import summary


# In[286]:


#summary.summary(model,input_size=(3, 256, 256));


# In[287]:


import torch


# In[288]:


import torch.nn as nn


# In[289]:


import torch.nn.functional as F


# In[290]:


X=torch.randn((1,3,256,256))


# In[291]:


Y=model(X)


# In[292]:


Y.shape


# In[293]:


fc0 = torch.nn.Linear(512 * 16 *16, 1024*4*4)


# In[294]:


Y=Y.view(-1, 512 * 16 *16)


# In[295]:


Y=F.relu(fc0(Y))


# In[296]:


Y.shape


# In[297]:


Y=Y.view(-1, 1024, 4, 4)


# In[298]:


Y.shape


# In[299]:


c0 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1,padding=1)


# In[300]:


Y=c0(Y)


# In[301]:


Y.shape


# In[302]:


c1 = nn.Conv2d(1024,1024,kernel_size=3, stride=1,padding=1)


# In[303]:


Y=c1(Y)


# In[304]:


Y.shape


# In[305]:


d0 = nn.ConvTranspose2d(1024,1024,kernel_size=3,stride=3,padding=2)


# In[306]:


Y=d0(Y)


# In[307]:


Y.shape


# In[308]:


c2 = nn.Conv2d(1024,1024,kernel_size=3,stride=1,padding=1)


# In[309]:


Y=c2(Y)


# In[310]:


Y.shape


# In[311]:


d1 = nn.ConvTranspose2d(1024,512,kernel_size=2,stride=2,padding=0)


# In[312]:


Y=d1(Y)


# In[313]:


Y.shape


# In[314]:


c3 = nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1)


# In[315]:


Y=c3(Y)


# In[316]:


Y.shape


# In[317]:


d2 = nn.ConvTranspose2d(512, 512, kernel_size = 2, stride = 2, padding = 0)


# In[318]:


Y = d2(Y)


# In[319]:


Y.shape


# In[320]:


c4 = nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1)


# In[321]:


Y=c4(Y)


# In[322]:


Y.shape


# In[323]:


d3 = nn.ConvTranspose2d(512, 256, kernel_size = 2, stride = 2, padding = 0)


# In[324]:


Y=d3(Y)


# In[325]:


Y.shape


# In[326]:


c5 = nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding = 1)


# In[327]:


Y=c5(Y)


# In[328]:


Y.shape


# In[329]:


d4 = nn.ConvTranspose2d(256, 128, kernel_size = 2, stride = 2, padding = 0)


# In[330]:


Y=d4(Y)


# In[331]:


Y.shape


# In[332]:


c6 = nn.Conv2d(128, 128, kernel_size = 3, stride = 1, padding = 1)


# In[333]:


Y=c6(Y)


# In[334]:


Y.shape


# In[335]:


d5 = nn.ConvTranspose2d(128, 64, kernel_size = 2, stride = 2, padding = 0)


# In[336]:


Y=d5(Y)


# In[337]:


Y.shape


# In[338]:


c7 = nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1)


# In[339]:


Y=c7(Y)


# In[340]:


Y.shape


# In[341]:


d6 = nn.ConvTranspose2d(64, 32, kernel_size = 3, stride = 1, padding = 1)


# In[342]:


Y=d6(Y)


# In[343]:


Y.shape


# In[344]:


c8 = nn.Conv2d(32, 3, kernel_size = 3, stride = 1, padding = 1)


# In[345]:


Y=c8(Y)


# In[346]:


Y.shape


# In[ ]:




