import cfg
import torch
import torch.utils.data as data
from PIL import Image
import normal_dataset
import normal_model
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np

class ContextDataset(data.Dataset):
    def __init__(self, input_dataset):
        self.normal_dataset = normal_dataset
        self.normal_model = normal_model
        self.context_vectors = []

        dataloader = DataLoader(self.normal_dataset, batch_size=1, shuffle=False)
        for data in dataloader:
          img, target = data
          img = Variable(img).cpu()
          context_vector = normal_model.encode(img)
          self.context_vectors.append(context_vector)

        self.context_vectors = [v.detach().numpy().flatten() \
                                for v in self.context_vectors]
        self.camera_matrices = [normal_dataset.get_view_matrix(i)
                        for i in range(len(normal_dataset))]
        self.camera_matrices = [v.flatten() for v in self.camera_matrices]
        self.input_size = self.context_vectors[0].shape[0]
        self.output_size = self.camera_matrices[0].shape[0]

    def __len__(self):
        return len(self.context_vectors)

    def __getitem__(self, idx):
      return self.context_vectors[idx].astype(np.float32),\
              self.camera_matrices[idx].astype(np.float32)

    def get_input_size(self):
      return self.input_size

    def get_output_size(self):
      return self.output_size



