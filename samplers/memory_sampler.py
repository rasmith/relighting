from torch.utils.data.sampler import Sampler

class MemorySampler(Sampler):
    def __init__(self, sampler):
        self.sampler = sampler
        self.indices = list(i for i in sampler)
    def __iter__(self):
        return (i for i in self.indices)
    def __len__(self):
        return len(self.indices)
    def __call__(self, idx):
        return self.indices[idx]
