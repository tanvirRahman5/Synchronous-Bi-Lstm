import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch

class ClientDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.X = torch.tensor(data["X"], dtype=torch.float32)
        self.y = torch.tensor(data["y"], dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def get_dataloader(npz_path, batch_size=32, shuffle=True):
    dataset = ClientDataset(npz_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
