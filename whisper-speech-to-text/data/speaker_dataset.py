import torch
from torch.utils.data import Dataset
import os

class SpeakerDataset(Dataset):
  def __init__(self, owner_path, other_path):
    self.owner_data = torch.load(owner_path)
    self.other_data = torch.load(other_path)

    self.features = torch.cat([self.owner_data, self.other_data], dim=0)
    self.labels = torch.cat([
      torch.zeros(len(self.owner_data)), # owner = 0
      torch.ones(len(self.other_data)) # other = 1
    ]).long()

  # get the length of the dataset
  def __len__(self):
    return len(self.features)
  
  # get the item at index idx
  def __getitem__(self, idx):
    return self.features[idx], self.labels[idx]
