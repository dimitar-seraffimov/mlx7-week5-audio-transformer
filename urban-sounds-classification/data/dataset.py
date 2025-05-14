import os
import json
import pandas as pd
import torchaudio
from torch.utils.data import Dataset
# imported here because we provide an instance of it when creating the dataset
from data.preprocess_audio import AudioPreprocessor # never called here, only used in __getitem__

class UrbanSoundDataset(Dataset):
  def __init__(self, csv_path, audio_dir, fold, transform=None):
    """
    Args:
      csv_path (str): path to UrbanSound8K metadata csv
      audio_dir (str): path to audio folder
      fold (int): fold number (1-10)
      transform: audio preprocessing function/class
    """
    self.metadata = pd.read_csv(csv_path)
    self.audio_dir = audio_dir
    self.fold = fold
    self.transform = transform

    # filter metadata for the selected fold
    self.metadata = self.metadata[self.metadata['fold'] == fold]

    # label encoder
    self.class_to_idx = {label: idx for idx, label in enumerate(sorted(self.metadata['class'].unique()))}
    # save class_to_idx mapping once
    if not os.path.exists('checkpoints/class_to_idx.json'):
      with open('checkpoints/class_to_idx.json', 'w') as f:
        json.dump(self.class_to_idx, f)

  # return the number of samples in the dataset
  def __len__(self):
    return len(self.metadata)
  
  # return patches and label for the sample at index idx
  def __getitem__(self, idx):
    sample = self.metadata.iloc[idx]
    file_path = os.path.join(self.audio_dir, f"fold{sample['fold']}", sample['slice_file_name'])

    # load audio
    waveform, sample_rate = torchaudio.load(file_path)
    if sample_rate != self.transform.sample_rate:
      waveform = torchaudio.functional.resample(waveform, sample_rate, self.transform.sample_rate)
    
    # apply preprocessing
    patches = self.transform(waveform)

    # encode label
    label_name = sample['class']
    label_idx = self.class_to_idx[label_name]

    # return patches and label
    return patches, label_idx