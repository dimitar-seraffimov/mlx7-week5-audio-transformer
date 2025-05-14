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

    # create or Load class_to_idx
    self.class_to_idx = self._get_or_create_class_to_idx()

  def _get_or_create_class_to_idx(self):
    """Load class_to_idx from file if exists, otherwise build and save."""
    mapping_path = os.path.join('checkpoints', 'class_to_idx.json')
    os.makedirs(os.path.dirname(mapping_path), exist_ok=True)

    if os.path.exists(mapping_path):
      with open(mapping_path, 'r') as f:
        class_to_idx = json.load(f)
    else:
      print("[INFO] Building class_to_idx mapping and saving...")
      # create a sorted mapping
      all_classes = sorted(self.metadata['class'].unique())
      class_to_idx = {label: idx for idx, label in enumerate(all_classes)}
      with open(mapping_path, 'w') as f:
        json.dump(class_to_idx, f)
    
    return class_to_idx

  def __len__(self):
    return len(self.metadata)
  
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

    return patches, label_idx