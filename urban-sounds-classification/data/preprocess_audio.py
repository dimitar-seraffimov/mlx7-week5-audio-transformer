import torch
import torchaudio.transforms as T

class AudioPreprocessor:
  # initialise required parameters
  def __init__(self, sample_rate=44100, n_mels=64, n_fft=1024, hop_length=256, num_patches=16):
    self.sample_rate = sample_rate
    self.n_mels = n_mels
    self.n_fft = n_fft
    self.hop_length = hop_length
    self.num_patches = num_patches

    self.mel_spectrogram = T.MelSpectrogram(
      sample_rate=self.sample_rate,
      n_mels=self.n_mels,
      n_fft=self.n_fft,
      hop_length=self.hop_length,
    )
  # apply preprocessing to audio to get spectrogram and call _split_into_patches to get patches
  def __call__(self, wave_form):
    # compute mel-spectrogram
    spectrogram = self.mel_spectrogram(wave_form) # (channel, n_mels, time)
    if spectrogram.ndim == 3:
      spectrogram = spectrogram.mean(dim=0) # convert to mono if stereo
    # normalize
    spectrogram = (spectrogram - spectrogram.mean()) / (spectrogram.std() + 1e-6)

    # pad spectrogram if too short
    min_total_width = self.num_patches * 44
    if spectrogram.shape[1] < min_total_width:
      pad_size = min_total_width - spectrogram.shape[1]
      spectrogram = torch.nn.functional.pad(spectrogram, (0, pad_size))

    # patch into 16 segments
    patches = self._split_into_patches(spectrogram)

    return patches
  
  # split spectrogram into 16 segments -> output is a tensor of shape (num_patches, n_mels, patch_length)
  def _split_into_patches(self, spectrogram):
    time_steps = spectrogram.shape[1]
    patch_size = time_steps // self.num_patches
    patches = []

    for i in range(self.num_patches):
      start = i * patch_size
      # for the last patch, take everything until the end
      if i == self.num_patches - 1:
        end = time_steps
      else:
        end = (i + 1) * patch_size
      
      patch = spectrogram[:, start:end]

      # pad if necessary
      if patch.shape[1] < patch_size:
        pad_size = patch_size - patch.shape[1]
        patch = torch.nn.functional.pad(patch, (0, pad_size))
      
      # crop if necessary (very rare with this fix)
      elif patch.shape[1] > patch_size:
        patch = patch[:, :patch_size]

      patches.append(patch)

    patches = torch.stack(patches) # (num_patches, n_mels, patch_length)
    return patches