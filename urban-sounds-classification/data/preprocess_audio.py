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
    if spectrogram.shape[0] == 1:
      spectrogram = spectrogram.mean(dim=0, keepdim=True) # convert to mono if stereo
    
    spectrogram = spectrogram.squeeze(0) # (n_mels, time)
    # normalize
    spectrogram = (spectrogram - self.mean()) / (spectrogram.std() + 1e-6)
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
      end = (i + 1) * patch_size if i < self.num_patches - 1 else time_steps
      patch = spectrogram[:, start:end]

      # if patches is shorter than patch_size, pad with zeros
      if patch.shape[1] < patch_size:
        pad_size = patch_size - patch.shape[1]
        patch = torch.nn.functional.pad(patch, (0, pad_size))

      patches.append(patch)
  
    patches = torch.stack(patches) # (num_patches, n_mels, patch_length)
    return patches