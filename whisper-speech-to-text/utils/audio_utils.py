import numpy as np
import librosa
import soundfile as sf
import torch
import torchaudio.transforms as T
from config import SAMPLE_RATE, N_MELS, N_FFT, HOP_LENGTH, CHUNK_DURATION

class AudioPreprocessor:
  def __init__(self):
    self.sample_rate = SAMPLE_RATE
    self.n_mels = N_MELS
    self.n_fft = N_FFT
    self.hop_length = HOP_LENGTH
    self.chunk_duration = CHUNK_DURATION

    self.mel_spectrogram = T.MelSpectrogram(
      sample_rate=self.sample_rate,
      n_mels=self.n_mels,
      n_fft=self.n_fft,
      hop_length=self.hop_length,
    )
  
  def normalize_audio(self, audio):
    return audio / np.max(np.abs(audio))
  
  def load_audio(self, file_path):
    audio, _ = librosa.load(file_path, sr=self.sample_rate, mono=True)
    return self.normalize_audio(audio)
  
  # split audio into chunks of CHUNK_DURATION seconds
  def split_chunks(self, audio):
    chunk_size = int(self.sample_rate * self.chunk_duration)
    chunks = []

    for i in range(0, len(audio), chunk_size):
      chunk = audio[i:i+chunk_size]

      if len(chunk) < chunk_size:
        # pad with zeros at the end if chunk is smaller than chunk_size
        pad_width = chunk_size - len(chunk)
        chunk = np.pad(chunk, (0, pad_width))
      elif len(chunk) > chunk_size:
        # trim chunk if it's longer than chunk_size, trim to match the chunk_size
        chunk = chunk[:chunk_size]
      
      # append the chunk to the list of chunks
      chunks.append(chunk)

    return chunks
  
  # save audio to file
  def save_audio(self, audio, file_path):
    sf.write(file_path, audio, self.sample_rate)
  
  # compute mel spectrogram of audio chunk for training on owner data
  def compute_mel_spectrogram(self, audio_chunk):
    if isinstance(audio_chunk, np.ndarray):
      audio_chunk = torch.tensor(audio_chunk)
    if audio_chunk.ndim == 1:
      audio_chunk = audio_chunk.unsqueeze(0) # add channel dimension
    spectrogram = self.mel_spectrogram(audio_chunk)
    spectrogram = (spectrogram - spectrogram.mean()) / (spectrogram.std() + 1e-6)
    return spectrogram.squeeze(0) # remove channel dimension