import os
import torch
from tqdm import tqdm
import sys
import os
# add the parent directory to sys.path to allow importing from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.audio_utils import AudioPreprocessor
from config import OWNER_WAV_DIR, PREPROCESSED_OUTPUT_DIR

def preprocess_owner_audio():
  preprocessor = AudioPreprocessor()

  os.makedirs(PREPROCESSED_OUTPUT_DIR, exist_ok=True)
  all_features = []

  wav_files = [f for f in os.listdir(OWNER_WAV_DIR) if f.endswith(".wav")]

  for filename in tqdm(wav_files, desc="Processing owner's WAV audio files..."):
    file_path = os.path.join(OWNER_WAV_DIR, filename)
    audio = preprocessor.load_audio(file_path)
    chunks = preprocessor.split_chunks(audio)

    for idx, chunk in enumerate(chunks):
      spectrogram = preprocessor.compute_mel_spectrogram(chunk)
      all_features.append(spectrogram)

  # stack all features into one tensor
  all_features_tensor = torch.stack(all_features)

  # save tensor
  torch.save(all_features_tensor, os.path.join(PREPROCESSED_OUTPUT_DIR, "owner_mel_spectrograms.pt"))
  print(f"Saved {all_features_tensor.shape[0]} spectrograms to {PREPROCESSED_OUTPUT_DIR}")

if __name__ == "__main__":
  preprocess_owner_audio()
