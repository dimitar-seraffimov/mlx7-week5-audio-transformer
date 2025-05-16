from datasets import Features, Audio, load_dataset
import numpy as np
import torch
from tqdm import tqdm
from itertools import islice
import os
import librosa

print(torch.cuda.is_available())
SAMPLE_RATE = 44100
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256
CLIP_DURATION = 3.0
TARGET_DURATION = 3 * 60 * 60
BATCH_SIZE = 32
OUTPUT_PATH = "preprocessed/other_mel_spectrograms.pt"

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

mel_filterbank = torch.tensor(
    librosa.filters.mel(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS),
    dtype=torch.float32
).cuda()

def extract_mel_spectrogram_batch(waveforms):
    window = torch.hann_window(N_FFT).cuda()
    stft = torch.stft(waveforms, n_fft=N_FFT, hop_length=HOP_LENGTH, window=window,
                      return_complex=True)
    power_spec = stft.abs() ** 2
    mel_spec = torch.matmul(mel_filterbank, power_spec)
    mel_spec_db = 10 * torch.log10(mel_spec + 1e-10)
    return mel_spec_db

def pad_or_trim_waveform(waveform, target_length):
    length = waveform.shape[-1]
    if length > target_length:
        return waveform[..., :target_length]
    elif length < target_length:
        padding = target_length - length
        return torch.nn.functional.pad(waveform, (0, padding))
    return waveform

def main():
    print("Loading Common Voice dataset from Hugging Face...")

    features = Features({"audio": Audio(sampling_rate=SAMPLE_RATE)})
    dataset = load_dataset(
        "mozilla-foundation/common_voice_17_0",
        "en",
        split="train",
        streaming=True,
        trust_remote_code=True,
        features=features
    )

    dataset = dataset.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))
    dataset = dataset.remove_columns([col for col in dataset.features if col != "audio"])
    dataset = dataset.with_format("python")

    stream = islice(dataset, 200_000)
    mel_spectrograms = []
    total_duration = 0.0
    buffer = []

    for sample in tqdm(stream, desc="Processing Common Voice samples", total=200_000):
        if total_duration >= TARGET_DURATION:
            break
        try:
            audio = sample["audio"]
            waveform = torch.tensor(audio["array"], dtype=torch.float32)
            if waveform.ndim > 1:
                waveform = waveform.mean(dim=0)
            waveform = pad_or_trim_waveform(waveform, int(CLIP_DURATION * SAMPLE_RATE))
            buffer.append(waveform)

            if len(buffer) >= BATCH_SIZE:
                batch = torch.stack(buffer).cuda()
                mels = extract_mel_spectrogram_batch(batch)
                mel_spectrograms.extend(mels.cpu())
                total_duration += CLIP_DURATION * BATCH_SIZE
                buffer = []
        except Exception as e:
            print(f"Error: {e}")

    if buffer:
        batch = torch.stack(buffer).cuda()
        mels = extract_mel_spectrogram_batch(batch)
        mel_spectrograms.extend(mels.cpu())

    torch.save(mel_spectrograms, OUTPUT_PATH)
    print(f"\nSaved {len(mel_spectrograms)} mel spectrograms to {OUTPUT_PATH} (~{total_duration/3600:.2f} hours of audio)")

if __name__ == '__main__':
    main()
