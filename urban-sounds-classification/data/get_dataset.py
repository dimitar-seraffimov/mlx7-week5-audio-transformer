import os
import pandas as pd
import soundata

def download_and_prepare_dataset(base_dir="urbansound8k"):
    dataset = soundata.initialize('urbansound8k')
    dataset.download()

    real_audio_dir = os.path.join(dataset.default_path, "audio")
    csv_path = os.path.join(base_dir, "UrbanSound8K.csv")

    # using the official metadata
    official_csv_path = os.path.join(dataset.default_path, "metadata", "UrbanSound8K.csv")
    metadata_df = pd.read_csv(official_csv_path)

    # save a copy
    os.makedirs(base_dir, exist_ok=True)
    metadata_df.to_csv(csv_path, index=False)

    print(f"Saved CSV metadata to {csv_path}")

    return csv_path, real_audio_dir