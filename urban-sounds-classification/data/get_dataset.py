# get_dataset.py

import os
import pandas as pd
import soundata

def download_and_prepare_dataset(base_dir="urbansound8k"):
    audio_dir = os.path.join(base_dir, "audio")
    csv_path = os.path.join(base_dir, "UrbanSound8K.csv")

    os.makedirs(base_dir, exist_ok=True)

    # download dataset if audio doesn't exist
    if not os.path.exists(audio_dir):
        print("UrbanSound8K audio folder not found. Downloading using soundata...")
        dataset = soundata.initialize('urbansound8k')
        dataset.download()
        print("Download complete.")
    else:
        print("UrbanSound8K audio already found locally. Skipping download.")

    # build the CSV metadata
    print("Building UrbanSound8K CSV metadata...")
    dataset = soundata.initialize('urbansound8k')  # reload to access clips
    metadata = []

    for clip_id in dataset.clip_ids:
        clip = dataset.clip(clip_id)
        metadata.append({
            "slice_file_name": os.path.basename(clip.audio_path),
            "fold": clip.fold,
            "class": clip.tags.labels[0]
        })


    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv(csv_path, index=False)

    print(f"Saved CSV metadata to {csv_path}")
    return csv_path, audio_dir
