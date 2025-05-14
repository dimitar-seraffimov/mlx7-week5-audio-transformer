# get_dataset.py

import os
import pandas as pd
import soundata

def download_and_prepare_dataset(base_dir="urbansound8k"):
    dataset = soundata.initialize('urbansound8k')
    dataset.download()

    real_audio_dir = os.path.join(dataset.dataset_path, "audio")  # the audio location
    csv_path = os.path.join(base_dir, "UrbanSound8K.csv")

    # build the CSV metadata
    metadata = []
    for clip_id in dataset.clip_ids:
        clip = dataset.clip(clip_id)
        metadata.append({
            "slice_file_name": os.path.basename(clip.audio_path),
            "fold": clip.fold,
            "class": clip.tags[0].value if clip.tags else "unknown"
        })

    os.makedirs(base_dir, exist_ok=True)
    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv(csv_path, index=False)

    return csv_path, real_audio_dir