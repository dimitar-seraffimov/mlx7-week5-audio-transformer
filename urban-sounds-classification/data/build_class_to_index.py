import os
import pandas as pd
import json

CSV_PATH = "urbansound8k/UrbanSound8K.csv"
CLASS_TO_IDX_PATH = "checkpoints/class_to_idx.json"

os.makedirs(os.path.dirname(CLASS_TO_IDX_PATH), exist_ok=True)

# Read the CSV
metadata = pd.read_csv(CSV_PATH)

# Get sorted list of classes
all_classes = sorted(metadata['class'].unique())

# Build class_to_idx mapping
class_to_idx = {label: idx for idx, label in enumerate(all_classes)}

# Save it
with open(CLASS_TO_IDX_PATH, 'w') as f:
    json.dump(class_to_idx, f)

print(f"Saved class_to_idx mapping to {CLASS_TO_IDX_PATH}")
