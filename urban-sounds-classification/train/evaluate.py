import torch
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader
from data.dataset import UrbanSoundDataset
from data.preprocess_audio import AudioPreprocessor
from models.final_model import UrbanSoundModel

#
#
# CONFIG
#
#

BATCH_SIZE = 32
NUM_CLASSES = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = "urbansound8k"
CSV_PATH = os.path.join(BASE_DIR, "UrbanSound8K.csv")
AUDIO_DIR = os.path.join(BASE_DIR, "data")
CHECKPOINT_DIR = "checkpoints"

#
#
# EVALUATE
#
#

# load dataset (full set for evaluation or per fold)
preprocessor = AudioPreprocessor(n_patches=16)

# evaluate a specific fold or all together
fold_to_evaluate = 1

dataset = UrbanSoundDataset(CSV_PATH, AUDIO_DIR, fold=fold_to_evaluate, transform=preprocessor)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# load model
model = UrbanSoundModel(num_layers=6, embed_dim=64, hidden_dim=128, num_heads=4, num_patches=16, num_classes=NUM_CLASSES)
checkpoint_path = os.path.join(CHECKPOINT_DIR, f"model_fold_{fold_to_evaluate}_YOUR_TIMESTAMP.pth")
model.load_state_dict(torch.load(checkpoint_path))
model = model.to(DEVICE)
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
  for patches, labels in dataloader:
    patches, labels = patches.to(DEVICE), labels.to(DEVICE)
    outputs = model(patches)
    preds = outputs.argmax(dim=1)
    all_preds.extend(preds.cpu().numpy())
    all_labels.extend(labels.cpu().numpy())

# calculate metrics
accuracy = accuracy_score(all_labels, all_preds)
print(f"Evaluation Accuracy on Fold {fold_to_evaluate}: {accuracy:.4f}")

# confusion matrix
cm = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:")
print(cm)

# plot confusion matrix
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_xlabel('Predicted Labels')
ax.set_ylabel('True Labels')
ax.set_title(f'Confusion Matrix for Fold {fold_to_evaluate}')
plt.show()
