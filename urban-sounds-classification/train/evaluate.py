import torch
import numpy as np
import json
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os
import wandb
import tempfile
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
AUDIO_DIR = os.path.join(BASE_DIR, "audio")
CHECKPOINT_DIR = "checkpoints"
CLASS_MAP_PATH = os.path.join(CHECKPOINT_DIR, "class_to_idx.json")

WANDB_PROJECT = "mlx7-week5-sounds-classification"
WANDB_RUN_ID = "o413ibhm"

# wandb API
api = wandb.Api()
run = api.run(f"mlx7-dimitar-projects/{WANDB_PROJECT}/{WANDB_RUN_ID}")

#
#
# EVALUATE ALL FOLDS

def evaluate_all_folds():
  fold_accuracies = []
  # Load class mapping  
  with open(CLASS_MAP_PATH, "r") as f:
    class_to_idx = json.load(f)
  idx_to_class = {v: k for k, v in class_to_idx.items()}
  class_labels = [idx_to_class[i] for i in range(NUM_CLASSES)]

  for fold in range(1, 11):
    print(f"\nEvaluating fold {fold}...")
    preprocessor = AudioPreprocessor(num_patches=16)
    dataset = UrbanSoundDataset(CSV_PATH, AUDIO_DIR, fold=fold, transform=preprocessor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # download checkpoint artifact
    artifact_name = f"model_fold{fold}_2025_05_14__16_05_56:v0"
    artifact = api.artifact(f"mlx7-dimitar-projects/{WANDB_PROJECT}/{artifact_name}")
    with tempfile.TemporaryDirectory() as temp_dir:
      model_path = artifact.download(root=temp_dir)
      model_file = os.path.join(model_path, f"model_fold_{fold}.pth")

      # load model
      model = UrbanSoundModel(num_layers=6, embedding_dim=64, hidden_dim=128, num_heads=4, num_patches=16, num_classes=NUM_CLASSES)
      model.load_state_dict(torch.load(model_file))
      model = model.to(DEVICE)
      model.eval()

      # evaluate
      all_predictions = []
      all_labels = []

      with torch.no_grad():
        for patches, labels in dataloader:
          patches, labels = patches.to(DEVICE), labels.to(DEVICE)
          outputs = model(patches)
          predictions = outputs.argmax(dim=1)

          all_predictions.extend(predictions.cpu().numpy())
          all_labels.extend(labels.cpu().numpy())

      # calculate metrics
      accuracy = accuracy_score(all_labels, all_predictions)
      fold_accuracies.append(accuracy)
      print(f"Fold {fold} accuracy: {accuracy:.4f}")

      # confusion matrix
      conf_matrix = confusion_matrix(all_labels, all_predictions)
      fig, ax = plt.subplots(figsize=(10, 8))
      sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels, ax=ax)
      ax.set_xlabel('Predicted Labels')
      ax.set_ylabel('True Labels')
      ax.set_title(f'Confusion Matrix for Fold {fold}')
      plt.show()
    
    avg_accuracy = np.mean(fold_accuracies)
    print(f"\nAverage accuracy across all folds: {avg_accuracy:.4f}")


if __name__ == "__main__":
  evaluate_all_folds()