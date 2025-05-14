import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm
import wandb
import seaborn as sns
import matplotlib.pyplot as plt
import os
import datetime
from huggingface_hub import hf_hub_download
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
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3
NUM_CLASSES = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
timestamp = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')

wandb.init(
  project='mlx7-week5-sounds-classification', 
  name=f'week5-audio-classification-{timestamp}',
  config={
    "batch_size": BATCH_SIZE,
    "num_epochs": NUM_EPOCHS,
    "learning_rate": LEARNING_RATE,
    "num_classes": NUM_CLASSES,
    "device": DEVICE
  }
)

BASE_DIR = "urbansound8k"
CSV_PATH = os.path.join(BASE_DIR, "UrbanSound8K.csv")
AUDIO_DIR = os.path.join(BASE_DIR, "data")
CHECKPOINT_DIR = "checkpoints"

# download dataset from huggingface if not already downloaded
def download_dataset():
  if not os.path.exists(BASE_DIR):
    os.makedirs(BASE_DIR)
    print(f"Created folder {BASE_DIR}...")
  if not os.path.exists(CSV_PATH):
    hf_hub_download(repo_id="danavery/urbansound8K", filename="UrbanSound8K.csv", cache_dir=BASE_DIR)
    print(f"Downloaded UrbanSound8K.csv to {BASE_DIR}...")
  if not os.path.exists(AUDIO_DIR):
    for i in range(1, 16):
      hf_hub_download(repo_id="danavery/urbansound8K", filename=f"data/train-000{i:02d}-of-00016-03506887d89adfc9.parquet", cache_dir=AUDIO_DIR)
    print(f"Downloaded all audio files to {AUDIO_DIR}...")
# download dataset
download_dataset()

# 10 fold cross validation
all_fold_accuracies = []

#
#
# TRAINING LOOP
#
#

for fold in range(1, 11):
  print(f"Training on fold {fold}...")
  
  # dataset and dataloader
  preprocessor = AudioPreprocessor(n_patches=16)
  train_dataset = UrbanSoundDataset(CSV_PATH, AUDIO_DIR, fold=fold, transform=preprocessor)
  train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)

  # model
  model = UrbanSoundModel(num_layers=6, embedding_dim=64, hidden_dim=128, num_heads=4, num_patches=16, num_classes=NUM_CLASSES)
  model.to(DEVICE)

  # loss function and optimizer
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

  # training loop for each epoch
  model.train()
  wandb.watch(model, log='all')
  for epoch in range(NUM_EPOCHS):
    running_loss = 0.0
    all_predictions = []
    all_labels = []
    loop = tqdm(train_loader, leave=True)

    for batch in loop:
      patches, labels = batch
      patches, labels = patches.to(DEVICE), labels.to(DEVICE)

      optimizer.zero_grad()
      outputs = model(patches)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

      running_loss += loss.item()

      predictions = torch.argmax(dim=1)
      all_predictions.extend(predictions.detach().cpu().numpy())
      all_labels.extend(labels.detach().cpu().numpy())

      # update tqdm loop
      loop.set_description(f"Epoch {epoch+1}/{NUM_EPOCHS}")
      loop.set_postfix(loss=loss.item())
    
    # calculate loss and accuracy
    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = accuracy_score(all_labels, all_predictions)

    # log to wandb
    wandb.log({
      "Fold": fold,
      "Epoch": epoch+1,
      "Loss": epoch_loss,
      "Accuracy": epoch_accuracy
    })
    print(f"Fold {fold} | Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {epoch_loss:.4f} | Accuracy: {epoch_accuracy:.4f}")
  
  # evaluate accuracy on last epoch
  fold_accuracy = accuracy_score(all_labels, all_predictions)
  all_fold_accuracies.append(fold_accuracy)

  # save model checkpoint to wandb artifact
  model_save_path = os.path.join(CHECKPOINT_DIR, f"model_fold_{fold}_{timestamp}.pth")
  torch.save(model.state_dict(), model_save_path)
  wandb.save(model_save_path)

  # log confusion matrix
  conf_matrix = confusion_matrix(all_labels, all_predictions)
  print(f"Confusion Matrix for fold {fold}:")
  print(conf_matrix)

  # log confusion matrix to wandb
  fig, ax = plt.subplots(figsize=(8, 16))
  sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", ax=ax)
  ax.set_xlabel("Predicted")
  ax.set_ylabel("True")
  ax.set_title(f"Confusion Matrix for Fold {fold}")
  wandb.log({f"Confusion Matrix for Fold {fold}": wandb.Image(fig)})
  plt.close(fig)

# after all folds
avg_accuracy = np.mean(all_fold_accuracies)
print(f"\n----------------------------------")
print(f"Average 10-Fold Cross-Validation Accuracy: {avg_accuracy:.4f}")

# create a wandb table to store all fold accuracies
fold_table = wandb.Table(columns=["Fold", "Accuracy"])
for i, accuracy in enumerate(all_fold_accuracies, 1):
  fold_table.add_data(i, accuracy)

wandb.log({
  "10-Fold Cross-Validation Accuracy": fold_table, 
  "Average Accuracy": avg_accuracy
})

# save model checkpoint to wandb artifact
artifact = wandb.Artifact(f'model_fold_{fold}_{timestamp}', type='model')
artifact.add_file(model_save_path)
wandb.log_artifact(artifact)

wandb.finish()