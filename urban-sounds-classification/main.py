import os
from data.get_dataset import get_dataset
from train.train import train_model
from train.evaluate import evaluate_all_folds

def main():
  # check if checkpoint folder exists
  if not os.path.exists("checkpoints"):
    os.makedirs("checkpoints")

  # Step 1: get the dataset
  print("[STEP 1] Getting the dataset...")
  csv_path, audio_dir = download_and_prepare_dataset()

  # Step 2: train the model across 10 folds
  print("[STEP 2] Starting training...")
  train_model(csv_path, audio_dir)

  # Step 3: evaluate the saved models across all folds
  print("[STEP 3] Starting evaluation...")
  evaluate_all_folds()

if __name__ == "__main__":
  main()