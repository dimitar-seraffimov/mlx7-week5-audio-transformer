import os
from data.get_dataset import download_and_prepare_dataset
from train.train import train_model
from train.evaluate import evaluate_fold

def main():
  # Step 1: get the dataset
  print("[STEP 1] Getting the dataset...")
  CSV_PATH, AUDIO_DIR = download_and_prepare_dataset()

  # Step 2: train the model across 10 folds
  print("[STEP 2] Starting training...")
  train_model(CSV_PATH, AUDIO_DIR)


  # Step 3: evaluate the saved models across all folds
  print("[STEP 3] Starting evaluation...")
  for fold in range(1, 11):
    evaluate_fold(fold)

if __name__ == "__main__":
  main()