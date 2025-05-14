import os
from train.train import train_model
from train.evaluate import evaluate_all_folds

def main():
  # check if checkpoint folder exists
  if not os.path.exists("checkpoints"):
    os.makedirs("checkpoints")

  # Step 1: train the model across 10 folds
  print("Starting training...")
  train_model()

  # Step 2: evaluate the saved models across all folds
  print("Starting evaluation...")
  evaluate_all_folds()

if __name__ == "__main__":
  main()