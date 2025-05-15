## Task 1: Urban Sound Classification

Paper: "A Dataset and Taxonomy for Urban Sound Research", Salamon et al. 2014

### Task Overview

Implement a model to classify urban environmental sounds into predefined categories using the UrbanSound8K dataset.

### Workflow steps:

    - Dataset Handling
        - download UrbanSound8K data via Soundata
        - use the original metadata CSV file mapping audio files to their corresponding labels and folds
    - Audio Preprocessing
        - load waveforms from audio files
        - compute mel-spectrogram representations
        - normalize spectrograms (mean 0, std 1)
    - Feature Encoding
        - split each mel-spectrogram into a fixed number of temporal patches
        - pass patches through a transformer encoder stack
    - Classification - Convolutional Neural Network
        - apply a 1D CNN classifier on top of the encoded patches
        - output logits for each class label
    - Training
        - use PyTorch for model definition, batching, and training loops
        - log and monitor metrics like training loss, validation accuracy, and confusion matrices
    - Evaluation Strategy
        - follow the UrbanSound8K standard evaluation protocol
        - perform 10-fold cross-validation
        - report per-fold and average accuracy
        - visualize confusion matrices for each fold

### Notes
    - I strictly follow the fold-split strategy from UrbanSound8K (no data leakage!)
    - Model checkpoints are saved and managed via W&B artifacts
    - evaluation matches the methodology outlined in Salamon et al. (2014)
