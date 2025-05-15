## Task 1: Urban Sound Classification

paper: "A Dataset and Taxonomy for Urban Sound Research", Salamon et al. 2014

### Overview

Implement a model to classify urban environmental sounds into predefined categories using the UrbanSound8K dataset.

### Tasks

REWRITE!

    - stream from Hugging Face and explore UrbanSound8K
    - preprocess audio: load waveform, compute mel-spectrograms, normalize
    - feature encoding: split each spectrogram into three time patches and pass through encoder blocks
    - classification head: feed encoder outputs into a CNN layer for class logits
    - training loop: integrate with PyTorch for batching, optimization, and evaluation.

### Expected Outcomes

REWRITE!

    - Dataset class for loading audio
    - preprocess_audio function for feature extraction
    - model architecture combining encoder + CNN classification head
    - training script with performance metrics (accuracy, confusion matrix)
