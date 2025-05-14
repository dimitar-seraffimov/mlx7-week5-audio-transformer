import torch
import torch.nn as nn
from models.encoder import TransformerEncoder
from models.cnn_classifier import CNNClassifier

#
#
# FINAL MODEL
#
#

class UrbanSoundModel(nn.Module):
  def __init__(self, num_layers=6, embedding_dim=64, hidden_dim=128, num_heads=4, num_patches=16, num_classes=10):
    super().__init__()
    self.encoder = TransformerEncoder(num_layers, embedding_dim, hidden_dim, num_heads, num_patches)
    self.classifier = CNNClassifier(embedding_dim, num_classes)

  def forward(self, x):
    x = self.encoder(x)
    x = self.classifier(x)
    return x

#
#
#
#
#

if __name__ == "__main__":
  model = UrbanSoundModel()
  dummy_input = torch.randn(32, 16, 64, 44) # batch_size, n_channels, n_mels, patch_length
  output = model(dummy_input)
  print("Output shape:", output.shape)
