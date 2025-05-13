import torch.nn as nn
import torch.nn.functional as F

#
#
# CNN CLASSIFIER
#
#

class CNNClassifier(nn.Module):
  def __init__(self, embedding_dim, num_classes):
    super().__init__()

    self.conv_block1 = nn.Conv1d(
      in_channels=embedding_dim, # input features per patch
      out_channels=128, # number of output feature maps per patch
      kernel_size=3, # size of the sliding window
      padding=1 # keep the same size after convolution
    )
    self.pool = nn.AdaptiveAvgPool1d(1) # reduce the size of the output to a 1D tensor
    self.fc = nn.Linear(128, num_classes) # fully connected layer

  def forward(self, x):
    # x shape: (batch_size, embedding_dim, n_patches)
    x = self.permute(0, 2, 1) # (batch_size, n_patches, embedding_dim)
    x = F.relu(self.conv_block1(x))
    x = self.pool(x).squeeze(-1) # (batch_size, 128)
    x = self.fc(x)
    return x