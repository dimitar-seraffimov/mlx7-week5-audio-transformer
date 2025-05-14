import torch
import torch.nn as nn
import torch.nn.functional as F

#
#
# ENCODER BLOCK
#
#

class EncoderBlock(nn.Module):
  def __init__(self, embedding_dim, hidden_dim, num_heads):
    super().__init__()

    self.self_attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, batch_first=True)
    self.linear1 = nn.Linear(embedding_dim, hidden_dim)
    self.linear2 = nn.Linear(hidden_dim, embedding_dim)

    self.norm1 = nn.LayerNorm(embedding_dim)
    self.norm2 = nn.LayerNorm(embedding_dim)

    self.dropout = nn.Dropout(0.1)

  def forward(self, x):
    # self-attention
    attention_output, _ = self.self_attention(x, x, x)
    x = x + self.dropout(attention_output)
    x = self.norm1(x)

    # feed-forward network
    feed_forward_output = self.linear2(F.relu(self.linear1(x)))
    x = x + self.dropout(feed_forward_output)
    x = self.norm2(x)

    return x

#
#
# TRANSFORMER ENCODER
#
#

class TransformerEncoder(nn.Module):
  def __init__(self, num_layers, embedding_dim, hidden_dim, num_heads, n_patches):
    super().__init__()

    self.n_patches = n_patches
    self.patch_embedding = nn.Linear(64 * 44, embedding_dim) # n_mels = 64, patch_length = 44

    self.pos_embedding = nn.Parameter(torch.zeros(1, n_patches, embedding_dim))
    self.layers = nn.ModuleList([
      EncoderBlock(embedding_dim, hidden_dim, num_heads) for _ in range(num_layers)
    ])

  def forward(self, x):
    batch_size, n_patches, n_mels, patch_length = x.shape
    x = x.view(batch_size, n_patches, -1) # flatten the (n_mels * patch_length) into a single dimension
    x = self.patch_embedding(x) # (batch_size, n_patches, embedding_dim)
    x = x + self.pos_embedding

    for layer in self.layers:
      x = layer(x)
    
    return x # shape (batch_size, n_patches, embedding_dim)
