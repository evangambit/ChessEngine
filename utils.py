import torch
from torch import nn
import torch.utils.data as tdata
import numpy as np
from sharded_matrix import ShardedLoader

class ExpandedLinear(nn.Linear):
  def __init__(self, in_features, out_features, *args, expansion = None, **kwargs):
    if expansion is not None:
      assert expansion.shape[0] == in_features, f'{expansion.shape[0]} != {in_features}'
      super().__init__(in_features + expansion.shape[1], out_features, *args, **kwargs)
      self.expansion = nn.Parameter(torch.tensor(
        expansion, dtype=torch.float32,
      ), requires_grad=False)
    else:
      super().__init__(in_features, out_features, *args, **kwargs)
      self.expansion = None

  def forward(self, x):
    if self.expansion is not None:
      x = torch.cat([x, x @ self.expansion], 1)
    return super().forward(x)


  def to_linear(self):
    if self.expansion is None:
      merged = self.weight
      dout, din = self.weight.shape
    else:
      dex = self.expansion.shape[1]
      din = self.weight.shape[1] - dex
      dout = self.weight.shape[0]
      merged = self.weight[:, :din] + self.weight[:, din:] @ self.expansion.T
    linear = nn.Linear(din, dout)
    with torch.no_grad():
      linear.weight[:] = merged
      if self.bias is not None:
        linear.bias[:] = self.bias
    return linear

import random
class ShardedMatrixDataset(tdata.IterableDataset):
  def __init__(self, X, Y):
    self.X = X
    assert Y.num_shards == 1, "Multiple Y shards not supported yet"
    Y = Y.load_shard(0)

    rows_per_shard = np.concatenate([[X.cumsum_rows[0]], np.diff(X.cumsum_rows)])
    self.Y = []
    i = 0
    for n in rows_per_shard:
      self.Y.append(Y[i:i+n])
      i += n

  def __iter__(self):
    """
    We don't want to load the entire training set into memory, but we still want to mix
    between shards. The compromise is that we load K shards into memory and randomly
    sample from them. When we've exhausted a shard, we delete it from memory and load
    a new one, randomly selected from the remaining shards.
    """
    kNumShardsAtOnce = 8
    rows_per_shard = np.concatenate([[self.X.cumsum_rows[0]], np.diff(self.X.cumsum_rows)])
    I = []
    for i, n in enumerate(rows_per_shard):
      I.append(np.arange(n))
      np.random.shuffle(I[-1])
    
    waiting_shards = list(range(self.X.num_shards))  # Shards we have yet to sample from.
    active_shards = []  # Shards we're actively sampling from.

    shard2tensor = {}
    shard2idx = {}

    while True:
      while len(active_shards) < kNumShardsAtOnce and len(waiting_shards) > 0:
        shard = waiting_shards.pop()
        active_shards.append(shard)
        shard2idx[shard] = 0
        shard2tensor[shard] = self.X.load_shard(shard)
        assert shard2tensor[shard].shape[0] == I[shard].shape[0], f'{shard}  {shard2tensor[shard].shape}  {I[shard].shape}'
      
      if len(active_shards) == 0:
        break

      shard = random.choice(active_shards)
      idx = shard2idx[shard]
      idx = I[shard][idx]
 
      yield torch.from_numpy(shard2tensor[shard][idx]), float(self.Y[shard][idx]) / 1000.0

      shard2idx[shard] += 1
      if shard2idx[shard] >= I[shard].shape[0]:
        active_shards.remove(shard)
        del shard2idx[shard]
        del shard2tensor[shard]

  def __len__(self):
    return self.X.num_rows
