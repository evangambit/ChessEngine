import os
import re
from collections import defaultdict

import chess
from chess import pgn

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
import torch.utils.data as tdata

from scipy.ndimage import uniform_filter1d as uf
from tqdm import tqdm

from sharded_matrix import ShardedLoader, ShardedWriter, linear_regression, MappingLoader
from utils import ExpandedLinear, ShardedMatrixDataset, varnames

"""
a="de4-md1"
a="de5-md2"

python3 generate.py --depth 4 --database positions-de4-md1.db --min_depth=1

sqlite3 "data/${a}/db.sqlite3" "select fen, wins, draws, losses from positions" > "data/${a}/positions.txt"

# Probably important, considering sequential positions come from the same game...
# May need to use "shuf" instead of "sort -r"
sort -r "data/${a}/positions.txt" > "data/${a}/positions.shuffled.txt"

bin/make_tables "data/${a}/positions.shuffled.txt" "data/${a}/data"

"""

def lpad(t, n, c=' '):
  t = str(t)
  return max(n - len(t), 0) * c + t

def logit(x):
  if isinstance(x, np.ndarray):
    return np.log(x / (1.0 - x))
  return torch.log(x / (1.0 - x))

cat = np.concatenate

def soft_clip(x, k = 4.0):
  x = nn.functional.leaky_relu(x + k) - k
  x = k - nn.functional.leaky_relu(k - x)
  return x

class LossFn(nn.Module):
  def forward(self, yhat, y):
    yhat = soft_clip(yhat)
    r = yhat - y
    return (r**2).mean()

class Model(nn.Module):
  def __init__(self, d):
    super().__init__()
    self.w = nn.ModuleDict()
    self.w["early"] = nn.Linear(d, 1)
    self.w["late"] = nn.Linear(d, 1)
    self.w["clipped"] = nn.Linear(d, 1)
    for k in self.w:
      nn.init.zeros_(self.w[k].weight)
      nn.init.zeros_(self.w[k].bias)

  def forward(self, x, t):
    early = self.w["early"](x).squeeze() * (1 - t)
    late = self.w["late"](x).squeeze() * t
    clipped = self.w["clipped"](x).squeeze().clip(-1, 1)
    return early + late + clipped

def forever(loader):
  while True:
    for batch in loader:
      yield batch

def compute_piece_counts(x):
  x = x[:,:-8].reshape((x.shape[0], 12, 8, 8)).sum((2, 3))
  return x

def compute_earliness(x):
  return (x[:,1:2] + x[:,2:3] + x[:,3:4] + x[:,4:5] * 3 + x[:,7:8] + x[:,8:9] + x[:,9:10] + x[:,10:11] * 3).clip(0, 18) / 18

def compute_lateness(earliness):
  return 1 - earliness

if __name__ == '__main__':
  a = "de5-md2"
  A = ShardedLoader(f'data/{a}/data-table')
  X = ShardedLoader(f'data/{a}/data-features')
  Y = ShardedLoader(f'data/{a}/data-eval')
  T = ShardedLoader(f'data/{a}/data-turn')
  PC = MappingLoader(A, compute_piece_counts)
  earliness = MappingLoader(PC, compute_earliness)
  lateness = MappingLoader(PC, compute_earliness, compute_lateness)

  dataset = ShardedMatrixDataset(X, Y, T)

  print('Setting up model...')
  model = Model(X.shape[0])
  opt = optim.AdamW(model.parameters(), lr=0.0)

  with open('weights.txt', 'r') as f:
    weights = Weights(f)

  with torch.no_grad():
    for k in model.w:
      model.w[k].weight *= 0.0
      model.w[k].weight += torch.tensor(weights.vecs[k] / 100, dtype=torch.float32)

  loss_fn = LossFn()

  L = []
  for bs in (np.linspace(8, 12, 3)**2).astype(np.int64).tolist():
    for pg in opt.param_groups:
      pg['lr'] = 0.0
    it = 0
    for x, y, turn in tqdm(tdata.DataLoader(dataset, batch_size=bs)):
      it += 1
      if it == 20:
        for pg in opt.param_groups:
          pg['lr'] = 3e-4
      x = x.float()
      time = x[:, varnames.index('TIME')].clip(0.0, 18.0) / 18.0
      y = logit((y.float().squeeze() + 1.0) / 1002.0) * turn.float().squeeze()
      yhat = model(x, time).squeeze()
      loss = loss_fn(yhat, y)
      opt.zero_grad()
      loss.backward()
      opt.step()
      L.append(float(loss))
      if len(L) % 10000 == 0:
        print((f'Loss: %.4f' % np.array(L[-10000:]).mean()).ljust(12), len(L))

  print('Computing residuals')
  with ShardedWriter('/tmp/res', dtype=np.float32, shape=(1,)) as w:
    for shard in tqdm(np.arange(A.num_shards)):
      i = A.shard_to_slice_indices(shard)
      a = torch.tensor(A.load_shard(shard), dtype=torch.float32)
      x = torch.tensor(X.load_slice(*i), dtype=torch.float32)
      y = torch.tensor(Y.load_slice(*i), dtype=torch.float32)
      t = torch.tensor(T.load_slice(*i), dtype=torch.float32).squeeze()
      res = logit((y + 1.0) / 1002.0).squeeze() - model(x, x[:, varnames.index('TIME')]) * t
      res = res.detach().numpy()
      w.write_many(res.reshape(-1, 1))
  
  Res = ShardedLoader('/tmp/res')
  
  print('Performing regression for piece squares')
  wEarly = linear_regression(A, Y, weights=earliness, regularization=10.0, num_workers=4).squeeze()
  wEarly = wEarly[:-8].reshape((12, 8, 8))

  wLate = linear_regression(A, Y, weights=lateness, regularization=10.0, num_workers=4).squeeze()
  wLate = wLate[:-8].reshape((12, 8, 8))

  wEarly = (wEarly[0:6] - np.flip(wEarly, 1)[6:12]) / 2
  wLate = (wLate[0:6] - np.flip(wLate, 1)[6:12]) / 2

  text = ""
  for k in ['early', 'late', 'clipped']:
    text += ('%i' % round(float(model.w[k].bias) * 100)).rjust(7) + f'  // {k} bias\n'
    for i, varname in enumerate(varnames):
      text += ('%i' % round(float(model.w[k].weight[0,i]) * 100)).rjust(7) + f'  // {k} {varname}\n'

  for w in [wEarly, wLate]:
    text += """// ?
     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0
"""
    for i, p in enumerate('PNBRQK'):
      text += "// %s\n" % p
      for r in range(8):
        for f in range(8):
          text += ('%i' % round(w[i,r,f])).rjust(6)
        text += '\n'

  with open('w2.txt', 'w+') as f:
    f.write(text)

  print(text)