import math
import os
import re

import chess
from chess import pgn

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

import torch
from torch import nn, optim
import torch.utils.data as tdata

from collections import defaultdict

from utils import ExpandedLinear, ShardedMatrixDataset

import chess.engine

"""

sqlite3 positions.db "SELECT fen, wins, draws, losses FROM positions ORDER BY fen ASC" > /tmp/pos.txt
./make_tables /tmp/pos.txt /tmp/pos

sqlite3 positions.remote.db "SELECT fen, wins, draws, losses FROM positions ORDER BY fen ASC" > /tmp/remote.pos.txt
./make_tables /tmp/remote.pos.txt /tmp/remote

import numpy as np
X = np.unpackbits(np.frombuffer(open('/tmp/tables-1', 'rb').read(), dtype=np.uint8)).reshape(-1, 12*64+8)
T = X[:,:-8].reshape(-1, 12, 8, 8)

1B1N1b1r/4pkp1/p4p2/7p/Pp1R3P/1b6/1P3PP1/6K1 b - - 3 28


"""

from tqdm import tqdm
from sharded_matrix import ShardedLoader

a = "de6-md2"
X = ShardedLoader(f'data/{a}/data-table')
Y = ShardedLoader(f'data/{a}/data-eval')

print(f'%.3f million positions loaded' % (X.num_rows / 1_000_000))


dataset = ShardedMatrixDataset(X, Y)
it = iter(dataset)

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
  def __init__(self):
    super().__init__()
    """
         512, 64: 0.04407808091491461
                  0.042864857465028765
                  0.0382  // with classic eval
    """
    expansion = [
    ]

    # Piece differences
    for i in range(5):
      a = np.zeros((12, 8, 8))
      a[i,:,:] = 1
      a[6 + i,:,:] = -1
      expansion.append(a)

    # Rough estimate of material imbalance
    expansion.append(
      expansion[-5] * 1 + expansion[-4] * 3 + expansion[-3] * 3 + expansion[-2] * 5 + expansion[-1] * 9
    )

    # King locations
    for i in [5, 11]:
      x = np.zeros((12, 8, 8))
      x[i,:,:] = np.tile(np.linspace(-1.0, 1.0, 8).reshape(1, 1, -1), (1, 8, 1))
      expansion.append(x)

      x = np.zeros((12, 8, 8))
      x[i,:,:] = np.tile(np.linspace(-1.0, 1.0, 8).reshape(1, -1, 1), (1, 1, 8))
      expansion.append(x)

    expansion = np.concatenate(expansion, 0).reshape(-1, 12 * 8 * 8)

    # Add dummy variables for misc features
    expansion = np.concatenate([expansion, np.zeros((expansion.shape[0], 8))], 1).T

    self.seq = nn.Sequential(
      ExpandedLinear(12 * 8 * 8 + 8, 512, expansion=expansion),
      nn.ReLU(),
      nn.Linear(512, 64),
      nn.ReLU(),
      nn.Linear(64, 1, bias=False),
      nn.Sigmoid(),
    )
    for layer in self.seq:
      if isinstance(layer, nn.Linear):
        nn.init.xavier_normal_(layer.weight)
        if layer.bias is not None:
          nn.init.zeros_(layer.bias)

    # nn.init.zeros_(self.seq[-1].weight)

  # w = w1[:,:-8].reshape((2, 12, 8, 8))
  def forward(self, tables, misc_features):
    tables = tables.reshape(tables.shape[0], -1)
    return self.seq(torch.cat([tables, misc_features], 1))

def forever(loader):
  while True:
    for batch in loader:
      yield batch

class PiecewiseFunction:
  def __init__(self, x, y):
    self.x = np.array(x, dtype=np.float64)
    self.y = np.array(y, dtype=np.float64)
  
  def __call__(self, x):
    if x <= self.x[0]:
      return self.y[0]
    if x >= self.x[-1]:
      return self.y[-1]
    high = np.searchsorted(self.x, x)
    low = high - 1
    t = (x - self.x[low]) / (self.x[high] - self.x[low])
    return self.y[low] * (1 - t) + self.y[high] * t

model = Model()
opt = optim.AdamW(model.parameters(), lr=3e-1, weight_decay=0.01)

loss_fn = nn.MSELoss(reduction='none')

L = []

dataloader = tdata.DataLoader(dataset, batch_size=2048, drop_last=True)
scheduler = PiecewiseFunction(
  [0, 50, len(dataloader) // 2, len(dataloader)],
  [0.0, 3e-2, 3e-3, 3e-4],
)


it = 0
for x, y in tqdm(dataloader):
  for pg in opt.param_groups:
    pg['lr'] = scheduler(it)

  # Unpacking bits into bytes.
  x = x.to(torch.float32)
  y = y.to(torch.float32) / 1000.0

  t = x[:,:768].reshape(-1, 12, 8, 8)
  m = x[:,768:]

  flipped_tables = torch.cat([
    torch.flip(t[:,6::,:,:], (2,)),
    torch.flip(t[:,:6,:,:], (2,)),
  ], 1)
  flipped_misc = torch.zeros(m.shape)
  flipped_misc[:,0] = 1 - m[:,0]  # turn
  flipped_misc[:,1] = m[:,3]
  flipped_misc[:,2] = m[:,4]
  flipped_misc[:,3] = m[:,1]
  flipped_misc[:,4] = m[:,2]

  t = torch.cat([t, flipped_tables], 0)
  m = torch.cat([m, flipped_misc], 0)
  y = torch.cat([y, 1.0 - y], 0)


  yhat = model(t, m)
  loss = loss_fn(yhat.squeeze(), y.squeeze())

  opt.zero_grad()
  loss.mean().backward()
  opt.step()
  L.append((float(loss.mean()), float(loss.std()) / math.sqrt(loss.shape[0])))
  it += 1
  if it % 100 == 0:
    mean, std = L[-1]
    print('%.4f ± %.4f' % (mean, std))

linears = [l for l in model.seq if isinstance(l, nn.Linear)]
linears = [l.to_linear() if isinstance(l, ExpandedLinear) else l for l in linears]

widths =  [l.weight.shape[1] for l in linears]
outfile = 'nnue-' + '-'.join(str(x) for x in widths) + '.bin'

print('writing out to "%s"' % outfile)

w1 = model.seq[0].to_linear().weight.detach().numpy()
w2 = model.seq[2].weight.detach().numpy()
w3 = model.seq[4].weight.detach().numpy()
b1 = model.seq[0].bias.detach().numpy()
b2 = model.seq[2].bias.detach().numpy()

# save
with open(outfile, 'wb') as f:
  f.write(w1.tobytes())
  f.write(w2.tobytes())
  f.write(w3.tobytes())
  f.write(b1.tobytes())
  f.write(b2.tobytes())

# 2048 1.4365 1.4369
