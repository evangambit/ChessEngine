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

"""

sqlite3 positions.db "SELECT fen FROM positions ORDER BY fen ASC" > /tmp/pos.txt
sqlite3 positions.db "SELECT wins, draws, losses FROM positions ORDER BY fen ASC" > /tmp/evals.txt

sqlite3 positions.remote.db "SELECT fen FROM positions ORDER BY fen ASC" > /tmp/remote.pos.txt
sqlite3 positions.remote.db "SELECT wins, draws, losses FROM positions ORDER BY fen ASC" > /tmp/remote.evals.txt
./make_tables /tmp/remote.pos.txt /tmp/remote.evals.txt /tmp/remote


./make_tables /tmp/toy.pos.txt /tmp/toy.evals.txt /tmp/toy.data

./make_tables /tmp/pos.txt /tmp/evals.txt /tmp/data

import numpy as np
X = np.unpackbits(np.frombuffer(open('/tmp/tables-1', 'rb').read(), dtype=np.uint8)).reshape(-1, 12*64+8)
T = X[:,:-8].reshape(-1, 12, 8, 8)

1B1N1b1r/4pkp1/p4p2/7p/Pp1R3P/1b6/1P3PP1/6K1 b - - 3 28


"""

from tqdm import tqdm

def load_shard(n, prefix='data'):
  path = os.path.join('/tmp/', f'{prefix}-{str(n).rjust(5, "0")}')
  score_path = os.path.join('/tmp/', f'{prefix}-scores-{str(n).rjust(5, "0")}')
  raw_features = np.frombuffer(open(path, 'rb').read(), dtype=np.uint8)
  labels = np.frombuffer(open(score_path, 'rb').read(), dtype=np.int16)
  n = labels.shape[0]
  raw_features = raw_features.reshape((n, -1))
  tables = raw_features[:,:-1]
  misc_features = np.unpackbits(raw_features[:,-1], bitorder='little').reshape(n, 8)
  return tables, misc_features, labels

print('loading')
tables, misc_features, labels = [], [], []
for i in tqdm(list(range(122))):
  a, b, c = load_shard(i + 1)
  tables.append(a)
  misc_features.append(b)
  labels.append(c)

for i in tqdm(list(range(77))):
  a, b, c = load_shard(i + 1, 'remote')
  tables.append(a)
  misc_features.append(b)
  labels.append(c)

tables = np.concatenate(tables)
misc_features = np.concatenate(misc_features)
labels = np.concatenate(labels)

print(f'%.3f million positions loaded' % (tables.shape[0] / 1_000_000))

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
     512, 64: 0.04689538702368736
    """
    self.seq = nn.Sequential(
      nn.Linear(12 * 8 * 8 + 8, 512),
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
opt = optim.AdamW(model.parameters(), lr=3e-2, weight_decay=0.01)

tables = torch.tensor(tables, dtype=torch.int8)
misc_features = torch.tensor(misc_features, dtype=torch.int8)
labels = torch.tensor(labels, dtype=torch.int16)

dataset = tdata.TensorDataset(tables, misc_features, labels)

loss_fn = nn.MSELoss()

L = []

dataloader = tdata.DataLoader(dataset, batch_size=2048, shuffle=True, drop_last=True)
scheduler = PiecewiseFunction(
  [0, 50, len(dataloader)],
  [0.0, 3e-3, 1e-4],
)


it = 0
for t, m, y in tqdm(dataloader):
  # Unpacking bits into bytes.
  t = t.detach().numpy().astype(np.uint8)
  t = np.unpackbits(t, bitorder='little').reshape((t.shape[0], 12, 8, 8))
  t = torch.from_numpy(t)

  t = t.to(torch.float32)
  m = m.to(torch.float32)
  y = y.to(torch.float32)

  for pg in opt.param_groups:
    pg['lr'] = scheduler(it)

  flipped_tables = torch.cat([
    torch.flip(t[:,:6:,:,:], (2,)),
    torch.flip(t[:,:6,:,:], (2,)),
  ], 1)
  flipped_misc = torch.zeros(m.shape)
  flipped_misc[:,-1] = 1 - m[:,-1]  # turn
  flipped_misc[:,-2] = m[:,-4]
  flipped_misc[:,-3] = m[:,-5]
  flipped_misc[:,-4] = m[:,-2]
  flipped_misc[:,-5] = m[:,-3]

  t = torch.cat([t, flipped_tables], 0)
  m = torch.cat([m, flipped_misc], 0)
  y = torch.cat([y, 1000 - y], 0)


  yhat = model(t, m)
  loss = loss_fn(yhat.squeeze(), y / 1000)

  opt.zero_grad()
  loss.backward()
  opt.step()
  L.append(float(loss))
  it += 1
  if it % 100 == 0:
    print(sum(L[-100:]) / 100)

test_raw = np.frombuffer(open('alice-00001', 'rb').read(), dtype=np.uint8)
test_raw = np.unpackbits(test_raw, bitorder='little')
test_raw = test_raw.reshape(5, -1)
x = test_raw[:,:-8].reshape((5, 12, 8, 8))
m = test_raw[:,-8:]
x = torch.from_numpy(x)
m = torch.from_numpy(m)
x = torch.cat([x.reshape((5, -1)), m], 1).to(torch.float32)
print(model.seq(x))

"""
rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1
rnbqkbnr/pppppppp/8/8/3PP3/2N2N2/PPP2PPP/R1BQKB1R w KQkq - 5 5
rnbqkb1r/ppp1pppp/5n2/3p4/5P2/8/PPPPP1PP/RNBQKBNR b KQkq - 3 3
r1b1k2r/ppq1bppp/2n2n2/2ppp3/8/8/PPPPPPPP/RNBQKBNR b KQkq - 3 8

Stockfish's depth=10 evaluations are

 +50
 -50
+218
 -80
-323

tensor([[0.5061],
        [0.5058],
        [0.7971],
        [0.3706],
        [0.1827]], grad_fn=<SigmoidBackward0>)


import numpy as np
A = np.arange(15, dtype=np.float32).reshape(3, 5)
f = open('a.bin', 'wb')
f.write(A.T.tobytes())


FILE *f = fopen(filename.c_str(), "wb");
Matrix<float, 3, 5> w;
fwrite(w.data(), sizeof(float), w.size(), f);
fclose(f);


"""

w1 = model.seq[0].weight.detach().numpy()
w2 = model.seq[2].weight.detach().numpy()
w3 = model.seq[4].weight.detach().numpy()
b1 = model.seq[0].bias.detach().numpy()
b2 = model.seq[2].bias.detach().numpy()

# save
with open('nnue.bin', 'wb') as f:
  f.write(w1.tobytes())
  f.write(w2.tobytes())
  f.write(w3.tobytes())
  f.write(b1.tobytes())
  f.write(b2.tobytes())

# 2048 1.4365 1.4369
