import inspect
import json
import os
import uuid

import numpy as np

import torch
from torch import nn, optim
import torch.utils.data as tdata

from collections import defaultdict

"""

sqlite3 positions.db "SELECT fen, wins, draws, losses FROM positions ORDER BY fen ASC" > /tmp/pos.txt
./make_tables /tmp/pos.txt /tmp/pos

sqlite3 positions.remote.db "SELECT fen, wins, draws, losses FROM positions ORDER BY fen ASC" > /tmp/remote.pos.txt
./make_tables /tmp/remote.pos.txt /tmp/remote

import numpy as np
X = np.unpackbits(np.frombuffer(open('/tmp/tables-1', 'rb').read(), dtype=np.uint8)).reshape(-1, 12*64+8)
T = X[:,:-8].reshape(-1, 12, 8, 8)

1B1N1b1r/4pkp1/p4p2/7p/Pp1R3P/1b6/1P3PP1/6K1 b - - 3 28


sqlite3 data/de7-md2/db.sqlite3 'select * from positions where abs(random()) % 4 = 0' > data/de7-md2/pos.txt
shuf data/de7-md2/pos.txt > data/de7-md2/pos.shuf.txt
./make_tables data/de7-md2/pos.shuf.txt data/de7-md2/data

"""

from tqdm import tqdm
from sharded_matrix import ShardedLoader, ShardedLoader
import chess

def x2board(vec):
  assert len(vec.shape) == 1
  assert vec.shape[0] == 37
  board = chess.Board.empty()
  board.turn = False
  castling = ''
  for val in vec:
    if val < 768:
      piece = val // 64
      val = val % 64
      y = 7 - val // 8
      x = val % 8
      sq = chess.square(x, y)
      board.set_piece_at(sq, chess.Piece(piece % 6 + 1, chess.WHITE if piece < 6 else chess.BLACK))
    if val == 768:
      board.turn = True
    if val == 769:
      castling += 'K'
    if val == 770:
      castling += 'Q'
    if val == 771:
      castling += 'k'
    if val == 772:
      castling += 'q'
  # board.set_castling_fen(castling) doesn't work?
  parts = board.fen().split(' ')
  parts[2] = castling if castling else '-'
  return chess.Board(' '.join(parts))

def board2x(board):
  vec = np.ones(37, dtype=np.int16) * 776
  i = 0
  for sq, piece in board.piece_map().items():
    val = 0
    val += 'PNBRQKpnbrqk'.index(piece.symbol()) * 64
    val += 8 * (7 - sq // 8)
    val += sq % 8
    vec[i] = val
    i += 1
  if board.turn:
    vec[i] = 768
    i += 1
  if board.has_kingside_castling_rights(chess.WHITE):
    vec[i] = 769
    i += 1
  if board.has_queenside_castling_rights(chess.WHITE):
    vec[i] = 770
    i += 1
  if board.has_kingside_castling_rights(chess.BLACK):
    vec[i] = 771
    i += 1
  if board.has_queenside_castling_rights(chess.BLACK):
    vec[i] = 772
    i += 1
  assert i <= 37
  vec.sort()
  return vec



class SimpleIterablesDataset(tdata.IterableDataset):
  def __init__(self, xpath, ypath):
    self.X = ShardedLoader(xpath)
    self.Y = ShardedLoader(ypath)

  def __iter__(self):
    xi, yi = 0, 0
    xj, yj = 0, 0
    x = self.X.load_shard(xi)
    y = self.Y.load_shard(yi)
    while True:
      if xj == x.shape[0]:
        xi += 1
        if xi >= self.X.num_shards:
          break
        x = self.X.load_shard(xi)
        xj = 0
      if yj == y.shape[0]:
        yi += 1
        y = self.Y.load_shard(yi)
        yj = 0
      yield x[xj].copy(), y[yj].copy()
      xj += 1
      yj += 1
  
  def __len__(self):
    return self.X.num_rows

class CReLU(nn.Module):
  def forward(self, x):
    return x.clip(0, 1)

class Emb(nn.Module):
  def __init__(self, dout):
    super().__init__()
    din = 776
    self.misc = nn.Parameter(torch.randn(8, dout) * 0.01)

    self.tiles = nn.Parameter(torch.randn(12, 8, 8, dout) * 0.01)
    self.coord = nn.Parameter(torch.randn(1, 8, 8, dout) * 0.01)
    self.piece = nn.Parameter(torch.randn(12, 1, 1, dout) * 0.01)
    self.row = nn.Parameter(torch.randn(1, 8, 1, dout) * 0.01)
    self.col = nn.Parameter(torch.randn(1, 1, 8, dout) * 0.01)
    self.tilecolor = nn.Parameter(torch.randn(1, 1, 1, dout) * 0.01)

    self.white_tile_mask = torch.zeros(1, 8, 8, 1)
    for y in range(8):
      for x in range(8):
        self.white_tile_mask[0, y, x, 0] = (y + x) % 2 == 0

    self.zeros = nn.Parameter(torch.zeros(1, dout))
    self.bias = nn.Parameter(torch.zeros(dout))
  
  @property
  def weight(self):
    T = (
      self.tiles + self.coord + self.piece + self.row + self.col
      + (self.tilecolor * self.white_tile_mask)
    ).reshape(12 * 8 * 8, -1)
    return torch.cat([T, self.misc, self.zeros], 0)
  
  def forward(self, x):
    return self.weight[x.to(torch.int32)].sum(1) + self.bias
  
  def forward2(self, x):
    return self.weight[x.to(torch.int32)]  # (x.shape[0], 37, dout)

class Model(nn.Module):
  def __init__(self):
    super().__init__()

    k1, k2, k3 = 128, 32, 32
    # 13317it [06:52, 34.86it/s]train: 0.0381 test: 0.0314 (16, 16)

    self.seq = nn.Sequential(
      Emb(k1),
      CReLU(),
      nn.Linear(k1, k2),
      CReLU(),
      nn.Linear(k2, k3),
      CReLU(),
      nn.Linear(k3, 1, bias=False),
    )
    for layer in self.seq:
      if isinstance(layer, nn.Linear):
        nn.init.xavier_normal_(layer.weight)
        if layer.bias is not None:
          nn.init.zeros_(layer.bias)

  def forward(self, x):
    penalty = 0.0
    for layer in self.seq:
      x = layer(x)
      if isinstance(layer, nn.Linear) and layer is not self.seq[-1]:
        scale = 256
        quant = 65_536
        low = quant // 2
        shift = quant * 5 + low

        x = x * scale

        # We penalize any value that is more than 0.5 away from the nearest quantization point, just to be safe.
        # In theory any value up to "low - 1" is technically okay.
        penalty += ((torch.relu(torch.abs(x) - low * 0.5) / scale)**2).mean()

        x = x + torch.rand(x.shape, device=x.device) - 0.5
        x = ((x + shift) % quant - quant // 2) / scale
    return x, penalty
  
  def layer_outputs(self, x):
    r = []
    for layer in self.seq:
      x = layer(x)
      if isinstance(layer, (nn.Linear, Emb)):
        r.append(x)
    return r

class Model(nn.Module):
  def __init__(self, k):
    super().__init__()
    self.seq = nn.Sequential(
      nn.Linear(k, 128),
      nn.LeakyReLU(0.1),
      nn.Linear(128, 1, bias=False),
    )
    nn.init.xavier_normal_(self.seq[0].weight)
    nn.init.xavier_normal_(self.seq[2].weight)

  def forward(self, x):
    x = nn.functional.leaky_relu(x)
    x = self.seq(x)  # (n, 1)
    return x, torch.zeros(1, device=x.device)


  # 5000: train: 0.0312 test: 0.0263
  def layer_outputs(self, x):
    return []


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

def interweaver(*A):
  """
  Merges multiple data loaders and loops over them endlessly.

  Example:

  # Run 1 test batch every 5 training batches.
  for split, (x, y) in interweaver(
    ("train", 5, trainloader),
    ("test", 1, testloader),
    ):
  """
  n = len(A)
  names = [a[0] for a in A]
  steps = [a[1] for a in A]
  loaders = [a[2] for a in A]
  iters = [iter(a) for a in loaders]
  i = 0
  while True:
    name = names[i % n]
    for _ in range(steps[i % n]):
      try:
        yield name, loaders[i % n].dataset, next(iters[i % n])
      except StopIteration:
        iters[i % n] = iter(loaders[i % n])
        yield name, loaders[i % n].dataset, next(iters[i % n])
    i += 1

trainset = SimpleIterablesDataset(f'data/de6-md2/data-nnue', f'data/de6-md2/data-eval')
testset = SimpleIterablesDataset(f'data/de7-md2/data-nnue', f'data/de7-md2/data-eval')
print(f'train: %.3fM   test: %.3fM' % (len(trainset) / 1_000_000, len(testset) / 1_000_000))

# train: 0.0357 test: 0.0294

k = 256
emb = Emb(k)
model = Model(k)
opt = optim.AdamW(list(model.parameters()) + list(emb.parameters()), lr=0.0, weight_decay=0.1)

def loss_fn(yhat: torch.Tensor, y: torch.Tensor):
  return (torch.abs(torch.sigmoid(yhat) - y)**2.5)

L = []

kBatchSize = 2048
trainloader = tdata.DataLoader(trainset, batch_size=kBatchSize, drop_last=True)
testloader = tdata.DataLoader(testset, batch_size=kBatchSize, drop_last=True)
maxlr = 0.01
scheduler = PiecewiseFunction(
  [0, 20, len(trainloader) // 2, len(trainloader)],
  [0.0, maxlr, maxlr * 0.1, maxlr * 0.01]
)

metrics = defaultdict(list)
it = 0
for split, _, (x, y) in tqdm(interweaver(
    ("train", 5, trainloader),
    ("test", 1, testloader),
  ), total=len(trainloader)):
  if split == 'train':
    it += 1
    if it % 1000 == 999:
      print(it, len(trainloader), sum(metrics['train:loss'][-50:]) / 50)
  lr = scheduler(it)
  for pg in opt.param_groups:
    pg['lr'] = lr

  # # Unpacking bits into bytes.
  # x = x.to(torch.float32)
  y = y.to(torch.float32) / 1000.0

  x = emb(x)
  yhat, penalty = model(x)
  penalty = penalty * 0.0  # Ignore the penalty for now.

  loss = loss_fn(yhat.squeeze(), y.squeeze())

  residuals = torch.sigmoid(yhat.squeeze()) - y.squeeze()

  if split == 'train':
    opt.zero_grad()
    (loss.mean() + penalty).backward()
    opt.step()
  metrics[f'{split}:loss'].append(loss.mean().item())
  metrics[f'{split}:penalty'].append(penalty.item())
  if split == 'train' and it % 50 == 0:
    train_loss = sum(metrics[f'train:loss'][-10:]) / len(metrics[f'train:loss'][-10:])
    test_loss = sum(metrics[f'test:loss'][-10:]) / len(metrics[f'test:loss'][-10:])
    print('train: %.4f test: %.4f' % (train_loss, test_loss))
  
  if it >= len(trainloader):
    break

torch.jit.trace(model, torch.zeros(size=(1, 256))).save("my_module.pt")
# torch.jit.script(model).save("my_module.pt")
with open('my_module.emb', 'wb') as f:
  mat = emb.weight.detach().numpy()[:-1]
  bias = emb.bias.detach().numpy()
  f.write(np.array([len(mat.shape)], dtype=np.int32).tobytes())
  f.write(np.array(mat.shape, dtype=np.int32).tobytes())
  f.write(mat.tobytes())
  f.write(np.array([len(bias.shape)], dtype=np.int32).tobytes())
  f.write(np.array(bias.shape, dtype=np.int32).tobytes())
  f.write(bias.tobytes())

