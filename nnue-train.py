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

model = Model()
opt = optim.AdamW(model.parameters(), lr=0.0, weight_decay=0.1)

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

  # t = x[:,:768].reshape(-1, 12, 8, 8)
  # m = x[:,768:]

  # flipped_tables = torch.cat([
  #   torch.flip(t[:,6::,:,:], (2,)),
  #   torch.flip(t[:,:6,:,:], (2,)),
  # ], 1)
  # flipped_misc = torch.zeros(m.shape)
  # flipped_misc[:,0] = 1 - m[:,0]  # turn
  # flipped_misc[:,1] = m[:,3]
  # flipped_misc[:,2] = m[:,4]
  # flipped_misc[:,3] = m[:,1]
  # flipped_misc[:,4] = m[:,2]
  # flipped_y = 1.0 - y

  # t = torch.cat([t, flipped_tables], 0)
  # m = torch.cat([m, flipped_misc], 0)
  # y = torch.cat([y, flipped_y], 0)

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

layer_outputs = model.layer_outputs(x.to(torch.int32))
for layer in layer_outputs:
  layer = layer.detach().numpy().flatten()
  print(np.percentile(layer, 1), layer.mean(), layer.std(), np.percentile(layer, 99))

linears = [l for l in model.seq if isinstance(l, (nn.Linear, Emb))]

widths =  [(l.weight.shape[1] if isinstance(l, nn.Linear) else l.weight.shape[0]) for l in linears]
run_id = uuid.uuid4().hex
outfile = os.path.join('runs', run_id, 'nnue-' + '-'.join(str(x) for x in widths) + '.bin')
os.makedirs(os.path.dirname(outfile), exist_ok=True)

with open(os.path.join('runs', run_id, 'config.txt'), 'w') as f:
  lines = [
    'Widths',
    ' '.join(str(x) for x in widths),
    '',
    'Train Loss: %.3f' % (np.array(metrics['train:loss'][-100:]).mean()),
    'Test Loss: %.3f' % (np.array(metrics['test:loss'][-100:]).mean()),
    '',
    'Schedule',
    json.dumps(scheduler.x.tolist()),
    json.dumps(scheduler.y.tolist()),
    '',
    'Batch size: %d\n' % kBatchSize,
    '',
    'Loss',
    inspect.getsource(loss_fn),
  ]
  f.write('\n'.join(lines))

print('writing out to "%s"' % outfile)

w1 = model.seq[0].weight.T.detach().numpy()[:,:-1]
w2 = model.seq[2].weight.detach().numpy()
w3 = model.seq[4].weight.detach().numpy()
b1 = model.seq[0].bias.detach().numpy()
b2 = model.seq[2].bias.detach().numpy()

# save
with open(outfile, 'wb') as f:
  for mat in [w1, w2, w3, b1, b2]:
    f.write(np.array([len(mat.shape)], dtype=np.int32).tobytes())
    f.write(np.array(mat.shape, dtype=np.int32).tobytes())
    f.write(mat.tobytes())

# 2048 1.4365 1.4369
