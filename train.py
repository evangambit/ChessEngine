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
from utils import ExpandedLinear, ShardedMatrixDataset

"""
a="de4-md1"
a="de5-md2"

python3 generate.py --depth 4 --database positions-de4-md1.db --min_depth=1

sqlite3 "data/${a}/db.sqlite3" "select fen, wins, draws, losses from positions" > "data/${a}/positions.txt"

bin/make_tables "data/${a}/positions.txt" "data/${a}/data"

"""

varnames = [
  "OUR_PAWNS",
  "OUR_KNIGHTS",
  "OUR_BISHOPS",
  "OUR_ROOKS",
  "OUR_QUEENS",
  "THEIR_PAWNS",
  "THEIR_KNIGHTS",
  "THEIR_BISHOPS",
  "THEIR_ROOKS",
  "THEIR_QUEENS",
  "IN_CHECK",
  "KING_ON_BACK_RANK",
  "KING_ON_CENTER_FILE",
  "KING_ACTIVE",
  "THREATS_NEAR_KING_2",
  "THREATS_NEAR_KING_3",
  "PASSED_PAWNS",
  "ISOLATED_PAWNS",
  "DOUBLED_PAWNS",
  "DOUBLE_ISOLATED_PAWNS",
  "PAWNS_CENTER_16",
  "PAWNS_CENTER_4",
  "ADVANCED_PASSED_PAWNS_2",
  "ADVANCED_PASSED_PAWNS_3",
  "ADVANCED_PASSED_PAWNS_4",
  "PAWN_MINOR_CAPTURES",
  "PAWN_MAJOR_CAPTURES",
  "PROTECTED_PAWNS",
  "PROTECTED_PASSED_PAWNS",
  "BISHOPS_DEVELOPED",
  "BISHOP_PAIR",
  "BLOCKADED_BISHOPS",
  "SCARY_BISHOPS",
  "SCARIER_BISHOPS",
  "BLOCKADED_ROOKS",
  "SCARY_ROOKS",
  "INFILTRATING_ROOKS",
  "KNIGHTS_DEVELOPED",
  "KNIGHT_MAJOR_CAPTURES",
  "KNIGHTS_CENTER_16",
  "KNIGHTS_CENTER_4",
  "KNIGHT_ON_ENEMY_SIDE",
  "OUR_HANGING_PAWNS",
  "OUR_HANGING_KNIGHTS",
  "OUR_HANGING_BISHOPS",
  "OUR_HANGING_ROOKS",
  "OUR_HANGING_QUEENS",
  "THEIR_HANGING_PAWNS",
  "THEIR_HANGING_KNIGHTS",
  "THEIR_HANGING_BISHOPS",
  "THEIR_HANGING_ROOKS",
  "THEIR_HANGING_QUEENS",
  "LONELY_KING_IN_CENTER",
  "LONELY_KING_AWAY_FROM_ENEMY_KING",
  "TIME",
  "KPVK_OPPOSITION",
  "SQUARE_RULE",
  "ADVANCED_PAWNS_1",
  "ADVANCED_PAWNS_2",
  "OPEN_ROOKS",
  "ROOKS_ON_THEIR_SIDE",
  "KING_IN_FRONT_OF_PASSED_PAWN",
  "KING_IN_FRONT_OF_PASSED_PAWN2",
  "OUR_MATERIAL_THREATS",
  "THEIR_MATERIAL_THREATS",
  "LONELY_KING_ON_EDGE",
  "OUTPOSTED_KNIGHTS",
  "OUTPOSTED_BISHOPS",
  "PAWN_MOVES",
  "KNIGHT_MOVES",
  "BISHOP_MOVES",
  "ROOK_MOVES",
  "QUEEN_MOVES",
  "PAWN_MOVES_ON_THEIR_SIDE",
  "KNIGHT_MOVES_ON_THEIR_SIDE",
  "BISHOP_MOVES_ON_THEIR_SIDE",
  "ROOK_MOVES_ON_THEIR_SIDE",
  "QUEEN_MOVES_ON_THEIR_SIDE",
  "KING_HOME_QUALITY",
  "BISHOPS_BLOCKING_KNIGHTS",
  "OUR_HANGING_PAWNS_2",
  "OUR_HANGING_KNIGHTS_2",
  "OUR_HANGING_BISHOPS_2",
  "OUR_HANGING_ROOKS_2",
  "OUR_HANGING_QUEENS_2",
  "THEIR_HANGING_PAWNS_2",
  "THEIR_HANGING_KNIGHTS_2",
  "THEIR_HANGING_BISHOPS_2",
  "THEIR_HANGING_ROOKS_2",
  "THEIR_HANGING_QUEENS_2",
  "QUEEN_THREATS_NEAR_KING",
  "MISSING_FIANCHETTO_BISHOP",
  "NUM_BAD_SQUARES_FOR_PAWNS",
  "NUM_BAD_SQUARES_FOR_MINORS",
  "NUM_BAD_SQUARES_FOR_ROOKS",
  "NUM_BAD_SQUARES_FOR_QUEENS",
  "IN_TRIVIAL_CHECK",
  "IN_DOUBLE_CHECK",
  "THREATS_NEAR_OUR_KING",
  "THREATS_NEAR_THEIR_KING",
  "NUM_PIECES_HARRASSABLE_BY_PAWNS",
  "PAWN_CHECKS",
  "KNIGHT_CHECKS",
  "BISHOP_CHECKS",
  "ROOK_CHECKS",
  "QUEEN_CHECKS",
  "BACK_RANK_MATE_THREAT_AGAINST_US",
  "BACK_RANK_MATE_THREAT_AGAINST_THEM",
  "OUR_KING_HAS_0_ESCAPE_SQUARES",
  "THEIR_KING_HAS_0_ESCAPE_SQUARES",
  "OUR_KING_HAS_1_ESCAPE_SQUARES",
  "THEIR_KING_HAS_1_ESCAPE_SQUARES",
  "OUR_KING_HAS_2_ESCAPE_SQUARES",
  "THEIR_KING_HAS_2_ESCAPE_SQUARES",
  "OPPOSITE_SIDE_KINGS_PAWN_STORM",
  "IN_CHECK_AND_OUR_HANING_QUEENS",
  "PROMOTABLE_PAWN",
  "PINNED_PIECES",
]

class Weights:
  def __init__(self, f):
    lines = f.read().split('\n')
    if lines[-1] == '':
      lines.pop()
    self.lines = []
    for line in lines:
      M = re.findall(r"^ +(-?\d+) +//(.+)$", line)
      if M:
        self.lines.append({
          "value": int(M[0][0]),
          "comment": M[0][1],
          "type": "float",
        })
        continue
      M = re.findall(r" +(-?\d+) +(-?\d+) +(-?\d+) +(-?\d+) +(-?\d+) +(-?\d+) +(-?\d+) +(-?\d+)", line)
      if M:
        self.lines.append({
          "value": list(map(int, M[0])),
          "type": "array",
        })
        continue
      assert line.startswith('//'), repr(line)
      self.lines.append({
        "comment": line[2:],
        "type": "comment",
      })

    self.biases = {}
    self.vecs = defaultdict(list)
    for line in self.lines:
       if line['type'] == 'float':
        name = line['comment'].strip().split(' ')[0]
        if ' bias' in line['comment']:
          self.biases[name] = line['value']
        else:
          self.vecs[name].append(line['value'])
    for k in self.vecs:
      self.vecs[k] = np.array(self.vecs[k],  dtype=np.float32)

  def write(self, f):
    for line in self.lines:
      if line["type"] == "float":
        f.write((" %d" % line["value"]).rjust(6) + "  //" + line["comment"] + "\n")
      elif line["type"] == "array":
        A = [str(a).rjust(4) for a in line["value"]]
        f.write(" ".join(A) + "\n")
      elif line["type"] == "comment":
        f.write("//%s\n" % line["comment"])

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
  opt = optim.AdamW(model.parameters(), lr=3e-3)

  with open('w2.txt', 'r') as f:
    weights = Weights(f)

  with torch.no_grad():
    for k in model.w:
      model.w[k].weight *= 0.0
      model.w[k].weight += torch.tensor(weights.vecs[k] / 100, dtype=torch.float32)

  loss_fn = LossFn()

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