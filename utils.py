import torch
from torch import nn
import torch.utils.data as tdata
import numpy as np
from sharded_matrix import ShardedLoader

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
  "KNOWN_KPVK_DRAW",
  "KNOWN_KPVK_WIN",
  "LONELY_KING_ON_EDGE_AND_NOT_DRAW",
  "LONELY_KING_IN_CORNER_AND_NOT_DRAW",
  "LONELY_KING_OPPOSITION_AND_NOT_DRAW",
  "LONELY_KING_ACHIEVABLE_OPPOSITION_AND_NOT_DRAW",
  "LONELY_KING_NEXT_TO_ENEMY_KING",
  "KING_TROPISM",
]

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
import torch
import torch.utils.data as tdata
class ShardedMatrixDataset(tdata.IterableDataset):
  def __init__(self, X, *Y):
    self.X = X
    self.Y = Y

  def __iter__(self):
    """
    We don't want to load the entire training set into memory, but we still want to mix
    between shards. The compromise is that we load K shards into memory and randomly
    sample from them. When we've exhausted a shard, we delete it from memory and load
    a new one, randomly selected from the remaining shards.
    """
    kNumShardsAtOnce = 8

    cumsum_rows = np.cumsum(self.X.num_rows_in_shards())
    rows_per_shard = np.concatenate([[cumsum_rows[0]], np.diff(cumsum_rows)])
    I = []
    for i, n in enumerate(rows_per_shard):
      I.append(np.arange(n))
      np.random.shuffle(I[-1])
    
    waiting_shards = list(range(self.X.num_shards))  # Shards we have yet to sample from.
    active_shards = []  # Shards we're actively sampling from.

    shard2tensors = {}
    shard2idx = {}

    while True:
      while len(active_shards) < kNumShardsAtOnce and len(waiting_shards) > 0:
        shard = waiting_shards.pop()
        active_shards.append(shard)
        shard2idx[shard] = 0
        x = self.X.load_shard(shard)
        indices = self.X.shard_to_slice_indices(shard)
        shard2tensors[shard] = (x,) + tuple([y.load_slice(*indices) for y in self.Y])
        for y in shard2tensors[shard][1:]:
          assert x.shape[0] == y.shape[0]
      
      if len(active_shards) == 0:
        break

      shard = random.choice(active_shards)
      idx = shard2idx[shard]
      idx = I[shard][idx]
 
      s = shard2tensors[shard]
      yield tuple([torch.from_numpy(y[idx]) for y in s])

      shard2idx[shard] += 1
      if shard2idx[shard] >= I[shard].shape[0]:
        active_shards.remove(shard)
        del shard2idx[shard]
        del shard2tensors[shard]

  def __len__(self):
    return self.X.num_rows

def table2fen(A, white_to_move):
  assert A.shape == (12, 8, 8)
  lines = []
  occ = A.sum(0)
  P = 'PNBRQKpnbrqk'
  for y in range(8):
    lines.append('')
    count = 0
    for x in range(8):
      if occ[y,x] == 0:
        count += 1
      else:
        if count != 0:
          lines[-1] += str(count)
          count = 0
        lines[-1] += P[A[:,y,x].argmax()]
    if count != 0:
      lines[-1] += str(count)
  return '/'.join(lines) + (' w ' if white_to_move else ' b ') + '- - 1 1'

import re
from collections import defaultdict
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
    for name in ['early', 'late', 'clipped']:
      f.write((" %d" % self.biases[name]).rjust(7) + "  // bias\n")
      for i, v in enumerate(self.vecs[name]):
        f.write((" %d" % v).rjust(7) + f"  // {name} {varnames[i]}\n")
    for line in self.lines:
      if line["type"] == "array":
        A = [str(a).rjust(5) for a in line["value"]]
        f.write(" " + " ".join(A) + "\n")
      elif line["type"] == "comment":
        f.write("//%s\n" % line["comment"])

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
  "KNOWN_KPVK_DRAW",
  "KNOWN_KPVK_WIN",
  "LONELY_KING_ON_EDGE_AND_NOT_DRAW",
  "LONELY_KING_IN_CORNER_AND_NOT_DRAW",
  "LONELY_KING_OPPOSITION_AND_NOT_DRAW",
  "LONELY_KING_ACHIEVABLE_OPPOSITION_AND_NOT_DRAW",
  "LONELY_KING_NEXT_TO_ENEMY_KING",
  "KING_TROPISM",
  "PAWNS_X_QUEENS",
  "PAWNS_X_QUEENS_2",
  "PAWNS_X_KNIGHTS",
  "PAWNS_X_KNIGHTS_2",
  "PAWNS_X_BISHOPS",
  "PAWNS_X_BISHOPS_2",
  "PAWNS_X_ROOKS",
  "PAWNS_X_ROOKS_2",
  "KNIGHTS_X_QUEENS",
  "KNIGHTS_X_QUEENS_2",
  "BISHOPS_X_QUEENS",
  "BISHOPS_X_QUEENS_2",
  "ROOKS_X_QUEENS",
  "ROOKS_X_QUEENS_2",
]