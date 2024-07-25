import os

import numpy as np

from sharded_matrix import ShardedLoader, ShardedWriter, linear_regression, MappingLoader, Slice, RowMapper, matmul
from utils import ExpandedLinear, ShardedMatrixDataset

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

def compute_piece_counts(x):
  x = x[:,:-8].reshape((x.shape[0], 12, 8, 8)).sum((2, 3))
  return x

def piece_counts_to_features(x):
  x = x.astype(np.float32)
  return np.concatenate([
    x[:,:5] - x[:,6:11],
    (x[:,1:2] >= 2.0).astype(np.float32) - (x[:,7:8] >= 2.0).astype(np.float32),  # Knight pair
    (x[:,2:3] >= 2.0).astype(np.float32) - (x[:,8:9] >= 2.0).astype(np.float32),  # Bishop pair
    (x[:,3:4] >= 2.0).astype(np.float32) - (x[:,9:10] >= 2.0).astype(np.float32),  # Rook pair
    (x[:,4:5] >= 2.0).astype(np.float32) - (x[:,10:11] >= 2.0).astype(np.float32),  # Queen pair
    np.ones((x.shape[0], 1)),
  ], 1)

def compute_earliness(x):
  return (x[:,1:2] + x[:,2:3] + x[:,3:4] + x[:,4:5] * 3 + x[:,7:8] + x[:,8:9] + x[:,9:10] + x[:,10:11] * 3).clip(0, 18) / 18

def compute_lateness(earliness):
  return 1 - earliness

def add_bias(x):
  return np.concatenate([x, np.ones((x.shape[0], 1), dtype=x.dtype)], 1)

def get_turn(x):
  return x[:, -8:-7] * 2 - 1

def logit(x):
  x = x.astype(np.float32)
  x += 4.0
  x /= 1008.0
  return np.log(x / (1.0 - x))

def table2monocolor(table):
  """
  Convert a table of shape (N, 12, 8, 8) to a table of shape (N, 6, 8, 8) by
  flipping the black pieces and subtracting them from the white pieces.
  """
  table = table.astype(np.int8)
  n = table.shape[0]
  table = table[:,:768].reshape((n, 2, 6, 8, 8))
  return (table[:,0] - np.flip(table[:,1], 2)).reshape((n, -1))

def quadratic_expansion(x):
  X = []
  X.append(np.ones(x.shape[0]))
  X.append(x[:,0] - x[:,6])
  X.append(x[:,1] - x[:,7])
  X.append(x[:,2] - x[:,8])
  X.append(x[:,3] - x[:,9])
  X.append(x[:,4] - x[:,10])

  # # Pawn interaction terms
  # X.append(x[:,0] * x[:,1] - x[:,6] * x[:,7])
  # X.append(x[:,0] * x[:,2] - x[:,6] * x[:,8])
  # X.append(x[:,0] * x[:,3] - x[:,6] * x[:,9])
  # X.append(x[:,0] * x[:,4] - x[:,6] * x[:,10])

  # # Queen interaction terms
  # X.append(x[:,4] * x[:,1] - x[:,10] * x[:,7])
  # X.append(x[:,4] * x[:,2] - x[:,10] * x[:,8])
  # X.append(x[:,4] * x[:,3] - x[:,10] * x[:,9])

  return np.stack(X, 1)

def times(a, b):
  return a * b

def to_signed_y(y, s):
  return logit(y) * s

def minus(a, b):
  return a - b

def early_late(x, time):
  return np.concatenate([x * (1.0 - time), x * time], 1)

def clip(x, low, high):
  return x.clip(low, high)

def clip_grad(x, grad, low, high):
  return (low < x) * (x < high) * grad

def times(a, b):
  return a * b


if __name__ == '__main__':
  a = 'de5-md2'
  X = ShardedLoader(f'data/{a}/data-table')
  F = ShardedLoader(f'data/{a}/data-features')
  Y = ShardedLoader(f'data/{a}/data-eval')
  T = ShardedLoader(f'data/{a}/data-turn')

  print(X.num_shards, X.num_rows / 1_000_000)

  # n = 5_000_000
  # X = Slice(X, 0, n)
  # F = Slice(F, 0, n)
  # Y = Slice(Y, 0, n)
  # T = Slice(T, 0, n)

  if not os.path.exists(f'data/{a}/derived/'):
    os.mkdir(f'data/{a}/derived/')
  os.system(f'rm -f data/{a}/derived/*')

  print('Computing piece counts...')
  PC = MappingLoader(X, compute_piece_counts)
  with ShardedWriter(f'data/{a}/derived/data-piece-counts', shape=(12,), dtype=np.int8) as w:
    for shard in range(PC.num_shards):
      w.write_many(PC.load_shard(shard))
  PC = ShardedLoader(f'data/{a}/derived/data-piece-counts')

  print('Computing mono table')
  MonoTable = MappingLoader(X, table2monocolor)
  MonoTable.save(f'data/{a}/derived/data-mono-table', force=True)
  MonoTable = ShardedLoader(f'data/{a}/derived/data-mono-table')

  print('Done')

  # Derived tables.
  earliness = MappingLoader(PC, compute_earliness)
  lateness = MappingLoader(PC, compute_earliness, compute_lateness)
  SignedY = RowMapper(to_signed_y, Y, T)
  features = RowMapper(early_late, F, lateness)  # cat([F * early, F * late], 1)


  if True:
    w = linear_regression(features, SignedY, regularization=1.0)
  else:
    w = np.zeros((features.shape[0], 1), dtype=np.float32)
  wEarly, wLate = w.reshape(2, -1)

  Yhat = matmul(features, w)
  Yhat.save('/tmp/yhat', dtype=np.float32, force=True)
  Yhat = ShardedLoader('/tmp/yhat')

  Residuals = RowMapper(minus, SignedY, Yhat)

  if False:
    dataset = ShardedMatrixDataset(F, Residuals)

    from torch import nn, optim
    w2 = nn.Linear(F.shape[0], 1)
    nn.init.zeros_(w2.weight)
    opt = optim.SGD(w2.parameters(), lr=0.01, momentum=0.95)

    L = []
    for _ in range(int(5_000_000 // len(dataset)) + 1):
      for x, y in tdata.DataLoader(dataset, batch_size=1024):
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        yhat = w2(x).clip(-1.0, 1.0)
        loss = ((yhat - y)**2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        L.append(float(loss))
      print('%.3f' % np.array(L[-10:]).mean())
    
    wClipped = w2.weight.detach().numpy().squeeze()
    # TODO: adjust residuals base don wClipped
  else:
    wClipped = np.zeros(wEarly.shape)
  

  UnsignedResiduals = RowMapper(times, Residuals, T)
  early = linear_regression(MonoTable, UnsignedResiduals, weights=earliness, regularization=0.01).reshape((6, 8, 8))
  late = linear_regression(MonoTable, UnsignedResiduals, weights=lateness, regularization=0.01).reshape((6, 8, 8))

  # If we never learned wEarly, use piece maps to determine centipawn value.
  if wEarly[0] == 0.0:
    wEarly[0] = early[0,2:-1].mean()

  text = ""
  for k, w in zip(['early', 'late', 'clipped'], [wEarly, wLate, wClipped]):
    text += ('%i' % round(0.0)).rjust(7) + f'  // {k} bias\n'
    for i, varname in enumerate(varnames):
      text += ('%i' % round(float(w[i]) * 100 / wEarly[0])).rjust(7) + f'  // {k} {varname}\n'

  for w in [early, late]:
    text += "// ?\n"
    for _ in range(8):
      text += '0'.rjust(6) * 8 + '\n'
    for i, p in enumerate('PNBRQK'):
      text += "// %s\n" % p
      for r in range(8):
        for f in range(8):
          text += ('%i' % round(w[i,r,f] * 100 / wEarly[0])).rjust(6)
        text += '\n'

  with open('w2.txt', 'w') as f:
    f.write(text)
