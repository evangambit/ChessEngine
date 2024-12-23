import os
from typing import Union

import numpy as np

from sharded_matrix import ShardedLoader, ShardedWriter, linear_regression, MappingLoader, Slice, RowMapper, matmul, curry, compute_inner_product, LoaderInterface
from utils import ExpandedLinear, ShardedMatrixDataset, varnames, table2fen

from functools import partial

"""
python3 generate.py --depth 6 --min_depth=2

sqlite3 "data/de6-md2/db.sqlite3" "select * from positions where abs(random()) % 2 = 0" > "data/de6-md2/pos-40M-20M.txt"

# Probably important, considering sequential positions come from the same game...
# May need to use "shuf" instead of "sort -r"
shuf data/de6-md2/pos-40M-20M.txt > data/de6-md2/pos-40M-20M.shuf.txt

bin/make_tables "data/${a}/positions.shuffled.txt" "data/${a}/data"

"""

def compute_earliness(x):
  return (x[:,1:2] + x[:,2:3] + x[:,3:4] + x[:,4:5] * 3 + x[:,6:7] + x[:,7:8] + x[:,8:9] + x[:,9:10] * 3).clip(0, 18) / 18

def compute_material_inequality(x):
  a =  (x[:, varnames.index('OUR_KNIGHTS')] - x[:, varnames.index('THEIR_KNIGHTS')]) * 3
  a += (x[:, varnames.index('OUR_BISHOPS')] - x[:, varnames.index('THEIR_BISHOPS')]) * 3
  a += (x[:, varnames.index('OUR_ROOKS')] - x[:, varnames.index('THEIR_ROOKS')]) * 5
  a += (x[:, varnames.index('OUR_QUEENS')] - x[:, varnames.index('THEIR_QUEENS')]) * 9
  return a.clip(-2, 2).reshape((-1, 1))

def compute_lateness(earliness):
  return 1 - earliness

def add_bias(x):
  return np.concatenate([x, np.ones((x.shape[0], 1), dtype=x.dtype)], 1)

def get_turn(x):
  return x[:, -8:-7] * 2 - 1

def table2monocolor(table):
  """
  Convert a table of shape (N, 12, 8, 8) to a table of shape (N, 6, 8, 8) by
  flipping the black pieces and subtracting them from the white pieces.
  """
  table = table.astype(np.int8)
  n = table.shape[0]
  table = table[:,:768].reshape((n, 2, 6, 8, 8))
  return (table[:,0] - np.flip(table[:,1], 2)).reshape((n, -1))

def times(a, b):
  return a * b

def to_signed_y(y, s):
  return y * s

def minus(a, b):
  return a - b

def row_merger(x, time, inequality):
  return np.concatenate([x * (1.0 - time), x * time, x * inequality], 1)

def clip(x, low, high):
  return x.clip(low, high)

def clip_grad(x, grad, low, high):
  return (low < x) * (x < high) * grad

def times(a, b):
  return a * b

def slice(X, start, end):
  return X[:, start:end]

def plus(a, b):
  return a + b

def minus(a, b):
  return a - b

import enum
class JobType(enum.Enum):
  ALL_FROM_SCRATCH = 0
  LABELS_FROM_SCRATCH = 1
  FEATURES_FROM_SCRATCH = 2
  ALL_CACHED = 3
  ZERO = 4

"""
perl -ne 'print if (rand() < .1)' data/de6-md2/positions-160M.txt > data/de6-md2/pos.sample.txt

shuf data/de6-md2/pos.sample.txt > data/de6-md2/pos.sample.shuf.txt

rm data/de6-md2/data-*

./make_tables data/de6-md2/pos.sample.shuf.txt data/de6-md2/data
"""

def logit(x):
  x = x.astype(np.float32)
  x += 4.0
  x /= 1008.0
  return np.log(x / (1.0 - x))

def maybe_delete_matrix(path):
  path, name = os.path.split(path)
  for fn in os.listdir(path):
    if fn.startswith(name + '-'):
      os.remove(os.path.join(path, fn))

if __name__ == '__main__':
  a = 'de6-md2'
  X = None # ShardedLoader(f'data/{a}/data-table')
  F = ShardedLoader(f'data/{a}/data-features')
  Y = ShardedLoader(f'data/{a}/data-eval')
  T = ShardedLoader(f'data/{a}/data-turn')
  PC = ShardedLoader(f'data/{a}/data-piece-counts')
  MonoTable = ShardedLoader(f'data/{a}/data-pst')

  # limit = 10_000_000
  # if limit is not None:
  #   X = Slice(X, 0, limit) if X is not None else None
  #   F = Slice(F, 0, limit)
  #   Y = Slice(Y, 0, limit)
  #   T = Slice(T, 0, limit)
  #   PC = Slice(PC, 0, limit)
  #   MonoTable = Slice(MonoTable, 0, limit)

  W = None  # Optional weights for each datapoint
  # Y = MappingLoader(Y, logit)

  print(F.num_rows, Y.num_rows, T.num_rows, PC.num_rows, MonoTable.num_rows)
  assert X is None or T.num_rows == X.num_rows
  assert T.num_rows == F.num_rows
  assert T.num_rows == Y.num_rows
  assert T.num_rows == PC.num_rows
  assert T.num_rows == MonoTable.num_rows
  assert W is None or T.num_rows == W.num_rows

  Y = MappingLoader(Y, logit)

  print(Y.num_rows / 1_000_000, F.num_shards)

  # Derived tables.
  earliness = MappingLoader(PC, compute_earliness)
  maybe_delete_matrix('/tmp/earliness')
  earliness.save('/tmp/earliness', dtype=np.float32)
  earliness = ShardedLoader('/tmp/earliness')

  lateness = MappingLoader(earliness, compute_lateness)

  inequality_ness = MappingLoader(F, compute_material_inequality)
  maybe_delete_matrix('/tmp/ineq')
  inequality_ness.save('/tmp/ineq', dtype=np.float32)
  inequality_ness = ShardedLoader('/tmp/ineq')

  SignedY = RowMapper(to_signed_y, Y, T)  # From mover's perspective
  features = RowMapper(row_merger, F, lateness, inequality_ness)

  # TODO: add back in a bias term and see if that helps.
  avg = SignedY.load_slice(0, SignedY.num_rows).mean()
  SignedY = RowMapper(partial(minus, b=avg), SignedY)

  num_workers = 1

  if not os.path.exists(f'data/{a}/derived'):
    os.makedirs(f'data/{a}/derived')

  cov = compute_inner_product(features, features, num_workers=num_workers, weights_loader=W)
  np.save(f'data/{a}/derived/cov.npy', cov)
  # cov = np.load(f'data/{a}/derived/cov.npy')

  dot = compute_inner_product(features, SignedY, num_workers=num_workers, weights_loader=W)
  np.save(f'data/{a}/derived/dot.npy', dot)
  # dot = np.load(f'data/{a}/derived/dot.npy')

  w = np.linalg.solve(cov + np.eye(cov.shape[0]) * 200, dot)
  wEarly, wLate, inEq = w.reshape(3, -1)

  print('YHAT')
  Yhat = matmul(features, w)
  maybe_delete_matrix('/tmp/yhat')
  Yhat.save('/tmp/yhat', dtype=np.float32)
  Yhat = ShardedLoader('/tmp/yhat')

  print('RESIDUALS')
  Residuals = RowMapper(minus, SignedY, Yhat)
  maybe_delete_matrix('/tmp/signed-residuals')
  Residuals.save('/tmp/signed-residuals', dtype=np.float32)
  Residuals = ShardedLoader('/tmp/signed-residuals')

  if X is not None:
    r = Residuals.load_slice(0, 1_000).squeeze()
    I = np.argsort(-np.abs(r))[:10]
    r = r[I]
    x = X.load_slice(0, 1_000).squeeze()[I]
    y = Y.load_slice(0, 1_000).squeeze()[I]
    turn = T.load_slice(0, 1_000).squeeze()[I]
    yhat = Yhat.load_slice(0, 1_000).squeeze()[I]

    """
    Issues:

    1. Positions where threatened pieces can be moved away are included in the dataset (maybe okay?)
    2. Threatened royal forks are included in the dataset (solution: exclude positions where checks are the best move)
    3. Very exposed king (r1q1b3/1p4k1/1p5p/2pn4/3p3P/P2P2B1/1P2QRPK/8 w - - 1 1)
    4. Promoting with check is good! (5rk1/6p1/p7/8/3p1BQ1/1P1P2P1/p7/5K2 b - - 1 1)
    5. 7 promoting pawns + exposed king vs QR -- hard position! 4Q2R/8/2k1p1K1/8/8/8/pp1p4/8 b - - 1 1
    6. I think this (1R6/P3qk2/6p1/2p5/P7/2PP4/6K1/8 w - - 1 1) and 5 (above) show that the engine
       doesn't appreciate that a Q is unlikely to mate by itself, so promoting (and protected) passed
       pawns are great, even if the king is exposed. OTOH Q+R can often mate, even when there are
       protected, promoting pawns. Solution: king safety penalty that includes number of involved
       opponent pieces?
    """
    import chess
    for i in range(10):
      fen = table2fen(x[i,:12*8*8].reshape(12, 8, 8), white_to_move=turn[i] > 0)
      print(fen.rjust(70), f'{y[i]:.3f}'.rjust(7), f'{yhat[i]:.3f}'.rjust(7), f'{r[i]:.3f}'.rjust(7))
      print(chess.Board(fen))
      print('')

  UnsignedResiduals = RowMapper(times, Residuals, T)

  if True:
    e_cov = compute_inner_product(MonoTable, MonoTable, weights_loader=RowMapper(times, earliness, W) if W is not None else earliness, num_workers=num_workers)
    np.save(f'data/{a}/derived/e_cov.npy', e_cov)
    l_cov = compute_inner_product(MonoTable, MonoTable, weights_loader=RowMapper(times, lateness, W) if W is not None else lateness, num_workers=num_workers)
    np.save(f'data/{a}/derived/l_cov.npy', l_cov)
    # e_cov = np.load(f'data/{a}/derived/e_cov.npy')
    # l_cov = np.load(f'data/{a}/derived/l_cov.npy')
  else:
    e_cov = np.eye(MonoTable.shape[0], dtype=np.float32)
    l_cov = np.eye(MonoTable.shape[0], dtype=np.float32)
  
  if True:
    e_dot = compute_inner_product(MonoTable, UnsignedResiduals, weights_loader=RowMapper(times, earliness, W) if W is not None else earliness, num_workers=num_workers)
    np.save(f'data/{a}/derived/e_dot.npy', e_dot)
    l_dot = compute_inner_product(MonoTable, UnsignedResiduals, weights_loader=RowMapper(times, lateness, W) if W is not None else lateness, num_workers=num_workers)
    np.save(f'data/{a}/derived/l_dot.npy', l_dot)
    # e_dot = np.load(f'data/{a}/derived/e_dot.npy')
    # l_dot = np.load(f'data/{a}/derived/l_dot.npy')
  else:
    e_dot = np.zeros(MonoTable.shape[0], dtype=np.float32)
    l_dot = np.zeros(MonoTable.shape[0], dtype=np.float32)
  
  if True:
    early = np.linalg.solve(e_cov + np.eye(e_cov.shape[0]) * 10000, e_dot)
    late = np.linalg.solve(l_cov + np.eye(l_cov.shape[0]) * 10000, l_dot)
  else:
    early = np.zeros(6 * 8 * 8, dtype=np.float32)
    late = np.zeros(6 * 8 * 8, dtype=np.float32)
  
  early = early.reshape((8, 8, 6))
  late = late.reshape((8, 8, 6))

  scale = 300

  text = ""
  for k, w in zip(['early', 'late', 'ineq'], [wEarly, wLate, inEq]):
    text += ('%i' % round(0)).rjust(7) + f'  // {k} bias\n'
    for i, varname in enumerate(varnames):
      text += ('%i' % round(float(w[i]) * scale)).rjust(7) + f'  // {k} {varname}\n'

  for w in [early, late]:
    text += "// ?\n"
    for _ in range(8):
      text += '0'.rjust(6) * 8 + '\n'
    for i, p in enumerate('PNBRQK'):
      text += "// %s\n" % p
      for r in range(8):
        for f in range(8):
          text += ('%i' % round(w[r,f,i] * scale)).rjust(6)
        text += '\n'

  with open('w2.txt', 'w') as f:
    f.write(text)

# Piece Square tables:
# python3 selfplay.py ./uci w-1.txt ./uci w-25600.txt
# 0.0327 ± 0.0125
# python3 selfplay.py ./uci w-1e6.txt ./uci w-0.01.txt
# -0.0067 ± 0.0141
# Conclusion: 1 is reasonable

# Features:
# python3 selfplay.py ./uci w-1e6.txt ./uci w-1.txt
# -0.350 ± 0.032
# python3 selfplay.py ./uci w-1e3.txt ./uci w-1.txt
# 0.0054 ± 0.0131