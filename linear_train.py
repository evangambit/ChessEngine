import os

import numpy as np

from sharded_matrix import ShardedLoader, ShardedWriter, linear_regression, MappingLoader, Slice, RowMapper, matmul, curry, compute_inner_product
from utils import ExpandedLinear, ShardedMatrixDataset, Weights, varnames, table2fen

from functools import partial

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

def compute_piece_counts(x):
  x = x[:,:-8].reshape((x.shape[0], 12, 8, 8)).sum((2, 3))
  return np.concatenate([x[:,:6], x[:,7:11]], 1)

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
  return (x[:,1:2] + x[:,2:3] + x[:,3:4] + x[:,4:5] * 3 + x[:,6:7] + x[:,7:8] + x[:,8:9] + x[:,9:10] * 3).clip(0, 18) / 18

def compute_material_inequality(x):
  a =  (x[:, varnames.index('OUR_KNIGHTS')] - x[:, varnames.index('THEIR_KNIGHTS')]) * 3
  a += (x[:, varnames.index('OUR_BISHOPS')] - x[:, varnames.index('THEIR_BISHOPS')]) * 3
  a += (x[:, varnames.index('OUR_ROOKS')] - x[:, varnames.index('THEIR_ROOKS')]) * 5
  a += (x[:, varnames.index('OUR_QUEENS')] - x[:, varnames.index('THEIR_QUEENS')]) * 9
  return a.clip(-1, 1).reshape((-1, 1))

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

if __name__ == '__main__':
  a = 'de6-md2'
  X = ShardedLoader(f'data/{a}/data-table')
  F = ShardedLoader(f'data/{a}/data-features')
  Y = ShardedLoader(f'data/{a}/data-eval')
  T = ShardedLoader(f'data/{a}/data-turn')

  Y = MappingLoader(Y, logit)

  print(X.num_shards, X.num_rows / 1_000_000)

  weights = Weights(open('weights.txt', 'r'))

  # n = 50_000
  # X = Slice(X, 0, n)
  # F = Slice(F, 0, n)
  # Y = Slice(Y, 0, n)
  # T = Slice(T, 0, n)


  if True:
    if not os.path.exists(f'data/{a}/derived/'):
      os.mkdir(f'data/{a}/derived/')
    os.system(f'rm -f data/{a}/derived/*')
    print('Computing piece counts...')
    PC = MappingLoader(X, compute_piece_counts)
    with ShardedWriter(f'data/{a}/derived/data-piece-counts', shape=(10,), dtype=np.int8) as w:
      for shard in range(PC.num_shards):
        w.write_many(PC.load_shard(shard))

    print('Computing mono table')
    MonoTable = MappingLoader(X, table2monocolor)
    MonoTable.save(f'data/{a}/derived/data-mono-table', force=True)
    print('Done')

  PC = ShardedLoader(f'data/{a}/derived/data-piece-counts')
  MonoTable = ShardedLoader(f'data/{a}/derived/data-mono-table')

  # Derived tables.
  earliness = MappingLoader(PC, compute_earliness)
  lateness = MappingLoader(earliness, compute_lateness)
  inequality_ness = MappingLoader(F, compute_material_inequality)
  SignedY = RowMapper(to_signed_y, Y, T)  # From mover's perspective
  features = RowMapper(row_merger, F, lateness, inequality_ness)

  # TODO: add back in a bias term and see if that helps.
  avg = SignedY.load_slice(0, SignedY.num_rows).mean()
  SignedY = RowMapper(partial(minus, b=avg), SignedY)

  foo = 0
  if foo == 0:
    w, cov, dot = linear_regression(features, SignedY, regularization=200.0)
    np.save(f'data/{a}/derived/cov.npy', cov)
    np.save(f'data/{a}/derived/dot.npy', dot)
  elif foo == 1:
    cov = np.load(f'data/{a}/derived/cov.npy')
    dot = np.load(f'data/{a}/derived/dot.npy')
    w = np.linalg.solve(cov + np.diag(np.ones(cov.shape[0]) * 1e3), dot)
  else:
    w = np.zeros((features.shape[0], 1), dtype=np.float32)
  wEarly, wLate, inEq = w.reshape(3, -1)

  Yhat = matmul(features, w)
  Yhat.save('/tmp/yhat', dtype=np.float32, force=True)
  Yhat = ShardedLoader('/tmp/yhat')

  Residuals = RowMapper(minus, SignedY, Yhat)
  UnsignedResiduals = RowMapper(times, Residuals, T)

  foo = 0
  if foo == 0:
    early, e_cov, e_dot = linear_regression(MonoTable, UnsignedResiduals, weights=earliness, regularization=100.0)
    early = early.reshape((6, 8, 8))
    np.save(f'data/{a}/derived/e_cov.npy', e_cov)
    np.save(f'data/{a}/derived/e_dot.npy', e_dot)
    late, l_cov, l_dot = linear_regression(MonoTable, UnsignedResiduals, weights=lateness, regularization=100.0)
    late = late.reshape((6, 8, 8))
    np.save(f'data/{a}/derived/l_cov.npy', l_cov)
    np.save(f'data/{a}/derived/l_dot.npy', l_dot)
  elif foo == 1:
    e_cov = np.load(f'data/{a}/derived/e_cov.npy')
    e_dot = compute_inner_product(MonoTable, UnsignedResiduals, weights_loader=earliness, num_workers=4)
    early = np.linalg.solve(e_cov + np.diag(np.ones(e_cov.shape[0]) * 1), e_dot).reshape((6, 8, 8))
    l_cov = np.load(f'data/{a}/derived/l_cov.npy')
    l_dot = compute_inner_product(MonoTable, UnsignedResiduals, weights_loader=lateness, num_workers=4)
    late = np.linalg.solve(l_cov + np.diag(np.ones(l_cov.shape[0]) * 1), l_dot).reshape((6, 8, 8))
  else:
    early = np.zeros((6, 8, 8))
    late = np.zeros((6, 8, 8))

  # If we never learned wEarly, use piece maps to determine centipawn value.
  if wEarly[:3].sum() == 0.0:
    scale = early[0,2:-1].mean()
  else:
    scale = 700 / wEarly[:3].mean()

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
          text += ('%i' % round(w[i,r,f] * scale)).rjust(6)
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