"""
Module to load, write, and perform basic linear operations on sharded matrices.

Useful for matrices that are too large to fit into memory, or when you want to
do computations using multiple processes.

Source: https://github.com/evangambit/sharded-matrix
"""

import multiprocessing
import os
from functools import lru_cache

import numpy as np

"""
General matrix format is

type: <4 bytes>  // type ("bool", "i8  ", "u16 ", etc.)
    <4 bytes>  // number of dimensions
    <4 bytes>  // size of dimension 1  (number of rows)
    <4 bytes>  // size of dimension 2
    ...
    [DATA]
    ...
"""

kSupportedTypes = set([np.float32, np.int16, np.int8, bool])
kType2ByteEncoding = {
  np.float32: b'f32 ',
  np.int16:   b'i16 ',
  np.int8:  b'i8  ',
  bool:     b'bool',
}
kByteEncodingToType = {
  b'f32 ': np.float32,
  b'i16 ': np.int16,
  b'i8  ': np.int8,
  b'bool': bool,
}
kType2SizeBits = {
  np.float32: 32,
  np.int16:   16,
  np.int8:  8,
  bool:     1,
}

def path2shardname(path, i):
  return f'{path}-{str(i).rjust(5, "0")}.sm'

def product(x):
  return int(np.prod(x))

def tensor2bytes(A):
  if A.dtype == bool:
    return np.packbits(A, bitorder='little').tobytes()
  else:
    return A.tobytes()

def bytes2tensor(A, dtype, shape):
  if dtype == bool:
    return np.unpackbits(np.frombuffer(A, dtype=np.uint8), bitorder='little').reshape(shape)
  else:
    return np.frombuffer(A, dtype=dtype).reshape(shape)

class ShardedWriter:
  def __init__(self, path: str, dtype: type, shape: tuple[int], shard_size_bytes: int = 25_000_000) -> None:
    assert dtype in kSupportedTypes
    # assert len(shape) > 0, 'At least one dimension must be provided'
    for d in shape:
      assert d > 0, 'All dimensions must be greater than 0'
    if dtype == bool:
      assert product(shape) % 8 == 0, 'Number of boolean elements per row must be a multiple of 8'
    self.path = path
    self.dtype = dtype
    self.shape = shape
    self.shard_index = 0
    self.bytes_per_row = product(self.shape) * kType2SizeBits[self.dtype] // 8
    self.rows_per_shard = shard_size_bytes // self.bytes_per_row
    self.bytes_per_shard = self.rows_per_shard * self.bytes_per_row
    assert self.rows_per_shard >= 1, 'Shape is too big :('
    self._i = 0
    self._data_to_write = bytearray(self.rows_per_shard * self.bytes_per_row)

  def write(self, x: np.ndarray):
    self.write_many(x.reshape((1,) + x.shape))
  
  def __enter__(self):
    return self
  
  def __exit__(self, type, value, tb):
    self.close()

  def close(self):
    if self._i == 0:
      return
    self._dump_data(self._i // self.bytes_per_row)

  def write_many(self, x: np.ndarray):
    if x.shape[0] == 0:
      return
    assert x.shape[1:] == self.shape, f'Expected shape {self.shape}, got {x.shape[1:]}'
    x = x.astype(self.dtype)
    delta_bytes = x.shape[0] * self.bytes_per_row

    while self._i + delta_bytes > self.bytes_per_shard:
      bytes_left = self.bytes_per_shard - self._i
      rows_left = bytes_left // self.bytes_per_row
      assert rows_left * self.bytes_per_row == bytes_left
      self._data_to_write[self._i:] = tensor2bytes(x[:bytes_left])
      self._dump_data(self.rows_per_shard)
      x = x[rows_left:]
      delta_bytes = x.shape[0] * self.bytes_per_row

    if self._i + delta_bytes <= self.bytes_per_shard:
      self._data_to_write[self._i:self._i+delta_bytes] = tensor2bytes(x)
      self._i += delta_bytes
      return
  
  def _dump_data(self, num_rows):
    with open(path2shardname(self.path, self.shard_index), 'wb') as f:
      f.write(kType2ByteEncoding[self.dtype])
      f.write(np.array(len(self.shape) + 1, dtype=np.int32).tobytes())
      for d in (num_rows,) + self.shape:
        f.write(np.array(d, dtype=np.int32).tobytes())
      f.write(self._data_to_write[:num_rows*self.bytes_per_row])
    self._i = 0
    self.shard_index += 1

def _load_shard_header(f):
  dtype = f.read(4)
  assert dtype in kByteEncodingToType, f'Unsupported type {dtype}'
  dtype = kByteEncodingToType[dtype]
  ndim = np.frombuffer(f.read(4), dtype=np.int32)[0]
  assert ndim >= 0, f'Invalid number of dimensions {ndim}'
  if ndim > 0:
    shape = np.frombuffer(f.read(4 * ndim), dtype=np.int32)
  else:
    shape = ()
  return dtype, shape

def load_shard(path):
  with open(path, 'rb') as f:
    dtype, shape = _load_shard_header(f)
    if dtype != bool:
      data = bytes2tensor(f.read(), dtype, shape)
    else:
      data = np.unpackbits(np.frombuffer(f.read(), dtype=np.uint8), bitorder='little')
    return data.reshape(shape)

def reshape(a, shape):
  return a.reshape(shape)

class LoaderInterface:
  def __init__(self) -> None:
    self.dtype = None
    self.shape = None
    self.num_shards = None
    self.num_rows = None

  def load_shard(self, shard_index) -> np.ndarray:
    raise NotImplementedError()

  def shard_to_slice_indices(self, shard_index) -> tuple[int,int]:
    raise NotImplementedError()

  def load_slice(self, start, end) -> np.ndarray:
    raise NotImplementedError()

  def reshape(self, shape):
    assert product(shape) == product(self.shape)
    return RowMapper(curry(reshape, (-1,) + shape), self)
  
  def num_rows_in_shards(self):
    result = []
    for shard in range(self.num_shards):
      start, end = self.shard_to_slice_indices(shard)
      result.append(end - start)
    return tuple(result)
  
  def save(self, path, force=None, dtype=None):
    if dtype == None:
      dtype = self.dtype
    if not force:
      assert not os.path.exists(path2shardname(path, 0))
    else:
      i = 0
      while os.path.exists(path2shardname(path, i)):
        os.remove(path2shardname(path, i))
        i += 1
    
    with ShardedWriter(path, shape=self.shape, dtype=dtype) as w:
      for shard in range(self.num_shards):
        w.write_many(self.load_shard(shard))
    

class ShardedLoader(LoaderInterface):
  def __init__(self, path: str):
    self._path = path
    self.num_shards = 0

    self.num_rows = 0
    num_rows_in_shards = []
    while os.path.exists(path2shardname(self._path, self.num_shards)):
      with open(path2shardname(self._path, self.num_shards), 'rb') as f:
        dtype, shape = _load_shard_header(f)
        num_rows_in_shards.append(shape[0])
        self.num_rows += shape[0]
      self.num_shards += 1
    assert self.num_shards > 0

    self._cumsum_rows = np.cumsum(num_rows_in_shards)

    self.dtype = dtype
    self.shape = tuple(shape[1:])
    self.rows_per_shard = shape[0]
  
  @lru_cache(maxsize=1)
  def load_shard(self, shard_index):
    return load_shard(path2shardname(self._path, shard_index))
  
  def shard_to_slice_indices(self, shard_index):
    start = self._cumsum_rows[shard_index - 1] if shard_index > 0 else 0
    end = self._cumsum_rows[shard_index]
    return start, end
  
  def load_slice(self, start, end):
    assert end > start, 'End index must be greater than start index'
    assert start < self._cumsum_rows[-1], 'Start index is out of bounds'
    i = (start < self._cumsum_rows).argmax()
    j = (end <= self._cumsum_rows).argmax()
    R = []
    for shard_index in range(i, j + 1):
      shard = self.load_shard(shard_index)

      if shard_index == 0:
        start_offset = start
        end_offset = end
      else:
        start_offset = start - self._cumsum_rows[shard_index - 1]
        end_offset = end - self._cumsum_rows[shard_index - 1]
      
      start_offset = max(0, start_offset)
      end_offset = min(shard.shape[0], end_offset)
      
      R.append(shard[start_offset:end_offset])
    
    return np.concatenate(R, 0)

  def iter(self, offset=0):
    for shard in range(self.num_shards):
      a, b = self.shard_to_slice_indices(shard)
      if offset >= a and offset < b:
        break
    while shard < self.num_shards:
      for row in self.load_shard(shard):
        yield offset, row
        offset += 1
      shard += 1
  
class MappingLoader(LoaderInterface):
  def __init__(self, loader: LoaderInterface, *mappings, width=None):
    super().__init__()

    self._loader = loader
    self._mappings = mappings

    self.dtype = loader.dtype
    self.shape = tuple(self._apply(np.ones((1,) + loader.shape) * 0.5).shape[1:])
    self.num_shards = loader.num_shards
    self.num_rows = loader.num_rows
  
  @lru_cache(maxsize=1)
  def load_shard(self, shard_index):
    return self._apply(self._loader.load_shard(shard_index))

  def shard_to_slice_indices(self, shard_index):
    return self._loader.shard_to_slice_indices(shard_index)

  def load_slice(self, start, end):
    return self._apply(self._loader.load_slice(start, end))
  
  def _apply(self, x):
    for f in self._mappings:
      x = f(x)
    return x

class Slice(LoaderInterface):
  def __init__(self, tensor: LoaderInterface, start: int, end: int):
    self._tensor = tensor
    self._start = start
    self._end = end

    assert start >= 0, 'Start index must be non-negative'
    assert end > start, 'End index must be greater than start index'
    assert end <= tensor.num_rows, 'End index is out of bounds'

    self._shard_start = 0
    while tensor.shard_to_slice_indices(self._shard_start)[0] < start:
      self._shard_start += 1

    self._shard_end = 0
    while tensor.shard_to_slice_indices(self._shard_end)[1] < end:
      self._shard_end += 1
    self._shard_end += 1
    
    self.dtype = tensor.dtype
    self.shape = tensor.shape
    self.num_shards = self._shard_end - self._shard_start
    self.num_rows = end - start
  
  def load_shard(self, shard_index):
    assert shard_index >= 0
    assert shard_index < self.num_shards
    return self._tensor.load_slice(*self.shard_to_slice_indices(shard_index))

  def shard_to_slice_indices(self, shard_index):
    i, j = self._tensor.shard_to_slice_indices(shard_index - self._shard_start)
    if i < self._start:
      i = self._start
    if j > self._end:
      j = self._end
    return i - self._start, j - self._start

  def load_slice(self, start, end):
    assert start >= 0, 'Start index must be non-negative'
    assert end > start, 'End index must be greater than start index'
    assert end <= self.num_rows, f'End index is out of bounds ({end} <= {self.num_rows})'
    return self._tensor.load_slice(start + self._start, end + self._start)

class RowMapper(LoaderInterface):
  def __init__(self, f, *loaders):
    super().__init__()
    self._loaders = loaders
    self._f = f

    result = self._f(*[loader.load_slice(0, 1) for loader in self._loaders])

    self.dtype = result.dtype
    self.shape = tuple(result.shape[1:])
    self.num_shards = loaders[0].num_shards
    self.num_rows = loaders[0].num_rows
    for loader in loaders[1:]:
      assert loader.num_rows == self.num_rows, 'All loaders must have the same number of rows'
  
  def load_shard(self, shard_index):
    indices = self._loaders[0].shard_to_slice_indices(shard_index)
    A = [
      self._loaders[0].load_shard(shard_index)
    ]
    for loader in self._loaders[1:]:
      A.append(loader.load_slice(*indices))
    return self._f(*A)
  
  def shard_to_slice_indices(self, shard_index):
    return self._loaders[0].shard_to_slice_indices(shard_index)
  
  def load_slice(self, start, end):
    A = [l.load_slice(start, end) for l in self._loaders]
    return self._f(*A)

def _compute_innerproduct(loader1, loader2, offset):
  print('_compute_innerproduct', offset)
  shard = loader1.load_shard(offset).astype(np.float32)
  i = loader1.shard_to_slice_indices(offset)
  slice = loader2.load_slice(*i).astype(np.float32)
  return shard.T @ slice

def _compute_weighted_innerproduct(loader1, loader2, weights_loader, offset):
  print('_compute_weighted_innerproduct', offset)
  shard = loader1.load_shard(offset).astype(np.float32)
  indices = loader1.shard_to_slice_indices(offset)
  slice = loader2.load_slice(*indices).astype(np.float32)
  weights = weights_loader.load_slice(*indices).astype(np.float32)

  shard = shard * weights
  slice = slice * weights
  return shard.T @ slice

def _compute_self_innerproduct(loader1, offset):
  print('_compute_self_innerproduct', offset)
  A = loader1.load_shard(offset).astype(np.float32)
  return A.T @ A

def _compute_weighted_self_innerproduct(loader1, weights_loader, offset):
  print('_compute_weighted_self_innerproduct', offset)
  A = loader1.load_shard(offset).astype(np.float32)
  weights = weights_loader.load_slice(*loader1.shard_to_slice_indices(offset)).astype(np.float32)
  A = A * weights
  return A.T @ A

class curry:
  def __init__(self, f, *args, **kwargs):
    self._f = f
    self._args = args
    self._kwargs = kwargs
  def __call__(self, *args, **kwargs):
    return self._f(*(self._args + args), **(self._kwargs | kwargs))
  
class compose:
  def __init__(self, *funcs):
    self._funcs = funcs
  def __call__(self, x):
    for f in self._funcs:
      x = f(x)
    return x
    

def _matmul(a, b):
  return a @ b

def matmul(loader: LoaderInterface, matrix: np.ndarray, num_workers: int = 4):
  _ = loader.load_slice(0, 1) @ matrix  # Test the operation is valid
  shards = list(range(0, loader.num_shards))
  return MappingLoader(loader, curry(_matmul, b=matrix))

def compute_inner_product(loader1: LoaderInterface, loader2: LoaderInterface, weights_loader=None, num_workers: int = 4):
  assert loader1.num_rows == loader2.num_rows, 'Both loaders must have the same number of shards'

  # Make loader1 the bigger loader. We'll be loading loader1 chunk-by-chunk and loader2 by slicing.
  should_transpose = product(loader1.shape) < product(loader2.shape)
  if should_transpose:
    loader1, loader2 = loader2, loader1

  shards = list(range(0, loader1.num_shards))
  result = None
  with multiprocessing.Pool(num_workers) as pool:
    if loader1 is loader2:
      if weights_loader is None:
        inner_products = pool.starmap(_compute_self_innerproduct, [(loader1, offset) for offset in shards])
      else:
        inner_products = pool.starmap(_compute_weighted_self_innerproduct, [(loader1, weights_loader, offset) for offset in shards])
    else:
      if weights_loader is None:
        inner_products = pool.starmap(_compute_innerproduct, [(loader1, loader2, offset) for offset in shards])
      else:
        inner_products = pool.starmap(_compute_weighted_innerproduct, [(loader1, loader2, weights_loader, offset) for offset in shards])
    result = sum(inner_products)

    if should_transpose:
      result = result.T
    return result

def linear_regression(X: LoaderInterface, Y: LoaderInterface, weights=None, regularization: float = 0.0, num_workers: int = 4):
  assert len(X.shape) == 1
  assert len(Y.shape) == 1
  assert X.num_rows == Y.num_rows
  assert num_workers > 0
  cov = compute_inner_product(X, X, num_workers=num_workers)
  if regularization > 0.0:
    cov += np.eye(cov.shape[0]) * regularization
  dot_product = compute_inner_product(X, Y, weights_loader=weights, num_workers=num_workers)
  return np.linalg.solve(cov, dot_product)

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

def table2fen(A, white_to_move):
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


if __name__ == '__main__':
  a = 'de5-md2'
  X = ShardedLoader(f'data/{a}/data-table')
  F = ShardedLoader(f'data/{a}/data-features')
  Y = ShardedLoader(f'data/{a}/data-eval')
  T = ShardedLoader(f'data/{a}/data-turn')

  print(X.num_shards)

  n = 5_000_000
  X = Slice(X, 0, n)
  F = Slice(F, 0, n)
  Y = Slice(Y, 0, n)
  T = Slice(T, 0, n)

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

  # Derived tables.
  earliness = MappingLoader(PC, compute_earliness)
  lateness = MappingLoader(PC, compute_earliness, compute_lateness)
  SignedY = RowMapper(to_signed_y, Y, T)
  features = RowMapper(early_late, F, lateness)  # cat([F * early, F * late], 1)


  w = linear_regression(features, SignedY, regularization=50.0)
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
  early = linear_regression(MonoTable, UnsignedResiduals, weights=earliness, regularization=50.0).reshape((6, 8, 8))
  late = linear_regression(MonoTable, UnsignedResiduals, weights=lateness, regularization=50.0).reshape((6, 8, 8))
  
  
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
