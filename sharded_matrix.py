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
    if dtype not in kSupportedTypes and hasattr(dtype, 'type'):
      dtype = dtype.type
    assert dtype in kSupportedTypes, f'Unsupported type {dtype}'
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

    result = self._apply(loader.load_slice(0, 1))
    self.dtype = result.dtype
    self.shape = tuple(result.shape[1:])
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
