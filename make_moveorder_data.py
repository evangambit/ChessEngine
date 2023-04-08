import time
import argparse
import json
import os
import random
import subprocess
import sqlite3

from collections import defaultdict

import chess
from chess import pgn

from stockfish import Stockfish
from stockfish.models import StockfishException

import numpy as np

def lpad(t, n, c=' '):
  t = str(t)
  return max(n - len(t), 0) * c + t

def pgn_iterator():
  assert len(args.pgnfiles) > 0
  for filename in args.pgnfiles:
    f = open(filename, 'r')
    game = pgn.read_game(f)
    while game is not None:
      for node in game.mainline():
        if random.randint(1, 50) != 1:
          continue
        board = node.board()
        if len(list(board.legal_moves)) == 0:
          continue
        yield board.fen()
      game = pgn.read_game(f)

def get_vec(fen):
  command = ["./a.out", "fen", *fen.split(' '), "makequiet", "0"]
  lines = subprocess.check_output(command).decode().strip().split('\n')
  if '</movedata>' not in lines:
    return None, None
  i = lines.index('<movedata>')
  lines = lines[i + 1 : -1]
  fen = lines[0]
  r = []
  for line in lines[1:]:
    parts = line.split(' ')
    move, features = parts[0], parts[1:]
    features = np.array([int(x) for x in features], dtype=np.int8)
    r.append((move, features))
  return fen, r

parser = argparse.ArgumentParser()
parser.add_argument("pgnfiles", nargs='*')
parser.add_argument("--mode", type=str, required=True)
parser.add_argument("--stockpath", default="/usr/local/bin/stockfish", type=str)
parser.add_argument("--depth", type=int, default=10)
args = parser.parse_args()

assert args.mode in ['generate', 'write_numpy']

conn = sqlite3.connect("db.sqlite3")
c = conn.cursor()
kTableName = "MoveOrder"

if args.mode == 'generate':
  c.execute(f"""CREATE TABLE IF NOT EXISTS {kTableName} (
    features BLOB
  );""")

  stockfish = Stockfish(path=args.stockpath, depth=args.depth)

  totalWrites = 0
  writesSinceCommit = 0

  tstart = time.time()
  for fen in pgn_iterator():
    fen, moves = get_vec(fen)
    if fen is None:
      continue

    if len(moves) <= 1:
      continue

    try:
      stockfish.set_fen_position(fen)
      topmoves = stockfish.get_top_moves(10)
    except StockfishException:
      print('reboot')
      # Restart stockfish.
      fens.remove(fen)
      stockfish = Stockfish(path=args.stockpath, depth=args.depth)
      continue

    if len(topmoves) <= 1:
      # Checkmate
      continue

    topuci = [m['Move'] for m in topmoves]

    moves = [m for m in moves if m[0] in topuci]

    x = np.array([m[1] for m in moves])
    y = np.array([topuci.index(m[0]) for m in moves])

    x = x[np.argsort(y)]

    for move, features in moves:
      c.execute(f"""INSERT INTO {kTableName}
        (features) 
        VALUES (?)""", (
        json.dumps(x.tolist()),
      ))
    writesSinceCommit += 1
    totalWrites += 1

    if writesSinceCommit >= 40:
      dt = time.time() - tstart
      print('commit %i; %.3f per sec' % (totalWrites, totalWrites / dt))
      writesSinceCommit = 0
      conn.commit()

  game = pgn.read_game(f)
  exit(0)

if args.mode == 'write_numpy':
  c.execute(f"SELECT features FROM {kTableName}")
  X, Y, M = [], [], []
  for features, in c:
    x = np.array(json.loads(features), dtype=np.int32)
    mask = np.ones(x.shape[0])
    if x.shape[0] < 10:
      n = 10 - x.shape[0]
      x = np.concatenate([x, np.zeros((n,) + x.shape[1:])], 0)
      mask = np.concatenate([mask, np.zeros((n,) + mask.shape[1:])], 0)
    X.append(x)
    M.append(mask)

  X = np.array(X)
  M = np.array(M)

  import torch
  from torch import nn, optim

  class NormalizedLinear(nn.Module):
    def __init__(self, a, b, bias=True):
      super().__init__()
      self.weight = nn.Parameter(torch.randn(a, b), requires_grad=True)
      self.bias = None
      if bias:
        self.bias = nn.Parameter(torch.zeros(1, b), requires_grad=True)

    def forward(self, x):
      # with torch.no_grad():
      #   self.weight /= torch.sqrt(self.weight**2).mean()
      r = x @ self.weight
      if self.bias is not None:
        r = r + self.bias
      return r

  X[:,:,-6] *= 0.0
  X[:,:,-9] *= 0.0
  X[:,:,-11] *= 0.0
  X[:,:,-13] *= 0.0

  # X, Y, M = X[:50], Y[:50], M[:50]

  X = torch.tensor(X, dtype=torch.float32)
  k = X.shape[1]
  Y = torch.tensor(Y, dtype=torch.float32)
  M = torch.tensor(M, dtype=torch.float32)
  # M = torch.tensor(M.reshape((M.shape[0], 1, k)) + M.reshape((M.shape[0], k, 1)) > 0, dtype=torch.float32)
  # tri = np.tri(k); tri = torch.tensor(tri.reshape((1,) + tri.shape), dtype=torch.float32)

  model = NormalizedLinear(X.shape[-1], 1, bias=False)
  opt = optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.001)

  def loglinspace(a, b, n):
    return np.exp(np.linspace(np.log(a), np.log(b), n))

  metrics = defaultdict(list)
  for lr in loglinspace(0.3, 0.0001, 300):
    for pg in opt.param_groups:
      pg['lr'] = lr

    S = model(X).squeeze() # (n, 10)
    # loss = nn.functional.leaky_relu((S - S[:,0:1]) * M).mean()

    S = (S.T - S.mean(1)).T
    P = nn.functional.softmax(S, 1)
    loss = -torch.log(torch.cumsum(P, 1)).mean()

    # S = model(X)
    # foo = S.reshape(S.shape[0], k, 1) - S.reshape(S.shape[0], 1, k)
    # foo = foo * M
    # loss = torch.relu(foo * tri + 0.5).mean()

    opt.zero_grad()
    loss.backward()
    opt.step()
    metrics['loss'].append(float(loss))
    # metrics['acc'].append(float(((foo < 0) * tri).mean() / tri.mean()))
    metrics['acc'].append(float((S.argmax(1) == 0).to(torch.float32).mean()))

  print(metrics['loss'][:3])
  print(metrics['loss'][-3:])

  print(metrics['acc'][:3])
  print(metrics['acc'][-3:])

  import matplotlib.pyplot as plt

  w = model.weight.squeeze().detach().numpy().copy()
  w *= (X.std((0,1)) != 0.0).detach().numpy()
  w = np.round(w * 100).astype(np.int64)
  w = w.reshape(-1, 6)
  for row in w:
    print(','.join([lpad(x, 5) for x in row.tolist()]))

"""
w(good) - w(bad) = 0.0001
"""



