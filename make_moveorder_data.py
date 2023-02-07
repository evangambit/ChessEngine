import time
import argparse
import os
import random
import subprocess
import sqlite3

import chess
from chess import pgn

from stockfish import Stockfish
from stockfish.models import StockfishException

import numpy as np

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
  command = ["./a.out", "fen", *fen.split(' ')]
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

assert args.mode in ['generate_positions', 'write_numpy']

conn = sqlite3.connect("db.sqlite3")
c = conn.cursor()
kTableName = "MoveOrder"

if args.mode == 'generate_positions':
  c.execute(f"""CREATE TABLE IF NOT EXISTS {kTableName} (
    isBest INTEGER,
    numMoves INTEGER,
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

    if len(moves) < 1:
      continue

    try:
      stockfish.set_fen_position(fen)
      bestmove = stockfish.get_best_move()
    except StockfishException:
      print('reboot')
      # Restart stockfish.
      fens.remove(fen)
      stockfish = Stockfish(path=args.stockpath, depth=args.depth)
      continue

    if bestmove is None:
      # Checkmate
      continue

    assert bestmove in [m[0] for m in moves]

    for move, features in moves:
      c.execute(f"""INSERT INTO {kTableName}
        (isBest, numMoves, features) 
        VALUES (?, ?, ?)""", (
        move == bestmove,
        len(moves),
        ' '.join(str(a) for a in features),
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
  c.execute(f"SELECT isBest, numMoves, features FROM {kTableName}")
  X, Y, N = [], [], []
  for isBest, numMoves, features in c:
    x = np.array([float(a) for a in features.split(' ')], dtype=np.int32)
    X.append(x)
    Y.append(isBest)
    N.append(numMoves)

  X = np.array(X)
  Y = np.array(Y)
  N = np.array(N)

  depth = X[:,-1]
  X = X[:,:-1].copy()

  print(X.shape, Y.shape, N.shape)
  np.save(os.path.join('traindata', 'x.moveorder.npy'), X)
  np.save(os.path.join('traindata', 'y.moveorder.npy'), Y)
  np.save(os.path.join('traindata', 'n.moveorder.npy'), Y)

  w = np.linalg.lstsq(X * N.reshape(-1, 1), Y * N, rcond=1e-6)[0]

  from sklearn.linear_model import LogisticRegression

  regr = LogisticRegression().fit(X, Y)

  print(np.round(regr.coef_ * 500).reshape(-1, 6))


  exit(0)

