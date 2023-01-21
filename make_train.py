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

"""
To generate training data from scratch:

python3 make_train.py --mode generate_positions lichess_db_standard_rated_2013-11.pgn
python3 make_train.py --mode write_numpy
zip traindata.zip -r traindata


When your features have changed but you want to re-use your stockfish evaluations:

python3 -i make_train.py --mode update_features
python3 make_train.py --mode write_numpy
zip traindata.zip -r traindata
"""

def get_vec(fen):
  q = "1" if args.type == "tactics" else "0"
  command = ["./a.out", "mode", "printvec-cpu", "fen", *fen.split(' '), "makequiet", q]
  lines = subprocess.check_output(command).decode().strip().split('\n')
  assert len(lines) == 2, lines
  fen = lines[0]
  x = [int(val) for val in lines[1].split(' ')]
  return fen, x

def get_vecs(fens):
  q = "1" if args.type == "tactics" else "0"
  filename = '/tmp/fens.txt'
  with open(filename, 'w+') as f:
    f.write('\n'.join(fens))
  command = ["./a.out", "mode", "printvec-cpu", "fens", filename, "makequiet", q]
  lines = subprocess.check_output(command).decode().strip().split('\n')
  assert len(lines) == len(fens) * 2
  for i in range(0, len(lines), 2):
    x = [int(val) for val in lines[i + 1].split(' ')]
    yield lines[i + 0], x

parser = argparse.ArgumentParser()
parser.add_argument("pgnfiles", nargs='*')
parser.add_argument("--mode", type=str, required=True)
parser.add_argument("--type", type=str, required=True)
args = parser.parse_args()

assert args.type in ["quiet", "any", "tactics"]

assert args.mode in ['generate_positions', 'update_features', 'write_numpy']

conn = sqlite3.connect("db.sqlite3")
c = conn.cursor()
kTableName = {
  "quiet": "QuietTable",
  "any": "AnyTable",
  "tactics": "TacticsTable"
}[args.type]

if args.mode == 'update_features':
  c.execute(f"SELECT fen FROM {kTableName}")
  fens = [fen for fen, in c.fetchall()]
  writesSinceCommit = 0
  totalWrites = 0
  for i, (fen, x) in enumerate(get_vecs(fens)):
    if fen == fens[i]:
      c.execute(f"UPDATE {kTableName} SET moverFeatures = ? WHERE fen = ?", (
        ' '.join(str(a) for a in x),
        fen,
      ))
    else:
      c.execute(f"DELETE FROM {kTableName} WHERE fen = ?", (fens[i],))
      print('DELETE')
    writesSinceCommit += 1
    totalWrites += 1
    if writesSinceCommit >= 1000:
      conn.commit()
      print('commit', totalWrites, len(fens), len(x))
      writesSinceCommit = 0
  conn.commit()
  exit(0)

if args.mode == 'generate_positions':
  assert len(args.pgnfiles) > 0
  c.execute(f"""CREATE TABLE IF NOT EXISTS {kTableName} (
    fen BLOB,
    bestMove BLOB,
    delta INTEGER,
    moverScore INTEGER,
    moverFeatures BLOB    -- one byte per feature; from mover's perspective
  );""")
  c.execute(f"""CREATE INDEX IF NOT EXISTS fenIndex ON {kTableName}(fen)""")

  fens = set()
  c.execute(f"SELECT fen FROM {kTableName}")
  for fen, in c:
    fens.add(fen)

  kDepth = 12
  stockfish = Stockfish(path="/usr/local/bin/stockfish", depth=kDepth)

  totalWrites = 0
  writesSinceCommit = 0

  tstart = time.time()
  for filename in args.pgnfiles:
    f = open(filename, 'r')

    game = pgn.read_game(f)
    while game is not None:
      for node in game.mainline():
        if random.randint(1, 50) != 1:
          continue
        board = node.board()
        fen = board.fen()

        fen, x = get_vec(fen)

        if fen in fens:
          continue
        fens.add(fen)

        try:
          stockfish.set_fen_position(board.fen())
          moves = stockfish.get_top_moves(2)
          if len(moves) != 2:
            continue
          if moves[0]['Mate'] is not None:
            continue
          if moves[1]['Mate'] is not None:
            continue
          bestmove = moves[0]['Move']
          if abs(moves[0]['Centipawn']) > 300:
            # Ignore positions with crazy centipawn points.
            continue
          scoreDelta = abs(moves[0]['Centipawn'] - moves[1]['Centipawn'])
          if scoreDelta < 100 and args.type == 'tactics':
            continue
          if ' b ' in fen:
            evaluation = -moves[0]['Centipawn']
          else:
            evaluation = moves[0]['Centipawn']

        except StockfishException:
          fens.remove(fen)
          stockfish = Stockfish(path="/usr/local/bin/stockfish", depth=kDepth)
          continue

        c.execute(f"""INSERT INTO {kTableName}
          (fen, bestMove, delta, moverScore, moverFeatures) 
          VALUES (?, ?, ?, ?, ?)""", (
          fen,
          bestmove,
          scoreDelta,
          evaluation,
          ' '.join(str(a) for a in x),
        ))
        writesSinceCommit += 1
        totalWrites += 1

      if writesSinceCommit >= 40:
        dt = time.time() - tstart
        print('commit %i; %.3f per sec' % (totalWrites, totalWrites / dt), len(x))
        writesSinceCommit = 0
        conn.commit()

      game = pgn.read_game(f)
  exit(0)

if args.mode == 'write_numpy':
  c.execute(f"SELECT fen, moverScore, moverFeatures FROM {kTableName}")
  X, Y, F = [], [], []
  for fen, moverScore, moverFeatures in c:
    F.append(fen)
    Y.append(moverScore)

    x = np.array([float(a) for a in moverFeatures.split(' ')], dtype=np.int32)
    X.append(x)

  X = np.array(X)
  Y = np.array(Y)
  F = np.array(F)
  if args.type == "quiet":
    np.save(os.path.join('traindata', 'x.quiet.npy'), X)
    np.save(os.path.join('traindata', 'y.quiet.npy'), Y)
    np.save(os.path.join('traindata', 'f.quiet.npy'), F)
  elif args.type == "any":
    np.save(os.path.join('traindata', 'x.any.npy'), X)
    np.save(os.path.join('traindata', 'y.any.npy'), Y)
    np.save(os.path.join('traindata', 'f.any.npy'), F)
  elif args.type == "tactics":
    np.save(os.path.join('traindata', 'x.tactics.npy'), X)
    np.save(os.path.join('traindata', 'y.tactics.npy'), Y)
    np.save(os.path.join('traindata', 'f.tactics.npy'), F)
  exit(0)

