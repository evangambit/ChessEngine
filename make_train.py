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
python3 make_train.py --mode generate_positions lichess_db_standard_rated_2013-11.pgn
python3 make_train.py --mode write_numpy
zip traindata.zip -r traindata
"""

def get_vec(fen):
  command = ["./a.out", "mode", "printvec", "fen", *fen.split(' ')]
  lines = subprocess.check_output(command).decode().strip().split('\n')
  assert lines[0].startswith('ORIGINAL_FEN')
  assert lines[1].startswith('FEN')
  assert lines[2].startswith('SCORE')
  originalFen = ' '.join(lines[1].split(' ')[1:])
  x = []
  for i, line in enumerate(lines[3:]):
    x.append(int(line.split(' ')[0]))
  return originalFen, x

parser = argparse.ArgumentParser()
parser.add_argument("pgnfiles", nargs='*')
parser.add_argument("--mode", type=str, required=True)
args = parser.parse_args()

assert args.mode in ['generate_positions', 'update_features', 'write_numpy']

conn = sqlite3.connect('train.sqlite3')
c = conn.cursor()
kTableName = 'TrainData'

if args.mode == 'update_features':
  c.execute(f"SELECT fen FROM {kTableName}")
  fens = c.fetchall()
  totalWrites = 0
  writesSinceCommit = 0
  for fen0, in fens:
    fen, x = get_vec(fen0)
    # assert fen == fen0, 'q search changed! Need to completely regenerate entire table :('
    if fen != fen0:
      c.execute(f"DELETE FROM {kTableName} WHERE fen = ?", (fen0,))
      print('DELETE')
      writesSinceCommit += 1
      totalWrites += 1
      continue
    c.execute(f"UPDATE {kTableName} SET moverFeatures = ? WHERE fen = ?", (
      ' '.join(str(a) for a in x),
      fen,
    ))
    writesSinceCommit += 1
    totalWrites += 1
    if writesSinceCommit >= 1000:
      writesSinceCommit = 0
      conn.commit()
      print('commit', totalWrites, len(fens), len(x))

  conn.commit()

  exit(0)

if args.mode == 'generate_positions':
  assert len(args.pgnfiles) > 0
  c.execute(f"""CREATE TABLE IF NOT EXISTS {kTableName} (
    fen BLOB,             -- a quiet position
    moverScore INTEGER,
    moverFeatures BLOB    -- one byte per feature; from mover's perspective
  );""")
  c.execute(f"""CREATE INDEX IF NOT EXISTS fenIndex ON {kTableName}(fen)""")

  fens = set()
  c.execute(f"SELECT fen FROM {kTableName}")
  for fen, in c:
    fens.add(fen)

  kDepth = 10
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

        # Makes fen quiet.
        fen, x = get_vec(fen)

        if fen in fens:
          continue
        fens.add(fen)

        try:
          stockfish.set_fen_position(fen)
          # stockfish.get_best_move_time
          evaluation = stockfish.get_evaluation()  # From white's perspective
        except StockfishException:
          fens.remove(fen)
          stockfish = Stockfish(path="/usr/local/bin/stockfish", depth=kDepth)
          continue
        if evaluation['type'] != 'cp':
          continue  # todo?
        if ' b ' in fen:
          evaluation['value'] *= -1

        c.execute(f"""INSERT INTO {kTableName}
          (fen, moverScore, moverFeatures) 
          VALUES (?, ?, ?)""", (
          fen,
          evaluation['value'],
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
  np.save(os.path.join('traindata', 'x.npy'), X)
  np.save(os.path.join('traindata', 'y.npy'), Y)
  np.save(os.path.join('traindata', 'f.npy'), F)
  exit(0)

