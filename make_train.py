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

python3 make_train.py --mode generate_positions --type quiet lichess_db_standard_rated_2013-11.pgn
python3 make_train.py --mode generate_positions --type endgame

python3 make_train.py --mode write_numpy --type quiet
python3 make_train.py --mode write_numpy --type endgame

zip traindata.zip -r traindata


When your features have changed but you want to re-use your stockfish evaluations:

python3 -i make_train.py --mode update_features
python3 make_train.py --mode write_numpy
zip traindata.zip -r traindata
"""

def pgn_iterator(noise):
  assert len(args.pgnfiles) > 0
  for filename in args.pgnfiles:
    f = open(filename, 'r')
    game = pgn.read_game(f)
    while game is not None:
      for node in game.mainline():
        if random.randint(1, 50) != 1:
          continue
        board = node.board()
        for _ in range(noise):
          moves = list(board.legal_moves)
          if len(moves) == 0:
            break
          random.shuffle(moves)
          board.push(moves[0])
        if len(list(board.legal_moves)) == 0:
          continue
        yield board.fen()
      game = pgn.read_game(f)

def endgame_iterator():
  endgameTypes = [
    # Types of end games by remaining pieces.
    ((0,0,0,0, 0,1), (0,0,0,0, 0,1)),  # KvK
    ((0,1,0,0, 0,1), (0,0,0,0, 0,1)),  # KNvK
    ((0,1,0,0, 0,1), (0,1,0,0, 0,1)),  # NvN
    ((0,1,1,0, 0,1), (0,1,0,0, 0,1)),  # NBvN
    ((0,1,1,0, 0,1), (0,1,1,0, 0,1)),  # NBvNB
    ((0,0,0,1, 0,1), (0,1,0,0, 0,1)),  # RvN
    ((0,0,0,1, 0,1), (0,0,1,0, 0,1)),  # RvB
    ((0,0,0,1, 0,1), (0,0,0,1, 0,1)),  # RvR
    ((0,0,0,1, 0,1), (0,1,1,0, 0,1)),  # RvBN
  ]
  while True:
    kind = random.choice(endgameTypes)
    if random.randint(0, 1) == 0:
      white, black = kind
    else:
      black, white = kind
    white, black = list(white), list(black)

    # Randomize number of pawns
    white[0] = random.randint(1, 4)
    black[0] = random.randint(1, 4)

    board = chess.Board('8/8/8/8/8/8/8/8 w - - 0 1')

    for color, array in enumerate([white, black]):
      for i, n in enumerate(array):
        for _ in range(n):
          sq = chess.Square(random.randint(0, 63))
          piece = chess.Piece(i + 1, color)
          board.set_piece_at(sq, piece)

    fen = board.fen()
    if random.randint(0, 1) == 0:
      fen = fen.replace(' w ', ' b ')

    if not board.is_valid():
      continue

    if board.is_insufficient_material():
      continue

    if board.is_game_over():
      continue

    yield fen

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
parser.add_argument("--noise", type=int, default=0)
args = parser.parse_args()

assert args.type in ["quiet", "any", "tactics", "endgame"]
assert args.mode in ['generate_positions', 'update_features', 'write_numpy']
assert args.noise >= 0

if args.type == "endgame":
  assert len(args.pgnfiles) == 0
  assert args.noise == 0

conn = sqlite3.connect("db.sqlite3")
c = conn.cursor()
kTableName = {
  "quiet": "QuietTable",
  "any": "AnyTable",
  "tactics": "TacticsTable",
  "endgame": "EndgameTable",
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

  iterator = None
  if args.type == 'endgame':
    iterator = endgame_iterator()
  else:
    iterator = pgn_iterator(args.noise)

  totalWrites = 0
  writesSinceCommit = 0

  tstart = time.time()
  for fen in iterator:
    fen, x = get_vec(fen)

    if fen in fens:
      continue

    fens.add(fen)

    try:
      stockfish.set_fen_position(fen)
      moves = stockfish.get_top_moves(2)
    except StockfishException:
      print('reboot')
      # Restart stockfish.
      fens.remove(fen)
      stockfish = Stockfish(path="/usr/local/bin/stockfish", depth=kDepth)
      continue

    if len(moves) != 2:
      continue

    bestmove = moves[0]['Move']

    if ' b ' in fen:
      for move in moves:
        if move['Centipawn'] is not None:
          move['Centipawn'] *= -1
        if move['Mate'] is not None:
          move['Mate'] *= -1

    for move in moves:
      if move['Mate'] is not None:
        if move['Mate'] > 0:
          move['Centipawn'] = 500
        else:
          move['Centipawn'] = -500
      move['Centipawn'] = max(-500, min(500, move['Centipawn']))

    scoreDelta = abs(moves[0]['Centipawn'] - moves[1]['Centipawn'])
    evaluation = moves[0]['Centipawn']

    assert evaluation is not None
    assert scoreDelta is not None

    if args.type == 'tactics' and scoreDelta < 100:
      continue

    scoreDelta = min(500, scoreDelta)

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
  print(X.shape, Y.shape, F.shape)
  np.save(os.path.join('traindata', f'x.{args.type}.npy'), X)
  np.save(os.path.join('traindata', f'y.{args.type}.npy'), Y)
  np.save(os.path.join('traindata', f'f.{args.type}.npy'), F)
  exit(0)

