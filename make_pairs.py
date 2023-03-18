import time
import argparse
import os
import random
import subprocess
import sqlite3

from multiprocessing import Process, Queue
 
import chess
from chess import pgn

from simple_stockfish import Stockfish

import numpy as np

"""
To generate training data from scratch:

$ python3 -i make_pairs.py --mode generate --type any --quiet 1 --stockpath /opt/homebrew/bin/stockfish ~/Downloads/lichess_elite_2022-06.pgn

$ python3 -i make_pairs.py --mode write_numpy --type any --quiet 1 --stockpath /opt/homebrew/bin/stockfish ~/Downloads/lichess_elite_2022-06.pgn

$ python3 train_pairs.py

"""

def sql_inserter(resultQueue, args):
  conn = sqlite3.connect("db.sqlite3")
  c = conn.cursor()
  c.execute(f"""CREATE TABLE IF NOT EXISTS {get_table_name(args)} (
    fen1 BLOB,
    fen2 BLOB,
    moverScore1 INTEGER,
    moverScore2 INTEGER,
    moverFeatures1 BLOB,
    moverFeatures2 BLOB
  );""")

  fens = set()
  c.execute(f"SELECT fen1, fen2 FROM {get_table_name(args)}")
  for fenpair in c:
    fens.add(fenpair)

  t0 = time.time()
  tlast = time.time()
  numInserted = 0

  while True:
    result = resultQueue.get()

    fenpair = result["fen1"], result["fen2"]
    if fenpair in fens:
      continue
    fens.add(fenpair)

    c.execute(f"""INSERT INTO {get_table_name(args)}
      (fen1, fen2, moverScore1, moverScore2, moverFeatures1, moverFeatures2)
      VALUES (?, ?, ?, ?, ?, ?)""", (
      result["fen1"],
      result["fen2"],
      result["moverScore1"],
      result["moverScore2"],
      ' '.join(str(a) for a in result["moverFeatures1"]),
      ' '.join(str(a) for a in result["moverFeatures2"]),
    ))

    numInserted += 1
    if numInserted % 20 == 0:
      print('%.1f inserts/sec (%.1f avg)' % (
        20 / (time.time() - tlast),
        numInserted / (time.time() - t0)
      ), len(fens))
      tlast = time.time()
      conn.commit()

def make_random(fen, noise):
  board = chess.Board(fen)
  for _ in range(noise):
    moves = list(board.legal_moves)
    if len(moves) == 0:
      return None
    random.shuffle(moves)
    board.push(moves[0])
  return board.fen()

def analyzer(fenQueue, resultQueue, args):
    while True:
        fen = fenQueue.get()

        fen, features = get_vec(fen, args)
        if fen is None:
          continue

        fen1 = make_random(fen, random.randint(1, 4))
        fen2 = make_random(fen, random.randint(1, 4))

        if fen1 == fen2:
          continue

        if fen1 is None or fen2 is None:
          continue

        stockfish = Stockfish(path=args.stockpath)

        fen1, x1 = get_vec(fen1, args)
        fen2, x2 = get_vec(fen2, args)

        try:
          score1 = stockfish.analyze(fen1, depth=args.depth)['score']
        except RuntimeError as _:
          stockfish.p.terminate()
          continue
        if score1[0] == 'mate':
          score1 = score1[1] * args.clip
        else:
          score1 = score1[1]

        try:
          score2 = stockfish.analyze(fen2, depth=args.depth)['score']
        except RuntimeError as _:
          stockfish.p.terminate()
          continue
        if score2[0] == 'mate':
          score2 = score2[1] * args.clip
        else:
          score2 = score2[1]

        score1 = max(-args.clip, min(args.clip, score1))
        score2 = max(-args.clip, min(args.clip, score2))

        low = min(abs(score1), abs(score2))
        if abs(score1 - score2) / (low + 100) < 0.33:
          stockfish.p.terminate()
          continue

        resultQueue.put({
          "fen1": fen1,
          "fen2": fen2,
          "moverScore1": score1,
          "moverScore2": score2,
          "moverFeatures1": x1,
          "moverFeatures2": x2,
        })

        stockfish.p.terminate()

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
    ((0,1,0,0, 0,1), (0,0,0,0, 0,1)),  # Nv
    ((0,0,1,0, 0,1), (0,0,0,0, 0,1)),  # Bv
    ((0,0,0,1, 0,1), (0,0,0,0, 0,1)),  # Rv
    ((0,0,0,0, 1,1), (0,0,0,0, 0,1)),  # Qv
    ((0,1,0,0, 0,1), (0,1,0,0, 0,1)),  # NvN
    ((0,0,1,0, 0,1), (0,0,1,0, 0,1)),  # BvB
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
    white[0] = random.randint(0, 4)
    black[0] = random.randint(0, 4)

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

def get_vec(fen, args):
  command = ["./a.out", "mode", "printvec-cpu", "fen", *fen.split(' '), "makequiet", str(args.quiet)]
  lines = subprocess.check_output(command).decode().strip().split('\n')
  if lines[0].startswith('PRINT FEATURE VEC FAIL'):
    return None, None
  assert len(lines) == 2, lines
  fen = lines[0]
  x = [int(val) for val in lines[1].split(' ')]
  return fen, x

def get_vecs(fens, args):
  filename = '/tmp/fens.txt'
  with open(filename, 'w+') as f:
    f.write('\n'.join(fens))
  command = ["./a.out", "mode", "printvec-cpu", "fens", filename, "makequiet", str(args.quiet)]
  lines = subprocess.check_output(command).decode().strip().split('\n')
  lines = [line for line in lines if line != 'PRINT FEATURE VEC FAIL (MATE)']
  for i in range(0, len(lines), 2):
    x = [int(val) for val in lines[i + 1].split(' ')]
    yield lines[i + 0], x

def get_table_name(args):
  return f"{args.type}_d{args.depth}_q{args.quiet}_n{args.noise}"

if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument("pgnfiles", nargs='*')
  parser.add_argument("--mode", type=str, required=True, help="{generate, update_features, write_numpy}")
  parser.add_argument("--type", type=str, required=True, help="{any, endgame}")
  parser.add_argument("--quiet", type=int, required=True, help="{0, 1}")
  parser.add_argument("--noise", type=int, default=0)
  parser.add_argument("--depth", type=int, default=10, help="{1, 2, ..}")
  parser.add_argument("--num_workers", type=int, default=4)
  parser.add_argument("--clip", type=int, default=5000)
  parser.add_argument("--stockpath", default="/usr/local/bin/stockfish", type=str)
  args = parser.parse_args()

  assert args.type in ["any", "endgame"]
  assert args.quiet in [0, 1]
  assert args.mode in ['generate', 'write_numpy']
  assert args.depth > 1

  if args.type == "endgame":
    assert len(args.pgnfiles) == 0

  if args.mode == 'generate':
    # generate work
    fenQueue = Queue()
    resultQueue = Queue()

    analyzers = [Process(target=analyzer, args=(fenQueue, resultQueue, args)) for _ in range(args.num_workers)]
    for p in analyzers:
      p.start()

    sqlThread = Process(target=sql_inserter, args=(resultQueue, args))
    sqlThread.start()

    iterator = None
    if args.type == 'endgame':
      iterator = endgame_iterator()
    else:
      iterator = pgn_iterator(args.noise)

    for fen in iterator:
      fenQueue.put(fen)

  if args.mode == 'write_numpy':
    conn = sqlite3.connect("db.sqlite3")
    c = conn.cursor()
    c.execute(f"SELECT fen1, fen2, moverScore1, moverScore2, moverFeatures1, moverFeatures2 FROM {get_table_name(args)}")
    X, Y, F, T = [], [], [], []
    for fen1, fen2, moverScore1, moverScore2, moverFeatures1, moverFeatures2 in c:
      F.append(fen1 + ":" + fen2)
      Y.append([moverScore1, moverScore2])

      x1 = np.array([float(a) for a in moverFeatures1.split(' ')], dtype=np.int32)
      x2 = np.array([float(a) for a in moverFeatures2.split(' ')], dtype=np.int32)
      X.append([x1, x2])
      T.append([
        int(' w ' in fen1),
        int(' w ' in fen2),
      ])

    X = np.array(X, dtype=np.int16)
    Y = np.array(Y, dtype=np.int16)
    F = np.array(F)
    T = np.array(T, dtype=np.int8)
    print(X.shape, Y.shape, F.shape, T.shape)
    np.save(os.path.join('traindata', f'x.pair.{get_table_name(args)}.npy'), X)
    np.save(os.path.join('traindata', f'y.pair.{get_table_name(args)}.npy'), Y)
    np.save(os.path.join('traindata', f'fen.pair.{get_table_name(args)}.npy'), F)
    np.save(os.path.join('traindata', f'turn.pair.{get_table_name(args)}.npy'), T)
    exit(0)
