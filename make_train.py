import time
import argparse
import os
import random
import subprocess
import sqlite3

from multiprocessing import Process, Queue
 
import chess
from chess import pgn

from stockfish import Stockfish
from stockfish.models import StockfishException

import numpy as np

"""
To generate training data from scratch:

$ python3 make_train.py --mode generate --type any --quiet 1 lichess_elite_2022-05.pgn

python3 -i make_train.py --mode write_numpy --type any --quiet 1 lichess_elite_2022-05.pgn

zip traindata.zip -r traindata




Tactics

$ python3 make_train.py --mode generate --type tactics --quiet 0 lichess_elite_2022-05.pgn

"""

def sql_inserter(resultQueue, args):
  conn = sqlite3.connect("db.sqlite3")
  c = conn.cursor()
  c.execute(f"""CREATE TABLE IF NOT EXISTS {get_table_name(args)} (
    fen BLOB,
    bestMove BLOB,
    delta INTEGER,
    moverScore INTEGER,
    moverFeatures BLOB    -- from mover's perspective
  );""")

  fens = set()
  c.execute(f"SELECT fen FROM {get_table_name(args)}")
  for fen, in c:
    fens.add(fen)

  t0 = time.time()
  numInserted = 0

  while True:
    result = resultQueue.get()

    if result["fen"] in fens:
      continue
    fens.add(result["fen"])

    c.execute(f"""INSERT INTO {get_table_name(args)}
      (fen, bestMove, delta, moverScore, moverFeatures) 
      VALUES (?, ?, ?, ?, ?)""", (
      result["fen"],
      result["bestmove"],
      result["scoreDelta"],
      result["evaluation"],
      ' '.join(str(a) for a in result["features"]),
    ))

    numInserted += 1
    if numInserted % 500 == 0:
      print('%.1f inserts/sec' % (numInserted / (time.time() - t0)), len(fens))
      conn.commit()

def analyzer(fenQueue, resultQueue, args):
    stockfish = Stockfish(path=args.stockpath, depth=args.depth)
    while True:
        fen = fenQueue.get()

        fen, features = get_vec(fen, args)

        try:
          stockfish.set_fen_position(fen)
          moves = stockfish.get_top_moves(2)
        except StockfishException:
          print('reboot')
          # Restart stockfish.
          stockfish = Stockfish(path=args.stockpath, depth=args.depth)
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

        resultQueue.put({
          "fen": fen,
          "bestmove": bestmove,
          "scoreDelta": scoreDelta,
          "evaluation": evaluation,
          "features": features,
        })

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

def get_vec(fen, args):
  command = ["./a.out", "mode", "printvec-cpu", "fen", *fen.split(' '), "makequiet", str(args.quiet)]
  lines = subprocess.check_output(command).decode().strip().split('\n')
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
  parser.add_argument("--type", type=str, required=True, help="{any, tactics, endgame}")
  parser.add_argument("--quiet", type=int, required=True, help="{0, 1}")
  parser.add_argument("--noise", type=int, default=0, help="{0, 1, 2, ..}")
  parser.add_argument("--depth", type=int, default=10, help="{1, 2, ..}")
  parser.add_argument("--stockpath", default="/usr/local/bin/stockfish", type=str)
  args = parser.parse_args()

  assert args.type in ["any", "tactics", "endgame"]
  assert args.quiet in [0, 1]
  assert args.mode in ['generate', 'update_features', 'write_numpy']
  assert args.noise >= 0
  assert args.depth > 1

  if args.type == "endgame":
    assert len(args.pgnfiles) == 0
    assert args.noise == 0

  if args.mode == 'update_features':
    conn = sqlite3.connect("db.sqlite3")
    c = conn.cursor()
    c.execute(f"SELECT fen, bestMove, delta, moverScore FROM {get_table_name(args)}")
    A = {}
    for fen, bestmove, delta, moverScore in c:
      A[fen] = (bestmove, delta, moverScore)

    c.execute(f"""DROP TABLE IF EXISTS tmpTable""")
    c.execute(f"""CREATE TABLE tmpTable (
      fen BLOB,
      bestMove BLOB,
      delta INTEGER,
      moverScore INTEGER,
      moverFeatures BLOB    -- from mover's perspective
    );""")
    fens = list(A.keys())
    writesSinceCommit = 0
    totalWrites = 0
    for fen, x in get_vecs(fens, args):
      if fen not in A:
        print('x')
        continue
      c.execute(f"""INSERT INTO tmpTable
        (fen, bestMove, delta, moverScore, moverFeatures) 
        VALUES (?, ?, ?, ?, ?)""", (
        fen,
        *A[fen],
        ' '.join(str(a) for a in x),
      ))

      writesSinceCommit += 1
      totalWrites += 1
      if writesSinceCommit >= 1000:
        conn.commit()
        print('commit', totalWrites, len(fens), len(x))
        writesSinceCommit = 0
    conn.commit()
    c.execute(f"""DROP TABLE IF EXISTS {get_table_name(args)}""")
    c.execute(f"""ALTER TABLE tmpTable RENAME TO {get_table_name(args)}""")
    exit(0)

  if args.mode == 'generate':
    # generate work
    fenQueue = Queue()
    resultQueue = Queue()

    analyzers = [Process(target=analyzer, args=(fenQueue, resultQueue, args)) for _ in range(10)]
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
    c.execute(f"SELECT fen, moverScore, moverFeatures FROM {get_table_name(args)}")
    X, Y, F = [], [], []
    for fen, moverScore, moverFeatures in c:
      F.append(fen)
      Y.append(moverScore)

      x = np.array([float(a) for a in moverFeatures.split(' ')], dtype=np.int32)
      X.append(x)

    X = np.array(X).astype(np.int16)
    Y = np.array(Y)
    F = np.array(F)
    print(X.shape, Y.shape, F.shape)
    np.save(os.path.join('traindata', f'x.{args.type}.npy'), X)
    np.save(os.path.join('traindata', f'y.{args.type}.npy'), Y)
    np.save(os.path.join('traindata', f'f.{args.type}.npy'), F)
    exit(0)

