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
    if numInserted % 50 == 0:
      print('%.1f inserts/sec (%.1f avg)' % (
        50 / (time.time() - tlast),
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

        stockfish = Stockfish(path=args.stockpath)

        if args.quiet != 2:
          fen1 = make_random(fen, random.randint(1, 4))
          fen2 = make_random(fen, random.randint(1, 4))
        else:
          # Use stockfish to find quiet FENs, which we define as "the best move is not a capture".
          # Try 8 times to generate 2 quiet fens.
          quietfens = []
          for _ in range(8):
            f = make_random(fen, random.randint(1, 4))
            if f in quietfens:
              continue
            try:
              tmp = stockfish.analyze(f, depth=args.depth)
              move = tmp['pv'][0]
              board = chess.Board(f)
              moves = list(board.legal_moves)
              moves = [m for m in moves if m.uci() == move]
              assert len(moves) == 1
              if board.piece_at(moves[0].to_square) is None:
                quietfens.append(f)
            except RuntimeError as _:
              stockfish._p.terminate()
              break
            if len(quietfens) == 2:
              break

          if len(quietfens) != 2:
            stockfish._p.terminate()
            continue

          fen1, fen2 = quietfens

        if fen1 == fen2:
          stockfish._p.terminate()
          continue

        if fen1 is None or fen2 is None:
          stockfish._p.terminate()
          continue

        fen1, x1 = get_vec(fen1, args)
        fen2, x2 = get_vec(fen2, args)

        try:
          score1 = stockfish.analyze(fen1, depth=args.depth)['score']
        except RuntimeError as _:
          stockfish._p.terminate()
          continue
        if score1[0] == 'mate':
          score1 = score1[1] * args.clip
        else:
          score1 = score1[1]

        try:
          score2 = stockfish.analyze(fen2, depth=args.depth)['score']
        except RuntimeError as _:
          stockfish._p.terminate()
          continue
        if score2[0] == 'mate':
          score2 = score2[1] * args.clip
        else:
          score2 = score2[1]

        score1 = max(-args.clip, min(args.clip, score1))
        score2 = max(-args.clip, min(args.clip, score2))

        low = min(abs(score1), abs(score2))
        if abs(score1 - score2) / (low + 100) < 0.33:
          stockfish._p.terminate()
          continue

        resultQueue.put({
          "fen1": fen1,
          "fen2": fen2,
          "moverScore1": score1,
          "moverScore2": score2,
          "moverFeatures1": x1,
          "moverFeatures2": x2,
        })

        stockfish._p.terminate()

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
  command = ["./a.out", "mode", "printvec-cpu", "fen", *fen.split(' '), "makequiet", "1" if args.quiet == 1 else "0"]
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
  command = ["./a.out", "mode", "printvec-cpu", "fens", filename, "makequiet", "0"]
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
  parser.add_argument("--quiet", type=int, required=True, help="{0, 1, 2} (2 uses stockfish to determine quiet-ness)")
  parser.add_argument("--noise", type=int, default=0)
  parser.add_argument("--depth", type=int, default=10, help="{1, 2, ..}")
  parser.add_argument("--num_workers", type=int, default=4)
  parser.add_argument("--clip", type=int, default=5000)
  parser.add_argument("--stockpath", default="/usr/local/bin/stockfish", type=str)
  args = parser.parse_args()

  assert args.type in ["any", "endgame"]
  assert args.quiet in [0, 1, 2]
  assert args.mode in ['generate', 'write_numpy', 'update_features']
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

  if args.mode == 'update_features':
    conn = sqlite3.connect("db.sqlite3")
    c = conn.cursor()
    c.execute(f"""DROP TABLE IF EXISTS tmpTable""")
    c.execute(f"""CREATE TABLE IF NOT EXISTS tmpTable (
      fen1 BLOB,
      fen2 BLOB,
      moverScore1 INTEGER,
      moverScore2 INTEGER,
      moverFeatures1 BLOB,
      moverFeatures2 BLOB
    );""")
    c.execute(f"SELECT fen1, fen2, moverScore1, moverScore2 FROM {get_table_name(args)}")
    A = c.fetchall()

    kTmpFn = '/tmp/fens.txt'
    if os.path.exists(kTmpFn):
      os.remove(kTmpFn)
    with open(kTmpFn, 'w+') as f:
      for fen1, fen2, _, _ in A:
        f.write(fen1 + '\n')
        f.write(fen2 + '\n')

    command = ["./a.out", "mode", "printvec-cpu", "fens", kTmpFn, "makequiet", "0"]
    lines = subprocess.check_output(command).decode().strip().split('\n')
    for line in lines:
      assert not line.startswith('PRINT FEATURE VEC FAIL')
    assert len(lines) == len(A) * 4
    for i in range(0, len(lines), 4):
      fen1 = lines[i + 0]
      x1 = [int(x) for x in lines[i + 1].split(' ')]
      fen2 = lines[i + 2]
      x2 = [int(x) for x in lines[i + 3].split(' ')]
      f1, f2, moverScore1, moverScore2 = A[i//4]
      assert fen1 == f1
      assert fen2 == f2
      c.execute(f"""INSERT INTO tmpTable
        (fen1, fen2, moverScore1, moverScore2, moverFeatures1, moverFeatures2)
        VALUES (?, ?, ?, ?, ?, ?)""", (
        fen1,
        fen2,
        moverScore1,
        moverScore2,
        ' '.join(str(a) for a in x1),
        ' '.join(str(a) for a in x2),
      ))
    c.execute(f"""DROP TABLE IF EXISTS {get_table_name(args)}""")
    c.execute(f"""ALTER TABLE tmpTable RENAME TO {get_table_name(args)}""")
    conn.commit()
    exit(1)

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

    kPieceName = 'PNBRQKpnbrqk'
    A = np.zeros((len(F), 2, 12, 64), dtype=np.int8)
    for i in range(len(F)):
      if i % 5000 == 0:
        print(i, len(F))
      fens = F[i].split(':')
      for j in range(2):
        board = chess.Board(fens[j])
        tiles = [line.split(' ') for line in str(board).split('\n')]
        for y in range(8):
          for x in range(8):
            if tiles[y][x] != '.':
              A[i, j, kPieceName.index(tiles[y][x]), y * 8 + x] = 1

    X = np.array(X, dtype=np.int16)
    Y = np.array(Y, dtype=np.int16)
    F = np.array(F)
    T = np.array(T, dtype=np.int8)
    print(X.shape, Y.shape, F.shape, T.shape, A.shape)
    np.save(os.path.join('traindata', f'x.pair.{get_table_name(args)}.npy'), X)
    np.save(os.path.join('traindata', f'y.pair.{get_table_name(args)}.npy'), Y)
    np.save(os.path.join('traindata', f'fen.pair.{get_table_name(args)}.npy'), F)
    np.save(os.path.join('traindata', f'turn.pair.{get_table_name(args)}.npy'), T)
    np.save(os.path.join('traindata', f'pm.pair.{get_table_name(args)}.npy'), A)
    exit(0)

