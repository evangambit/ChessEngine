import time
import argparse
import os
import random
import subprocess
import sqlite3

from scipy.special import logit

from multiprocessing import Process, Queue
 
import chess
from chess import pgn

from uci_player import UciPlayer

import numpy as np

"""
To generate training data from scratch:

$ python3 -i make_train.py --mode generate --type any noise 2 ~/Downloads/lichess_elite_2022-07.pgn

$ python3 -i make_train.py --mode write_numpy --type any noise 2 ~/Downloads/lichess_elite_2022-07.pgn

$ python3 train.py

"""

def sql_inserter(resultQueue, args):
  conn = sqlite3.connect("db.sqlite3")
  c = conn.cursor()
  c.execute(f"""CREATE TABLE IF NOT EXISTS {get_table_name(args)} (
    fen BLOB,
    win INTEGER,
    draw INTEGER,
    lose INTEGER,
    moverFeatures BLOB    -- from mover's perspective
  );""")

  fens = set()
  c.execute(f"SELECT fen FROM {get_table_name(args)}")
  for fen, in c:
    fens.add(fen)

  t0 = time.time()
  tlast = time.time()
  numInserted = 0

  while True:
    result = resultQueue.get()

    if result["fen"] in fens:
      continue
    fens.add(result["fen"])

    c.execute(f"""INSERT INTO {get_table_name(args)}
      (fen, win, draw, lose, moverFeatures)
      VALUES (?, ?, ?, ?, ?)""", (
      result["fen"],
      result["win"],
      result["draw"],
      result["lose"],
      ' '.join(str(a) for a in result["features"]),
    ))

    numInserted += 1
    if numInserted % 50 == 0:
      print('%.1f inserts/sec (%.1f avg)' % (
        50 / (time.time() - tlast),
        numInserted / (time.time() - t0)
      ), len(fens))
      tlast = time.time()
      conn.commit()

def analyzer(fenQueue, resultQueue, args):
    stockfish = UciPlayer(path=args.stockpath)
    stockfish.set_multipv(1)
    stockfish.setoption('UCI_ShowWDL', 'true')
    while True:
        fen = fenQueue.get()
        if len(fen) == 2:
          fen, features = fen
        else:
          fen, features = get_vec(fen, args)
          if fen is None:
            continue

        board = chess.Board(fen)
        if not board.is_valid():
          continue

        try:
          stockfish.command(f"position {fen}")
          moves = stockfish.go(fen=fen, depth=args.depth)
        except KeyboardInterrupt:
          stockfish = UciPlayer(path=args.stockpath)
          stockfish.set_multipv(1)
          stockfish.setoption('UCI_ShowWDL', 'true')
          continue

        board = chess.Board(fen)
        captures = set([str(m) for m in list(board.legal_moves) if board.piece_at(m.to_square) is not None])

        if len(moves) != 1:
          continue

        if moves[0]['pv'][0] in captures:
          continue

        bestmove = moves[0]['pv'][0]

        resultQueue.put({
          "fen": fen,
          "win": moves[0]['wdl'][0],
          "draw": moves[0]['wdl'][1],
          "lose": moves[0]['wdl'][2],
          "features": features,
        })

def pgn_iterator(noise, downsample):
  seen = set()
  assert len(args.pgnfiles) > 0
  for filename in args.pgnfiles:
    f = open(filename, 'r')
    game = pgn.read_game(f)
    while game is not None:
      for node in game.mainline():
        if random.randint(1, downsample) != 1:
          continue
        board = node.board()
        for _ in range(noise):
          moves = list(board.legal_moves)
          if len(moves) == 0:
            break
          random.shuffle(moves)
          board.push(moves[0])
        fen = board.fen()
        if fen in seen:
          continue
        seen.add(fen)
        if len(list(board.legal_moves)) == 0:
          continue
        seen.add(fen)
        yield fen
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

def leafer(it):
  # g++ src/uci.cpp src/game/*.cpp -std=c++20 -O3 -DNDEBUG -DPRINT_LEAVES -o uci_leaves
  p = subprocess.Popen("./uci_leaves", stdin=subprocess.PIPE, stdout=subprocess.PIPE)
  p.stdin.write(("uci\n").encode())
  for fen in it:
    p.stdin.write(f"position fen {fen}\n".encode())
    p.stdin.flush()
    p.stdin.write("go nodes 20000\n".encode())
    p.stdin.flush()
    while True:
      line = p.stdout.readline().decode().strip()
      if line.startswith('bestmove'):
        break
      if line.startswith('info') or line.startswith('Chess') or line.startswith('Position'):
        continue
      leaf_fen = line
      line = p.stdout.readline().decode().strip()
      leaf_features = [int(x) for x in line.split(' ')]
      yield leaf_fen, leaf_features

def get_vec(fen, args):
  command = ["./main", "mode", "printvec-cpu", "fen", *fen.split(' ')]
  lines = subprocess.check_output(command).decode().strip().split('\n')
  if lines[1].startswith('PRINT FEATURE VEC FAIL'):
    return None, None
  assert len(lines) == 2, lines
  fen = lines[0]
  x = [int(val) for val in lines[1].split(' ')]
  return fen, x

def get_vecs(fens, args):
  filename = '/tmp/fens.txt'
  with open(filename, 'w+') as f:
    f.write('\n'.join(fens))
  command = ["./main", "mode", "printvec-cpu", "fens", filename]
  lines = subprocess.check_output(command).decode().strip().split('\n')
  lines = [line for line in lines if not line.startswith('PRINT FEATURE VEC FAIL')]
  for i in range(0, len(lines), 2):
    x = [int(val) for val in lines[i + 1].split(' ')]
    yield lines[i + 0], x

def get_table_name(args):
  r = f"make_train_{args.type}_d{args.depth}_n{args.noise}"
  if args.leaf:
    r += '_leaf'
  return r

if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument("pgnfiles", nargs='*')
  parser.add_argument("--mode", type=str, required=True, help="{generate, update_features, write_numpy}")
  parser.add_argument("--type", type=str, required=True, help="{any, endgame}")
  parser.add_argument("--noise", type=int, default=0, help="{0, 1, 2, ..}")
  parser.add_argument("--depth", type=int, default=10, help="{1, 2, ..}")
  parser.add_argument("--stockpath", default="/usr/local/bin/stockfish", type=str)
  parser.add_argument("--downsample", type=int, default=50, help="{1, 2, ..}")
  parser.add_argument("--leaf", type=int, default=0, help="0 or 1")
  args = parser.parse_args()

  assert args.type in ["any", "endgame"]
  assert args.mode in ['generate', 'update_features', 'write_numpy']
  assert args.noise >= 0
  assert args.depth > 1

  if args.type == "endgame":
    assert len(args.pgnfiles) == 0
    assert args.noise == 0

  if args.mode == 'update_features':
    conn = sqlite3.connect("db.sqlite3")
    c = conn.cursor()
    c.execute(f"SELECT fen, win, draw, lose, moverFeatures FROM {get_table_name(args)}")
    A = {}
    for fen, win, draw, lose, moverFeatures in c:
      A[fen] = (win, draw, lose, moverFeatures)

    c.execute(f"""DROP TABLE IF EXISTS tmpTable""")
    c.execute(f"""CREATE TABLE tmpTable (
      fen BLOB,
      win INTEGER,
      draw INTEGER,
      lose INTEGER,
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
        (fen, win, draw, lose, moverFeatures) 
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
      iterator = pgn_iterator(args.noise, args.downsample)

    if args.leaf:
      iterator = leafer(iterator)

    for fen in iterator:
      fenQueue.put(fen)

  if args.mode == 'write_numpy':
    conn = sqlite3.connect("db.sqlite3")
    c = conn.cursor()
    c.execute(f"SELECT fen, win, draw, lose, moverFeatures FROM {get_table_name(args)}")
    X, Y, F = [], [], []
    for fen, win, draw, lose, moverFeatures in c:
      win += 1
      draw += 1
      lose += 1
      F.append(fen)
      Y.append(logit((win + draw * 0.5) / (win + draw + lose)))

      x = np.array([float(a) for a in moverFeatures.split(' ')], dtype=np.int32)
      X.append(x)

    X = np.array(X).astype(np.int16)
    Y = np.array(Y)
    F = np.array(F)
    print(X.shape, Y.shape, F.shape)
    np.save(os.path.join('traindata', f'x.{get_table_name(args)}.npy'), X)
    np.save(os.path.join('traindata', f'y.{get_table_name(args)}.npy'), Y)
    np.save(os.path.join('traindata', f'f.{get_table_name(args)}.npy'), F)
    exit(0)

