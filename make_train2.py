import time
import json
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

from tqdm import tqdm

ESTR = [
  "OUR_PAWNS",
  "OUR_KNIGHTS",
  "OUR_BISHOPS",
  "OUR_ROOKS",
  "OUR_QUEENS",
  "THEIR_PAWNS",
  "THEIR_KNIGHTS",
  "THEIR_BISHOPS",
  "THEIR_ROOKS",
  "THEIR_QUEENS",
  # "IN_CHECK",
  # "KING_ON_BACK_RANK",
  # "KING_ON_CENTER_FILE",
  # "KING_ACTIVE",
  # "THREATS_NEAR_KING_2",
  # "THREATS_NEAR_KING_3",
  # "PASSED_PAWNS",
  # "ISOLATED_PAWNS",
  # "DOUBLED_PAWNS",
  # "DOUBLE_ISOLATED_PAWNS",
  # "PAWNS_CENTER_16",
  # "PAWNS_CENTER_4",
  # "ADVANCED_PASSED_PAWNS_2",
  # "ADVANCED_PASSED_PAWNS_3",
  # "ADVANCED_PASSED_PAWNS_4",
  # "PAWN_MINOR_CAPTURES",
  # "PAWN_MAJOR_CAPTURES",
  # "PROTECTED_PAWNS",
  # "PROTECTED_PASSED_PAWNS",
  # "BISHOPS_DEVELOPED",
  # "BISHOP_PAIR",
  # "BLOCKADED_BISHOPS",
  # "SCARY_BISHOPS",
  # "SCARIER_BISHOPS",
  # "BLOCKADED_ROOKS",
  # "SCARY_ROOKS",
  # "INFILTRATING_ROOKS",
  # "KNIGHTS_DEVELOPED",
  # "KNIGHT_MAJOR_CAPTURES",
  # "KNIGHTS_CENTER_16",
  # "KNIGHTS_CENTER_4",
  # "KNIGHT_ON_ENEMY_SIDE",
  # "OUR_HANGING_PAWNS",
  # "OUR_HANGING_KNIGHTS",
  # "OUR_HANGING_BISHOPS",
  # "OUR_HANGING_ROOKS",
  # "OUR_HANGING_QUEENS",
  # "THEIR_HANGING_PAWNS",
  # "THEIR_HANGING_KNIGHTS",
  # "THEIR_HANGING_BISHOPS",
  # "THEIR_HANGING_ROOKS",
  # "THEIR_HANGING_QUEENS",
  # "LONELY_KING_IN_CENTER",
  # "LONELY_KING_AWAY_FROM_ENEMY_KING",
  # "TIME",
  # "KPVK_OPPOSITION",
  # "SQUARE_RULE",
  # "ADVANCED_PAWNS_1",
  # "ADVANCED_PAWNS_2",
  # "OPEN_ROOKS",
  # "ROOKS_ON_THEIR_SIDE",
  # "KING_IN_FRONT_OF_PASSED_PAWN",
  # "KING_IN_FRONT_OF_PASSED_PAWN2",
  # "OUR_MATERIAL_THREATS",
  # "THEIR_MATERIAL_THREATS",
  # "LONELY_KING_ON_EDGE",
  # "OUTPOSTED_KNIGHTS",
  # "OUTPOSTED_BISHOPS",
  # "PAWN_MOVES",
  # "KNIGHT_MOVES",
  # "BISHOP_MOVES",
  # "ROOK_MOVES",
  # "QUEEN_MOVES",
  # "PAWN_MOVES_ON_THEIR_SIDE",
  # "KNIGHT_MOVES_ON_THEIR_SIDE",
  # "BISHOP_MOVES_ON_THEIR_SIDE",
  # "ROOK_MOVES_ON_THEIR_SIDE",
  # "QUEEN_MOVES_ON_THEIR_SIDE",
  # "KING_HOME_QUALITY",
  # "BISHOPS_BLOCKING_KNIGHTS",
  # "OUR_HANGING_PAWNS_2",
  # "OUR_HANGING_KNIGHTS_2",
  # "OUR_HANGING_BISHOPS_2",
  # "OUR_HANGING_ROOKS_2",
  # "OUR_HANGING_QUEENS_2",
  # "THEIR_HANGING_PAWNS_2",
  # "THEIR_HANGING_KNIGHTS_2",
  # "THEIR_HANGING_BISHOPS_2",
  # "THEIR_HANGING_ROOKS_2",
  # "THEIR_HANGING_QUEENS_2",
  # "QUEEN_THREATS_NEAR_KING",
  # "MISSING_FIANCHETTO_BISHOP",
  # "NUM_BAD_SQUARES_FOR_PAWNS",
  # "NUM_BAD_SQUARES_FOR_MINORS",
  # "NUM_BAD_SQUARES_FOR_ROOKS",
  # "NUM_BAD_SQUARES_FOR_QUEENS",
  # "IN_TRIVIAL_CHECK",
  # "IN_DOUBLE_CHECK",
  # "THREATS_NEAR_OUR_KING",
  # "THREATS_NEAR_THEIR_KING",
  # "NUM_PIECES_HARRASSABLE_BY_PAWNS",
  # "PAWN_CHECKS",
  # "KNIGHT_CHECKS",
  # "BISHOP_CHECKS",
  # "ROOK_CHECKS",
  # "QUEEN_CHECKS",
  # "BACK_RANK_MATE_THREAT_AGAINST_US",
  # "BACK_RANK_MATE_THREAT_AGAINST_THEM",
  # "OUR_KING_HAS_0_ESCAPE_SQUARES",
  # "THEIR_KING_HAS_0_ESCAPE_SQUARES",
  # "OUR_KING_HAS_1_ESCAPE_SQUARES",
  # "THEIR_KING_HAS_1_ESCAPE_SQUARES",
  # "OUR_KING_HAS_2_ESCAPE_SQUARES",
  # "THEIR_KING_HAS_2_ESCAPE_SQUARES",
  # "OPPOSITE_SIDE_KINGS_PAWN_STORM",
  # "IN_CHECK_AND_OUR_HANING_QUEENS",
  # "PROMOTABLE_PAWN",
  # "THEIR_HANGING_PAWNS_MODERN",
  # "THEIR_HANGING_KNIGHTS_MODERN",
  # "THEIR_HANGING_BISHOPS_MODERN",
  # "THEIR_HANGING_ROOKS_MODERN",
  # "THEIR_HANGING_QUEENS_MODERN",

  "TIME",
  "CENTER_CONTROL",
  "DOMINATION",
  "BACK_RANK_CONTROL",
  "FREEDOM",
  "KING_HOME_QUALITY",
  "OUR_HANGING_PAWNS",
  "THEIR_HANGING_PAWNS",
  "OUR_HANGING_KNIGHTS",
  "THEIR_HANGING_KNIGHTS",
  "OUR_HANGING_BISHOPS",
  "THEIR_HANGING_BISHOPS",
  "OUR_HANGING_ROOKS",
  "THEIR_HANGING_ROOKS",
  "OUR_HANGING_QUEENS",
  "THEIR_HANGING_QUEENS",
  "OUR_HANGING_PAWNS_2",
  "THEIR_HANGING_PAWNS_2",
  "OUR_HANGING_KNIGHTS_2",
  "THEIR_HANGING_KNIGHTS_2",
  "OUR_HANGING_BISHOPS_2",
  "THEIR_HANGING_BISHOPS_2",
  "OUR_HANGING_ROOKS_2",
  "THEIR_HANGING_ROOKS_2",
  "OUR_HANGING_QUEENS_2",
  "THEIR_HANGING_QUEENS_2",
  "PASSED_PAWN_PROGRESS",
  "PASSED_PAWN_KING_DIST",
  "KING_ACTIVE",
  "KING_ON_BACK_RANK",
  "KING_ON_CENTER_FILE",
  "THREATS_NEAR_KING_2",
  "THREATS_NEAR_KING_3",
  "QUEEN_THREATS_NEAR_KING",
  "THREATS_NEAR_OUR_KING",
  "THREATS_NEAR_THEIR_KING",
  "KNOWN_DRAW",
  "SPECIAL",
  "BISHOP_PAIR",
  "OPEN_ROOKS",
]

def sql_inserter(resultQueue, args):
  conn = sqlite3.connect("db.sqlite3")
  c = conn.cursor()
  c.execute(f"""CREATE TABLE IF NOT EXISTS {get_table_name(args)} (
    fen BLOB,
    moverWin INTEGER,
    moverDraw INTEGER,
    moverLose INTEGER,
    moverEval STRING
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
      (fen, moverWin, moverDraw, moverLose, moverEval)
      VALUES (?, ?, ?, ?, ?)""", (
      result["fen"],
      result["moverWin"],
      result["moverDraw"],
      result["moverLose"],
      json.dumps(result["moverEval"]),
    ))

    numInserted += 1
    if numInserted % 50 == 0:
      print('%.1f inserts/sec (%.1f avg)' % (
        50 / (time.time() - tlast),
        numInserted / (time.time() - t0)
      ), len(fens))
      tlast = time.time()
      conn.commit()

def sign(x):
  return 1.0 if x > 0.0 else -1.0

def eval2score(e, clip):
  assert e[0] in ['cp', 'mate']
  assert isinstance(e[1], int)
  if e[0] == 'mate':
    return sign(e[1]) * clip
  return e[1]

def analyzer(fenQueue, resultQueue, args):
    stockfish = UciPlayer(path=args.stockpath)
    stockfish.set_multipv(2)
    stockfish.setoption('UCI_ShowWDL', 'true')
    while True:
        fen = fenQueue.get()

        board = chess.Board(fen)
        if not board.is_valid():
          continue

        try:
          stockfish.command(f"position {fen}")
          moves = stockfish.go(fen=fen, depth=args.depth)
        except KeyboardInterrupt:
          stockfish = UciPlayer(path=args.stockpath)
          stockfish.set_multipv(2)
          stockfish.setoption('UCI_ShowWDL', 'true')
          continue

        if len(moves) < 2:
          continue

        if 'score' not in moves[0]:
          print('****' * 9)
          print(fen)
          print('****' * 9)

        moves[0]['score'] = moves[0]['score']
        moves[1]['score'] = moves[1]['score']

        # Skip not-quiet moves.
        if abs(eval2score(moves[0]['score'], clip=1000) - eval2score(moves[1]['score'], clip=1000)) > 100:
          continue

        board = chess.Board(fen)

        bestmove = moves[0]['pv'][0]

        resultQueue.put({
          "fen": fen,
          "moverWin": moves[0]['wdl'][0],
          "moverDraw": moves[0]['wdl'][1],
          "moverLose": moves[0]['wdl'][2],
          "moverEval": moves[0]['score'],
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

def get_table_name(args):
  r = f"make_train2_d{args.depth}_n{args.noise}"
  return r

if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument("pgnfiles", nargs='*')
  parser.add_argument("--stage", type=str, help="{positions, vectors}")
  parser.add_argument("--noise", type=int, default=0, help="{0, 1, 2, ..}")
  parser.add_argument("--depth", type=int, default=6, help="{1, 2, ..}")
  parser.add_argument("--threads", type=int, default=10)
  parser.add_argument("--stockpath", default="/usr/local/bin/stockfish", type=str)
  parser.add_argument("--downsample", type=int, default=50, help="{1, 2, ..}")
  args = parser.parse_args()

  assert args.stage in ['positions', 'vectors']
  assert args.noise >= 0
  assert args.depth > 1

  if args.stage == 'positions':
    # generate work
    fenQueue = Queue()
    resultQueue = Queue()

    conn = sqlite3.connect("db.sqlite3")
    c = conn.cursor()
    c.execute(f"""SELECT fen FROM {get_table_name(args)}""")
    fens = set([x[0] for x in c.fetchall()])
    conn.close()

    analyzers = [Process(target=analyzer, args=(fenQueue, resultQueue, args)) for _ in range(args.threads)]
    for p in analyzers:
      p.start()

    sqlThread = Process(target=sql_inserter, args=(resultQueue, args))
    sqlThread.start()

    iterator = pgn_iterator(args.noise, args.downsample)

    for fen in iterator:
      if fen not in fens:
        fens.add(fen)
        fenQueue.put(fen)
  else:  # vectors
    conn = sqlite3.connect("db.sqlite3")
    c = conn.cursor()
    c.execute(f"""SELECT fen, moverEval FROM {get_table_name(args)}""")
    rows = c.fetchall()
    fens = [row[0] for row in rows]
    evals = [row[1] for row in rows]

    # Remove duplicates
    A, B = [], []
    seen = set()
    for fen, e in zip(fens, evals):
      if fen in seen:
        continue
      seen.add(fen)
      A.append(fen)
      B.append(e)
    fens, evals = A, B

    with open('/tmp/fens.txt', 'w') as f:
      f.write('\n'.join(fens))
    os.system("sh build.sh src/main.cpp -o main -DSquareControl && ./main mode printvec-cpu fens /tmp/fens.txt > /tmp/vecs.txt")

    os.system("sh build.sh src/make_tables.cpp -o make_tables -DSquareControl && ./make_tables")

    with open('/tmp/vecs.txt', 'r') as f:
      lines = f.read().split('\n')
    assert lines[-1] == ''
    lines.pop()
    assert len(lines) == len(fens) * 2

    X, F, Y = [], [], []
    good_indices = []
    for i, fen in tqdm(enumerate(fens), total=len(fens)):
      assert lines[2 * i + 0] == fen
      line = lines[2 * i + 1].split(' ')
      try:
        line = [int(val) for val in line]
        good_indices.append(i)
      except:
        line = [0] * X[-1].shape[0]
      F.append(fen)
      X.append(np.array(line, dtype=np.int16))
      Y.append(eval2score(json.loads(evals[i]), clip=1000))

    good_indices = np.array(good_indices)

    F = np.array(F)[good_indices]
    X = np.stack(X)[good_indices]
    Y = np.stack(Y, dtype=np.float32)[good_indices]

    np.save('F.npy', F)
    np.save('X.npy', X)
    np.save('Y.npy', Y)

    good = (np.abs(Y) < 300)
    X, Y, F = X[good], Y[good], F[good]

    thq = X[:, ESTR.index('THEIR_HANGING_QUEENS')].reshape(-1, 1)
    thr = X[:, ESTR.index('THEIR_HANGING_ROOKS')].reshape(-1, 1)
    thb = X[:, ESTR.index('THEIR_HANGING_BISHOPS')].reshape(-1, 1)
    thn = X[:, ESTR.index('THEIR_HANGING_KNIGHTS')].reshape(-1, 1)

    thq2 = X[:, ESTR.index('THEIR_HANGING_QUEENS_2')].reshape(-1, 1)
    thr2 = X[:, ESTR.index('THEIR_HANGING_ROOKS_2')].reshape(-1, 1)
    thb2 = X[:, ESTR.index('THEIR_HANGING_BISHOPS_2')].reshape(-1, 1)
    thn2 = X[:, ESTR.index('THEIR_HANGING_KNIGHTS_2')].reshape(-1, 1)

    ohq = X[:, ESTR.index('OUR_HANGING_QUEENS')].reshape(-1, 1)
    ohr = X[:, ESTR.index('OUR_HANGING_ROOKS')].reshape(-1, 1)
    ohb = X[:, ESTR.index('OUR_HANGING_BISHOPS')].reshape(-1, 1)
    ohn = X[:, ESTR.index('OUR_HANGING_KNIGHTS')].reshape(-1, 1)

    ohq2 = X[:, ESTR.index('OUR_HANGING_QUEENS_2')].reshape(-1, 1)
    ohr2 = X[:, ESTR.index('OUR_HANGING_ROOKS_2')].reshape(-1, 1)
    ohb2 = X[:, ESTR.index('OUR_HANGING_BISHOPS_2')].reshape(-1, 1)
    ohn2 = X[:, ESTR.index('OUR_HANGING_KNIGHTS_2')].reshape(-1, 1)

    def npmax(*args):
      r = np.maximum(args[0], args[1])
      for arg in args[2:]:
        r = np.maximum(r, arg)
      return r

    Z = np.concatenate([
      X,
      # TODO: why doesn't "4r2k/6pp/3KQq2/p2P4/6P1/8/8/2R3R1 b - - 5 34" count as OUR_HANGING_QUEENS?
      # Answer: bc we use "isHanging = threats.theirTargets & ~threats.ourTargets & pos.colorBitboards_[US]"
      # which is not very good for a queen
    ], 1)

    time = (X[:,ESTR.index('TIME')].clip(0, 18) / 18).reshape(-1, 1)
    Xlate = Z * time
    Xearly = Z * (1.0 - time)
    Ylate = Y * time.squeeze()
    Yearly = Y * (1.0 - time.squeeze())

    wEarly = np.linalg.lstsq(Xearly, Yearly, rcond=0.001)[0]
    wLate = np.linalg.lstsq(Xlate, Ylate, rcond=0.001)[0]

    Yhat = Xearly @ wEarly + Xlate @ wLate

    # for i in range(wEarly.shape[0]):
    #   print(
    #     ('%.1f' % wEarly[i]).rjust(8),
    #     ('%.1f' % wLate[i]).rjust(8),
    #     ESTR[i]
    #   )

    for i in range(wEarly.shape[0]):
      print(f"early += features[{ESTR[i]}] * {int(round(wEarly[i] * 4))} / 4;")

    for i in range(wEarly.shape[0]):
      print(f"late += features[{ESTR[i]}] * {int(round(wLate[i] * 4))} / 4;")

    I = np.argsort(-np.abs(Yhat - Y))

    Rearly = (Yearly - Xearly @ wEarly)
    Rlate = (Ylate - Xlate @ wLate)

    tables = np.frombuffer(open('/tmp/tables.txt', 'rb').read(), dtype=np.uint8)
    tables = tables.reshape(-1, 7 * 64)
    tables = tables[good_indices]
    np.save('tables.npy', tables)

    # Naive estimate of piece square tables
    for R in [Rearly, Rlate]:
      t = (((R - R.mean()) / R.std()).reshape((1, -1)) @ (tables / (tables.std(0) + 0.1))).reshape((7, 8, 8)) / tables.shape[0]
      t *= R.std() / ((tables.std(0) + 0.1).reshape(t.shape) + 1e-6)

