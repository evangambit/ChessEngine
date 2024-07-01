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
  "IN_CHECK",
  "KING_ON_BACK_RANK",
  "KING_ON_CENTER_FILE",
  "KING_ACTIVE",
  "THREATS_NEAR_KING_2",
  "THREATS_NEAR_KING_3",
  "PASSED_PAWNS",
  "ISOLATED_PAWNS",
  "DOUBLED_PAWNS",
  "DOUBLE_ISOLATED_PAWNS",
  "PAWNS_CENTER_16",
  "PAWNS_CENTER_4",
  "ADVANCED_PASSED_PAWNS_2",
  "ADVANCED_PASSED_PAWNS_3",
  "ADVANCED_PASSED_PAWNS_4",
  "PAWN_MINOR_CAPTURES",
  "PAWN_MAJOR_CAPTURES",
  "PROTECTED_PAWNS",
  "PROTECTED_PASSED_PAWNS",
  "BISHOPS_DEVELOPED",
  "BISHOP_PAIR",
  "BLOCKADED_BISHOPS",
  "SCARY_BISHOPS",
  "SCARIER_BISHOPS",
  "BLOCKADED_ROOKS",
  "SCARY_ROOKS",
  "INFILTRATING_ROOKS",
  "KNIGHTS_DEVELOPED",
  "KNIGHT_MAJOR_CAPTURES",
  "KNIGHTS_CENTER_16",
  "KNIGHTS_CENTER_4",
  "KNIGHT_ON_ENEMY_SIDE",
  "OUR_HANGING_PAWNS",
  "OUR_HANGING_KNIGHTS",
  "OUR_HANGING_BISHOPS",
  "OUR_HANGING_ROOKS",
  "OUR_HANGING_QUEENS",
  "THEIR_HANGING_PAWNS",
  "THEIR_HANGING_KNIGHTS",
  "THEIR_HANGING_BISHOPS",
  "THEIR_HANGING_ROOKS",
  "THEIR_HANGING_QUEENS",
  "LONELY_KING_IN_CENTER",
  "LONELY_KING_AWAY_FROM_ENEMY_KING",
  "TIME",
  "KPVK_OPPOSITION",
  "SQUARE_RULE",
  "ADVANCED_PAWNS_1",
  "ADVANCED_PAWNS_2",
  "OPEN_ROOKS",
  "ROOKS_ON_THEIR_SIDE",
  "KING_IN_FRONT_OF_PASSED_PAWN",
  "KING_IN_FRONT_OF_PASSED_PAWN2",
  "OUR_MATERIAL_THREATS",
  "THEIR_MATERIAL_THREATS",
  "LONELY_KING_ON_EDGE",
  "OUTPOSTED_KNIGHTS",
  "OUTPOSTED_BISHOPS",
  "PAWN_MOVES",
  "KNIGHT_MOVES",
  "BISHOP_MOVES",
  "ROOK_MOVES",
  "QUEEN_MOVES",
  "PAWN_MOVES_ON_THEIR_SIDE",
  "KNIGHT_MOVES_ON_THEIR_SIDE",
  "BISHOP_MOVES_ON_THEIR_SIDE",
  "ROOK_MOVES_ON_THEIR_SIDE",
  "QUEEN_MOVES_ON_THEIR_SIDE",
  "KING_HOME_QUALITY",
  "BISHOPS_BLOCKING_KNIGHTS",
  "OUR_HANGING_PAWNS_2",
  "OUR_HANGING_KNIGHTS_2",
  "OUR_HANGING_BISHOPS_2",
  "OUR_HANGING_ROOKS_2",
  "OUR_HANGING_QUEENS_2",
  "THEIR_HANGING_PAWNS_2",
  "THEIR_HANGING_KNIGHTS_2",
  "THEIR_HANGING_BISHOPS_2",
  "THEIR_HANGING_ROOKS_2",
  "THEIR_HANGING_QUEENS_2",
  "QUEEN_THREATS_NEAR_KING",
  "MISSING_FIANCHETTO_BISHOP",
  "NUM_BAD_SQUARES_FOR_PAWNS",
  "NUM_BAD_SQUARES_FOR_MINORS",
  "NUM_BAD_SQUARES_FOR_ROOKS",
  "NUM_BAD_SQUARES_FOR_QUEENS",
  "IN_TRIVIAL_CHECK",
  "IN_DOUBLE_CHECK",
  "THREATS_NEAR_OUR_KING",
  "THREATS_NEAR_THEIR_KING",
  "NUM_PIECES_HARRASSABLE_BY_PAWNS",
  "PAWN_CHECKS",
  "KNIGHT_CHECKS",
  "BISHOP_CHECKS",
  "ROOK_CHECKS",
  "QUEEN_CHECKS",
  "BACK_RANK_MATE_THREAT_AGAINST_US",
  "BACK_RANK_MATE_THREAT_AGAINST_THEM",
  "OUR_KING_HAS_0_ESCAPE_SQUARES",
  "THEIR_KING_HAS_0_ESCAPE_SQUARES",
  "OUR_KING_HAS_1_ESCAPE_SQUARES",
  "THEIR_KING_HAS_1_ESCAPE_SQUARES",
  "OUR_KING_HAS_2_ESCAPE_SQUARES",
  "THEIR_KING_HAS_2_ESCAPE_SQUARES",
  "OPPOSITE_SIDE_KINGS_PAWN_STORM",
  "IN_CHECK_AND_OUR_HANGING_QUEENS",
  "PROMOTABLE_PAWN",
]

def sql_inserter(resultQueue, args):
  conn = sqlite3.connect("db.sqlite3")
  c = conn.cursor()
  c.execute(f"""CREATE TABLE IF NOT EXISTS {get_table_name(args)} (
    fen STRING,
    Move1 STRING,
    MoverScore1 INTEGER,
    Move1IsCapture INTEGER,
    Move2 STRING,
    MoverScore2 INTEGER,
    Move2IsCapture INTEGER,
    Move3 STRING,
    MoverScore3 INTEGER,
    Move3IsCapture INTEGER,
    Move4 STRING,
    MoverScore4 INTEGER,
    Move4IsCapture INTEGER
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
      (fen,
      Move1, MoverScore1, Move1IsCapture,
      Move2, MoverScore2, Move2IsCapture,
      Move3, MoverScore3, Move3IsCapture,
      Move4, MoverScore4, Move4IsCapture
      )
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", (
      result["fen"],
      result["Move1"],
      result["MoverScore1"],
      result["Move1IsCapture"],
      result["Move2"],
      result["MoverScore2"],
      result["Move2IsCapture"],
      result["Move3"],
      result["MoverScore3"],
      result["Move3IsCapture"],
      result["Move4"],
      result["MoverScore4"],
      result["Move4IsCapture"],
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
  return max(-clip, min(clip, e[1]))

def is_capture(fen, uci_move):
  board = chess.Board(fen)
  move = chess.Move.from_uci(uci_move)
  return board.is_capture(move)

def analyzer(fenQueue, resultQueue, args):
    stockfish = UciPlayer(path=args.stockpath)
    stockfish.set_multipv(4)
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
          stockfish.set_multipv(4)
          continue

        if len(moves) < 4:
          continue

        moves[0]['score'] = eval2score(moves[0]['score'], clip=1000)
        moves[1]['score'] = eval2score(moves[1]['score'], clip=1000)
        moves[2]['score'] = eval2score(moves[2]['score'], clip=1000)
        moves[3]['score'] = eval2score(moves[3]['score'], clip=1000)

        # We've already searched this position... might as well insert it. We can filter afterwards.
        # Skip not-quiet moves.
        # if abs(eval2score(moves[0]['score'], clip=1000) - eval2score(moves[1]['score'], clip=1000)) > 100:
        #   continue

        resultQueue.put({
          "fen": fen,
          "Move1": moves[0]['pv'][0],
          "MoverScore1": moves[0]['score'],
          "Move1IsCapture": is_capture(fen, moves[0]['pv'][0]),
          "Move2": moves[1]['pv'][0],
          "MoverScore2": moves[1]['score'],
          "Move2IsCapture": is_capture(fen, moves[1]['pv'][0]),
          "Move3": moves[2]['pv'][0],
          "MoverScore3": moves[2]['score'],
          "Move3IsCapture": is_capture(fen, moves[2]['pv'][0]),
          "Move4": moves[3]['pv'][0],
          "MoverScore4": moves[3]['score'],
          "Move4IsCapture": is_capture(fen, moves[3]['pv'][0]),
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
  parser.add_argument("--analysis", type=str, default='')
  parser.add_argument("--filter_range", type=int, default=300, help="9999 to disable")
  parser.add_argument("--filter_quiet1", type=int, default=50, help="9999 to disable")
  parser.add_argument("--filter_quiet2", type=int, default=50, help="9999 to disable")
  parser.add_argument("--filter_quiet3", type=int, default=50, help="9999 to disable")
  args = parser.parse_args()

  assert args.stage in ['positions', 'vectors', 'analysis']
  assert args.noise >= 0
  assert args.depth > 1

  if args.stage == 'positions':
    fenQueue = Queue()
    resultQueue = Queue()

    try:
      with sqlite3.connect("db.sqlite3") as conn:
        c = conn.cursor()
        c.execute(f"""SELECT fen FROM {get_table_name(args)}""")
        fens = set([x[0] for x in c.fetchall()])
    except sqlite3.OperationalError:
      fens = set()

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
  elif args.stage == 'vectors':
    conn = sqlite3.connect("db.sqlite3")
    c = conn.cursor()
    c.execute(f"""SELECT fen, MoverScore1, MoverScore2, MoverScore3, MoverScore4 FROM {get_table_name(args)}
      WHERE
          MoverScore1 < MoverScore2 + {args.filter_quiet1}
      AND MoverScore1 < MoverScore3 + {args.filter_quiet2}
      AND MoverScore1 < MoverScore4 + {args.filter_quiet3}
      AND MoverScore1 > -{args.filter_range}
      AND MoverScore1 < {args.filter_range}
      AND Move1IsCapture = 0
    """)
    rows = c.fetchall()
    fens = [row[0] for row in rows]
    evals = [row[1:] for row in rows]

    print('Found %i positions' % len(fens))

    # Remove duplicates
    A, B, C = [], [], []
    seen = set()
    for fen, e in zip(fens, evals):
      if fen in seen:
        continue
      seen.add(fen)
      A.append(fen)
      B.append(e)
      C.append(int(' w ' in fen) * 2 - 1)
    fens, evals, turns = A, B, C

    print('Found %i positions after removing duplicates' % len(fens))

    with open('/tmp/fens.txt', 'w') as f:
      f.write('\n'.join(fens))

    np.save('F.npy', np.array(fens))
    np.save('turns', np.array(turns, dtype=np.int16).reshape((-1, 1)))
    # np.save('Y.npy', np.array(evals, dtype=np.int16))

    # print('Computing tables')

    # # os.system("sh build.sh src/make_tables.cpp -o make_tables && ./make_tables")
    # os.system("./make_tables /tmp/fens.txt > /tmp/tables.txt")

    # tables = np.frombuffer(open('/tmp/tables.txt', 'rb').read(), dtype=np.int8) - 97
    # tables = tables.reshape(-1, 784)
    # np.save('tables.npy', tables)

