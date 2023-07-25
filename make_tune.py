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
python3 -i make_train.py --mode generate ~/Downloads/lichess_elite_2022-07.pgn --stockpath /usr/local/bin/stockfish

"""

def sql_inserter(resultQueue, args):
  conn = sqlite3.connect("db.sqlite3")
  c = conn.cursor()
  c.execute(f"""CREATE TABLE IF NOT EXISTS {get_table_name(args)} (
    fen BLOB,
    moves BLOB,
    scores BLOB
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

    c.execute(f"""INSERT INTO {get_table_name(args)} (fen, moves, scores) VALUES (?, ?, ?)""", (
      result["fen"],
      ' '.join(result["moves"]),
      ' '.join(str(x) for x in result["scores"]),
    ))

    numInserted += 1
    if numInserted % 500 == 0:
      print('%.1f inserts/sec (%.1f avg)' % (
        500 / (time.time() - tlast),
        numInserted / (time.time() - t0)
      ), len(fens))
      tlast = time.time()
      conn.commit()

def analyzer(fenQueue, resultQueue, args):
    stockfish = Stockfish(path=args.stockpath, depth=args.depth)
    while True:
        fen = fenQueue.get()

        try:
          stockfish.set_fen_position(fen)
          moves = stockfish.get_top_moves(args.multipv)
        except StockfishException:
          print('reboot')
          # Restart stockfish.
          stockfish = Stockfish(path=args.stockpath, depth=args.depth)
          continue

        if len(moves) != args.multipv:
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
              move['Centipawn'] = args.clip
            else:
              move['Centipawn'] = -args.clip
          move['Centipawn'] = max(-args.clip, min(args.clip, move['Centipawn']))

        resultQueue.put({
          "fen": fen,
          "moves": [m['Move'] for m in moves],
          "scores": [m['Centipawn'] for m in moves],
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

def get_table_name(args):
  return f"tmp_d{args.depth}"

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("pgnfiles", nargs='*')
  parser.add_argument("--mode", type=str, required=True, help="{generate, update_features, write_numpy}")
  parser.add_argument("--depth", type=int, default=10, help="{1, 2, ..}")
  parser.add_argument("--clip", type=int, default=5000)
  parser.add_argument("--threads", type=int, default=4)
  parser.add_argument("--multipv", type=int, default=5, help="{1, 2, ..}")
  parser.add_argument("--stockpath", default="/usr/local/bin/stockfish", type=str)
  args = parser.parse_args()

  assert args.mode in ['generate', 'update_features', 'write_numpy']
  assert args.depth > 1

  if args.mode == 'generate':
    # generate work
    fenQueue = Queue()
    resultQueue = Queue()

    analyzers = [Process(target=analyzer, args=(fenQueue, resultQueue, args)) for _ in range(args.threads)]
    for p in analyzers:
      p.start()

    sqlThread = Process(target=sql_inserter, args=(resultQueue, args))
    sqlThread.start()

    iterator = pgn_iterator(noise = 0)

    for fen in iterator:
      fenQueue.put(fen)
