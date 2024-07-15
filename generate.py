import argparse
import asyncio
import random
import re
import sqlite3
import os
import subprocess
import time

from collections import defaultdict
from multiprocessing import Process, Queue

import chess
from chess import engine as chess_engine

def wdl2score(wdl):
  return (wdl.wins + wdl.draws * 0.5) / (wdl.wins + wdl.draws + wdl.losses)

def score2float(score):
  wdl = score.white().wdl()
  return (wdl.wins + wdl.draws * 0.5) / (wdl.wins + wdl.draws + wdl.losses)

def analyzer(resultQueue, args):
  engine = chess_engine.SimpleEngine.popen_uci(args.engine)
  while True:
    helper(engine, resultQueue, args)

def helper(engine, resultQueue, args):
  board = chess.Board()
  t = ''
  while not board.is_game_over() and not board.is_repetition() and board.ply() < 100:
    lines = engine.analyse(board, chess_engine.Limit(depth=args.depth), multipv=args.multipv)

    if len(lines) < 3:
      board.push(lines[0]['pv'][0])
      continue

    for line in lines:
      wdl = line['score'].white().wdl()
      moves = line['pv']
      if len(moves) < 3:
        continue
      b = chess.Board(board.fen())
      b.push(moves[0])
      is_quiet = False
      for i in range(1, len(moves) - args.min_depth):
        if b.piece_at(moves[i].to_square) is not None:
          is_quiet = True
          break
      if is_quiet:
        resultQueue.put((b.fen(), wdl.wins, wdl.draws, wdl.losses))
      b = None

    # wdl = lines[0]['score'].white().wdl()
    # if wdl.wins > 995 or wdl.losses > 995:
    #   break

    # Drop blunders
    lines = [l for l in lines if abs(score2float(l['score']) - score2float(lines[0]['score'])) < 0.1]

    if score2float(lines[0]['score']) < 0.25 and board.turn:
      # If white is losing, make the best move
      line = lines[0]
    elif score2float(lines[0]['score']) > 0.75 and not board.turn:
      # If black is losing, make the best move
      line = lines[0]
    else:
      # Pick a random move (biased towards better moves)
      L = []
      for i, line in enumerate(lines[::-1]):
        L += [line] * (i + 1)
      line = random.choice(L)
    board.push(line['pv'][0])

def sql_inserter(resultQueue, args, database):
  conn = sqlite3.connect(database)
  c = conn.cursor()
  c.execute('CREATE TABLE IF NOT EXISTS positions (fen TEXT PRIMARY KEY, wins INTEGER, draws INTEGER, losses INTEGER)')
  conn.commit()

  c.execute('SELECT COUNT(1) FROM positions')
  n = c.fetchone()[0]

  t = time.time()

  while True:
    fen, wins, draws, losses = resultQueue.get()
    c.execute('INSERT OR IGNORE INTO positions VALUES (?, ?, ?, ?)', (fen, wins, draws, losses))
    n += 1
    if n % 500 == 499:
      print(f'Inserted {("%.4f" % ((n + 1) / 1_000_000)).rjust(7)}M positions ({str(int(500 / (time.time() - t))).rjust(4)} / sec)')
      t = time.time()
      conn.commit()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--engine', default='/opt/homebrew/bin/stockfish')
  parser.add_argument('--depth', type=int, default=8)
  parser.add_argument('--multipv', type=int, default=5)
  parser.add_argument('--num_workers', type=int, default=4)
  parser.add_argument('--min_depth', type=int, default=3)
  args = parser.parse_args()

  database = os.path.join('data', f'de{args.depth}-md{args.min_depth}', f'db.sqlite3')
  if not os.path.exists(os.path.dirname(database)):
    os.makedirs(os.path.dirname(database))

  resultQueue = Queue()

  analyzers = [Process(target=analyzer, args=(resultQueue, args)) for _ in range(args.num_workers)]
  for p in analyzers:
    p.start()
  
  sql_inserter(resultQueue, args, database)


