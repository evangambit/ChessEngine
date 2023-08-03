import time
import argparse
import os
import random
import re
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

class UciPlayer:
  def __init__(self, path, weights):
    self.name = (path, weights)
    self._p = subprocess.Popen(path, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    self.allin = []
    self.allout = []
    self.command("uci")
    if weights.lower() != 'none':
      self.command(f"loadweights {weights}")

  def __del__(self):
    self._p.terminate()

  def command(self, text):
    self.allin.append(text)
    self._p.stdin.write((text + '\n').encode())
    self._p.stdin.flush()

  def go(self, fen, depth):
    self.command(f"position fen {fen}")
    self.command(f"go depth {depth}")
    lines = []
    while True:
      line = self._p.stdout.readline().decode()
      self.allout.append(line)
      if line == '':
        print('lines', repr(lines))
        for l in self.allin:
          print(l)
        print('====' * 9)
        for l in self.allout:
          print(repr(l))
        raise RuntimeError('empty line')
      lines.append(line.strip())
      if line.startswith('bestmove '):
        break

    assert 'bestmove ' in lines[-1] # e.g. "bestmove h6h7 ponder a2a3"
    lines = [line for line in lines if re.match(rf"info depth {depth}.+", line)]

    R = []
    for line in lines:
      r = {
        "depth": re.findall(r" depth (\d+)", line)[0],
        "pv": re.findall(r" pv (.+)", line)[0],
        "score": re.findall(r"score (\S+ -?\d+)", line)[0],
        "multipv": re.findall(r"multipv (\d+)", line)[0],
        "nodes": re.findall(r"nodes (\d+)", line)[0],
        "time": re.findall(r"time (\d+)", line)[0],
      }
      R.append(r)

    return R

def analyzer(fenQueue, resultQueue, args):
    stockfish = Stockfish(path=args.stockpath, depth=args.stockdepth)
    ourEngine = UciPlayer("./new", "weights.txt")
    ourEngine.command("setoption name MultiPV value 2")

    while True:
        fen = fenQueue.get()

        try:
          stockfish.set_fen_position(fen)
          moves = stockfish.get_top_moves(args.multipv)
        except StockfishException:
          print('reboot')
          # Restart stockfish.
          stockfish = Stockfish(path=args.stockpath, depth=args.stockdepth)
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

        ourEval = ourEngine.go(fen, depth=args.depth)
        assert len(ourEval) == 2, ourEval

        if 'mate' in ourEval[0]['score']:
          continue
        if 'mate' in ourEval[1]['score']:
          continue

        score1 = int(ourEval[0]['score'].split(' ')[1])
        score2 = int(ourEval[1]['score'].split(' ')[1])

        if abs(score1 - score2) > args.margin:
          continue

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
  return f"tmp_sd{args.stockdepth}_d{args.depth}_margin{args.margin}"

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("pgnfiles", nargs='*')
  parser.add_argument("--stockdepth", type=int, default=10, help="{1, 2, ..}")
  parser.add_argument("--depth", type=int, default=1, help="{1, 2, ..}")
  parser.add_argument("--clip", type=int, default=5000)
  parser.add_argument("--threads", type=int, default=4)
  parser.add_argument("--multipv", type=int, default=5, help="{1, 2, ..}")
  parser.add_argument("--margin", type=int, default=50, help="We ignore any positions where the difference between our engine's first and second best moves are farther than --margin. The idea is that if the difference is larger than this, then any small change will never affect which move the engine chooses, so the position is not suitable for finetuning.")
  parser.add_argument("--stockpath", default="/usr/local/bin/stockfish", type=str)
  args = parser.parse_args()

  assert args.stockdepth > 1
  assert args.depth > 1

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
