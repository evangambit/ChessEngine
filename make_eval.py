import argparse
import os
import random
import subprocess
import sqlite3

import chess
from chess import pgn

from pystockfish import Engine

import numpy as np

conn = sqlite3.connect('eval.sqlite3')
c = conn.cursor()
kTableName = 'TrainData'

"""
python3 make_eval.py foo.pgn --stage 1
python3 make_eval.py eval.txt --stage 2
./a.out mode printvec fens eval.txt > train.txt
python3 make_eval.py train.txt traindata
zip traindata.zip -r traindata
"""


parser = argparse.ArgumentParser()
parser.add_argument("files", nargs='+')
parser.add_argument("--stage", type=int)
args = parser.parse_args()

assert args.stage in [1, 2, 3]

if args.stage == 1:
  c.execute(f"""CREATE TABLE IF NOT EXISTS {kTableName} (
    fen BLOB,
    bestMove BLOB,
    score INTEGER,
    numMoves INTEGER
  );""")

  results = {}

  fens = set()
  c.execute(f"SELECT fen FROM {kTableName}")
  for fen, in c:
    fens.add(fen)

  kDepth = 10
  engine = Engine(depth=kDepth)
  engine.setoption('MultiPV', 2)

  totalWrites = 0
  writesSinceCommit = 0

  for filename in args.files:
    f = open(filename, 'r')

    game = pgn.read_game(f)
    while game is not None:
      for node in game.mainline():
        if random.randint(1, 50) != 1:
          continue
        board = node.board()
        fen = board.fen()
        if fen in fens:
          continue

        fens.add(fen)

        engine.setfenposition(fen)
        infos = engine.go_infos()['infos']
        infos = [i for i in infos if i['depth'] == kDepth]
        if len(infos) != 2:
          continue
        if ' b ' in fen:
          for info in infos:
            info['score'] *= -1

        infos.sort(key=lambda i: i['multipv'])

        if abs(infos[0]['score'] - infos[1]['score']) < 20:
          continue

        bestMove = infos[0]['bestmoves'].split(' ')[0]

        c.execute(f"INSERT INTO {kTableName} (fen, bestMove, score, numMoves) VALUES (?, ?, ?, ?)", (
          fen,
          bestMove,
          info['score'],
          len(list(board.legal_moves)),
        ))
        writesSinceCommit += 1
        totalWrites += 1

      if writesSinceCommit >= 40:
        print('commit', totalWrites)
        writesSinceCommit = 0
        conn.commit()

      game = pgn.read_game(f)


if args.stage == 2:
  assert len(args.files) == 1
  c.execute(f"SELECT fen, bestMove, score FROM {kTableName}")
  with open(args.files[0], 'w+') as f:
    for row in c:
      f.write(":".join([str(x) for x in row]) + "\n")

if args.stage == 3:
  def process_position(lines):
    assert lines[0].startswith('FEN ')
    fen = lines[0].split(' ')[1]
    assert lines[1].startswith('SCORE ')
    score = int(lines[1].split(' ')[1])
    x = np.zeros(len(lines) - 2, dtype=np.int8)
    for i, line in enumerate(lines[2:]):
      val = int(line.split(' ')[0])
      assert val >= -127 and val <= 127
      x[i] = val
    return fen, score, x

  fin, dout = args.files

  if not os.path.exists(dout):
    os.mkdir(dout)

  F, Y, X = [], [], []
  f = open(fin, 'r')
  lines = []
  for line in f:
    if line[:4].startswith('FEN ') and len(lines) > 0:
      fen, score, x = process_position(lines)
      F.append(fen)
      Y.append(score)
      X.append(x)
      lines = []
    lines.append(line.strip())

  fen, score, x = process_position(lines)
  F.append(fen)
  Y.append(score)
  X.append(x)

  F = np.array(F)
  Y = np.array(Y)
  X = np.array(X)

  np.save(os.path.join(dout, 'f.npy'), F)
  np.save(os.path.join(dout, 'y.npy'), Y)
  np.save(os.path.join(dout, 'x.npy'), X)
