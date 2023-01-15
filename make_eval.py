import argparse
import random
import subprocess
import sqlite3

import chess
from chess import pgn

from pystockfish import Engine

import numpy as np

import gmpy2

conn = sqlite3.connect('eval.sqlite3')
c = conn.cursor()
kTableName = 'TrainData'

if False:
  c.execute(f"""CREATE TABLE IF NOT EXISTS {kTableName} (
    fen BLOB,
    bestMove BLOB,
    score INTEGER,
    numMoves INTEGER
  );""")

  results = {}

  parser = argparse.ArgumentParser()
  parser.add_argument("pgns", nargs='+')
  args = parser.parse_args()

  fens = set()
  c.execute(f"SELECT fen FROM {kTableName}")
  for fen, in c:
    fens.add(fen)

  kDepth = 10
  engine = Engine(depth=kDepth)
  engine.setoption('MultiPV', 2)

  totalWrites = 0
  writesSinceCommit = 0

  for filename in args.pgns:
    f = open(filename, 'r')

    game = pgn.read_game(f)
    while game is not None:
      for node in game.mainline():
        if random.randint(1, 20) != 1:
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

        bestMove = info['bestmoves'].split(' ')[0]

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


c.execute(f"SELECT fen, bestMove FROM {kTableName}")
with open('eval2.txt', 'w+') as f:
  for fen, bestMove in c:
    f.write(fen + ":" + bestMove + "\n")


