import math
import random
import re
import subprocess
import chess
import sys
import time

import multiprocessing as mp

from tqdm import tqdm

from scipy import stats
import numpy as np

"""
50,000 nodes/move

stockfish nodes, score, n
  500,  0.2178 ± 0.0155, 256
 1000, -0.0488 ± 0.0164, 256
 1500, -0.1973 ± 0.0154, 256
 2000, -0.2715 ± 0.0145, 256
 2500, -0.3291 ± 0.0126, 256
 3000, -0.3652 ± 0.0119, 256
 3500, -0.3721 ± 0.0102, 256
 4000, -0.3994 ± 0.0097, 256
 4500, -0.3955 ± 0.0096, 256
 9000, -0.4336 ± 0.0079, 256
18000, -0.4609 ± 0.0062, 256
"""

class UciPlayer:
  def __init__(self, path, weights):
    self.name = (path, weights)
    self._p = subprocess.Popen(path, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    self.io = []
    self.command("uci")
    if weights.lower() != 'none':
      if weights.endswith('.bin'):
        self.command(f"loadnnue {weights}")
      else:
        self.command(f"loadweights {weights}")

  def __del__(self):
    self._p.terminate()

  def command(self, text):
    self.io.append(">> " + text)
    self._p.stdin.write((text + '\n').encode())
    self._p.stdin.flush()

  def best_move(self, fen, nodes, moves = [], from_whites_perspective = True):
    if len(moves) == 0:
      self.command(f"position fen {fen}")
    else:
      self.command(f"position fen {fen} moves {' '.join(moves)}")
    if 'stockfish' not in self.name[0]:
      self.command(f"go nodes {int(nodes)}")
    else:
      self.command(f"go nodes {self.name[1]}")
    lines = []
    while True:
      line = self._p.stdout.readline().decode()
      self.io.append(line)
      if line == '':
        print('lines', repr(lines))
        for l in self.io:
          if l.startswith('>> '):
            print(l)
        print('====' * 9)
        raise RuntimeError('empty line')
      lines.append(line.strip())
      if line.startswith('bestmove '):
        break

    assert 'bestmove ' in lines[-1] # e.g. "bestmove h6h7 ponder a2a3"
    return lines[-1].split(' ')[1]

def play(fen0, player1, player2, nodes = 50_000):
  player1.command("setoption name Clear Hash")
  player2.command("setoption name Clear Hash")
  isPlayer1White = ' w ' in fen0
  board = chess.Board(fen0)
  moves = []
  mover, waiter = player1, player2
  while not board.can_claim_draw() and not board.is_stalemate() and not board.is_checkmate() and not board.is_insufficient_material():
    try:
      move = mover.best_move(fen0, nodes, moves)
    except Exception as e:
      print('a')
      print('isPlayer1', mover == player1)
      print(fen0)
      print(' '.join(moves))
      print(board.fen())
      raise e
    moves.append(move)
    if move == 'a8a8':
      print('b', board.fen())
      break
    try:
      board.push_uci(move)
    except (ValueError) as e:
      print('=====')
      print(player1.name, player2.name)
      print(mover.name)
      print('c', board.fen(), move)
      print(fen0 + ' ' + ' '.join(moves))
      print('=====')
      raise e
    mover, waiter = waiter, mover
    if len(moves) > 250:
      print('d', board.fen(), move)
      print('isPlayer1', mover == player1)
      break

  # if isPlayer1White:
  #   print(player1.name, player2.name)
  # else:
  #   print(player2.name, player1.name)
  # print(fen0 + ' ' + ' '.join(moves))

  if board.is_checkmate():
    if waiter == player1:
      return 0.5
    else:
      return -0.5
  return 0

def play_random(board, n):
  if n <= 0:
    return board.fen()
  move = random.choice(list(board.legal_moves))
  board.push(move)
  return play_random(board, n - 1)

def thread_main(fen):
  player1 = UciPlayer(sys.argv[1], sys.argv[2])
  player2 = UciPlayer(sys.argv[3], sys.argv[4])
  try:
    a = play(fen, player1, player2)  # Returns {-0.5, 0, 0.5}
    b = play(fen, player2, player1)  # Returns {-0.5, 0, 0.5}
  except Exception as e:
    print('!!!!    ERROR    !!!!')
    print(e)
    return 0
  return (a - b) / 2.0

def create_fen_batch(n):
  return [play_random(chess.Board(), 4) for _ in range(n)]

"""
winrate = 1 / (1 + 10**(-x/400))

1 / winrate = 1 + 10**(-x/400)
1 / winrate - 1 = 10**(-x/400)
log(1 / winrate - 1) / log(10) = x / -400

x = log(1 / winrate - 1) / log(10) * -400

>>> math.log(1 / (0.5 + (-0.0456 + 0.0124)) - 1.0) / math.log(10) * -400
-23.103717347395612
>>> math.log(1 / (0.5 + (-0.0456 - 0.0124)) - 1.0) / math.log(10) * -400
-40.48477183539473

"""

def int2str(x):
  x = str(x)
  if x[0] != '-':
    return '+' + x
  return x

if __name__ == '__main__':
  mp.set_start_method('spawn')
  t0 = time.time()
  numWorkers = 8
  batches = []
  for i in range(0, 512, numWorkers):
    batches.append(create_fen_batch(numWorkers))

  R = []
  with mp.Pool(numWorkers) as pool:
    for batch in tqdm(batches):
      try:
        r = pool.map_async(thread_main, batch).get(timeout=120)
        R += r
        r = np.array(R, dtype=np.float64).reshape(-1)
        stderr = r.std(ddof=1) / np.sqrt(r.shape[0])
        avg = r.mean()
        print('%.3f ± %.3f' % (avg, stderr))
        try:
          a = math.log(1 / (0.5 + (avg - stderr)) - 1.0) / math.log(10) * -400
          b = math.log(1 / (0.5 + (avg + stderr)) - 1.0) / math.log(10) * -400
          # print(f'ELO_STDERR({int2str(int(a))}, {int2str(int(b))})')
          print('ELO_STDERR(%.1f %.1f)' % (a, b))
        except ValueError:
          pass
        print(
          ('%d' % (r == 0.5).sum()).rjust(4),
          ('%d' % (r == 0.25).sum()).rjust(4),
          ('%d' % (r == 0).sum()).rjust(4),
          ('%d' % (r == -0.25).sum()).rjust(4),
          ('%d' % (r == -0.5).sum()).rjust(4),
        )
      except mp.context.TimeoutError:
        print('timeout')
        pass

  dt = time.time() - t0
  if dt < 60:
    print('%.1f secs' % dt)
  else:
    print('%.1f min' % (dt / 60.0))
  print('%.4f ± %.4f' % (avg, stderr))
  print(stats.norm.cdf(avg / stderr))

