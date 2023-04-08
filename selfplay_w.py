"""
Prints a positive number if weights1.txt is better than weights2.txt

python3 selfplay_w.py weights1.txt weights2.txt
"""

import argparse
import random
import re
import subprocess
import chess
import sys
import time
import numpy as np
from scipy import stats
from multiprocessing import Pool

from tqdm import tqdm

from simple_stockfish import Stockfish

def play_random(board, n):
  if n <= 0:
    return board.fen()
  move = random.choice(list(board.legal_moves))
  board.push(move)
  return play_random(board, n - 1)

# Returns 1 if weights0 wins both games
# Returns -1 if weights1 wins both games
def thread_main(args):
  fen, w0, w1 = args
  command = [
    "./selfplay",
    "weights", w0, w1,
    "maxmoves", "200",
    "fen", *fen.split(' ')
  ]
  stdout = subprocess.check_output(command).decode()
  return sum([int(x) for x in stdout.strip().split('\n')]) / 2.0

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("weights", nargs=2)
  parser.add_argument("--nodes", type=int, default=500)
  parser.add_argument("--num_workers", type=int, default=4)
  parser.add_argument("--num_trials", type=int, default=3200)
  args = parser.parse_args()

  t0 = time.time()
  thread_arguments = [(
    play_random(chess.Board(), 4),
    args.weights[0],
    args.weights[1],
  ) for _ in range(args.num_trials)]
  with Pool(args.num_workers) as p:
    r = list(tqdm(p.imap(thread_main, thread_arguments), total=args.num_trials))
  r = np.array(r, dtype=np.float64).reshape(-1)

  stderr = r.std(ddof=1) / np.sqrt(r.shape[0])
  avg = r.mean()

  dt = time.time() - t0
  if dt < 60:
    print('%.1f secs' % dt)
  else:
    print('%.1f min' % (dt / 60.0))
  print('%.4f ± %.4f' % (avg, stderr))
  print(stats.norm.cdf(avg / stderr))

