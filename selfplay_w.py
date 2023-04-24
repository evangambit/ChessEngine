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
import multiprocessing as mp

from tqdm import tqdm

def play_random(board, n):
  if n <= 0:
    return board.fen()
  move = random.choice(list(board.legal_moves))
  board.push(move)
  return play_random(board, n - 1)

# Returns 1 if weights0 wins both games
# Returns -1 if weights1 wins both games
def thread_main(args):
  fen, w0, w1, nodes = args
  command = [
    "./selfplay",
    "weights", w0, w1,
    "maxmoves", "200",
    "nodes", str(nodes),
    "fen", *fen.split(' '),
  ]
  stdout = subprocess.check_output(command).decode()
  return sum([int(x) for x in stdout.strip().split('\n')]) / 2.0

def create_fen_batch(args):
  return [(
    play_random(chess.Board(), 4),
    args.weights[0],
    args.weights[1],
    args.nodes,
  ) for _ in range(args.num_workers)]

if __name__ == '__main__':
  mp.set_start_method('spawn')
  parser = argparse.ArgumentParser()
  parser.add_argument("weights", nargs=2)
  parser.add_argument("--nodes", type=int, default=10000)
  parser.add_argument("--num_workers", type=int, default=4)
  parser.add_argument("--timeout", type=float, default=30.0)
  parser.add_argument("--num_trials", type=int, default=2000)
  args = parser.parse_args()

  batches = [[]]
  for i in range(0, args.num_trials // args.num_workers, args.num_workers):
    batches.append(create_fen_batch(args))

  t0 = time.time()

  R = []
  with mp.Pool(args.num_workers) as pool:
    for batch in tqdm(batches):
      try:
        r = pool.map_async(thread_main, batch).get(timeout=args.timeout)
        R += r
      except mp.context.TimeoutError:
        print('timeout')
        pass
  r = np.array(R, dtype=np.float64).reshape(-1)

  t1 = time.time()

  stderr = r.std(ddof=1) / np.sqrt(r.shape[0])
  avg = r.mean()

  print("%.1f secs" % (t1 - t0))
  print('%.4f ± %.4f' % (avg, stderr))
  print(stats.norm.cdf(avg / stderr))

