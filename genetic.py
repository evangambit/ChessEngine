"""
Running this gave us a bonus of 0.050 ± 0.021 (p = 0.008) so seems good!
Running 20 iterations on just material weight gave us 0.072 ± 0.019!

# TODO: PCA on variables before mutating?
"""

import argparse
import copy
import math
import os
import random
import re
import subprocess
import time

import multiprocessing as mp

import chess
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from statsmodels.api import OLS

varnames = [
  "BIAS",
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
  "OUR_PASSED_PAWNS",
  "THEIR_PASSED_PAWNS",
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
  "NUM_TARGET_SQUARES",
  "TIME",
  "KPVK_OPPOSITION",
  "KPVK_IN_FRONT_OF_PAWN",
  "KPVK_OFFENSIVE_KEY_SQUARES",
  "KPVK_DEFENSIVE_KEY_SQUARES",
  "SQUARE_RULE",
  "ADVANCED_PAWNS_1",
  "ADVANCED_PAWNS_2",
  "OPEN_ROOKS",
  "ROOKS_ON_THEIR_SIDE",
  "KING_CASTLED",
  "CASTLING_RIGHTS",
  "KING_IN_FRONT_OF_PASSED_PAWN",
  "KING_IN_FRONT_OF_PASSED_PAWN2",
  "PAWN_V_LONELY_KING",
  "KNIGHTS_V_LONELY_KING",
  "BISHOPS_V_LONELY_KING",
  "ROOK_V_LONELY_KING",
  "QUEEN_V_LONELY_KING",
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
  "BISHOP_PAWN_DISAGREEMENT",
  "CLOSED_1",
  "CLOSED_2",
  "CLOSED_3",
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
]

def play_random(board, n):
  if n <= 0:
    return board.fen()
  move = random.choice(list(board.legal_moves))
  board.push(move)
  return play_random(board, n - 1)

def load_weights(fn):
  with open(fn, 'r') as f:
    lines = f.read().split('\n')
    weights = []
    for line in lines[:-2]:
      weights += [float(x) for x in re.split(r'\s+', line) if len(x.strip()) > 0]
  weights = np.array(weights)
  assert len(weights) == len(varnames) * 4, f"{len(weights)} != {len(varnames)} * 4"
  return weights.astype(np.int32)

def save_weights(w, fn):
  w = np.round(w).reshape(4, len(varnames)).astype(np.int32)
  with open(fn, 'w+') as f:
    for row in w:
      f.write(' '.join([str(x) for x in row]) + '\n')
    f.write("""0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 3 1 3 0 -1 -1 -1 -2 -5 -7 5 -2 7 12 5 -9 4 8 3 1 9 31 25 18 -3 -11 0 -7 -6 7 -3 3 -10 -9 -7 -10 -2 -2 15 5 -6 -6 -11 -15 -9 18 13 11 0 0 0 0 0 0 0 0 -13 -8 -4 1 1 -4 -8 -13 -8 -4 1 6 6 1 -4 -8 -4 1 6 10 10 6 1 -4 1 6 10 15 15 10 6 1 1 6 10 15 15 10 6 1 -4 1 6 10 10 6 1 -4 -8 -4 1 6 6 1 -4 -8 -13 -8 -4 1 1 -4 -8 -13 2 -2 -1 0 -1 -1 -1 0 -5 1 2 -3 0 1 -4 -9 1 1 -4 5 9 7 8 13 0 3 5 4 12 6 7 0 3 2 1 10 7 3 -4 0 -5 10 6 10 15 -4 3 2 0 15 16 4 4 8 9 2 0 16 15 -8 -3 1 0 -6 2 -1 1 2 0 0 1 1 0 4 6 4 4 3 3 2 1 2 2 2 6 6 4 2 -4 -3 1 -2 -1 3 3 1 -3 0 0 0 1 0 4 4 -9 -1 -3 -3 4 4 14 5 -14 -1 -2 -4 -4 4 1 -9 3 6 5 16 21 15 -2 13 -1 -1 0 -1 1 1 0 1 -2 -1 2 1 2 8 2 8 -3 1 0 5 9 13 12 23 -3 -5 1 3 8 9 8 11 0 -2 -3 -2 6 2 5 16 -3 0 3 -3 1 7 17 10 -1 3 9 5 11 12 2 2 -6 -5 1 8 4 -3 -1 -2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 -1 -1 0 0 0 -1 1 -3 -2 0 1 -1 -2 -3 -6 1 -8 5 5 2 -8 2 2 37 38 1 18 -4 -15 16 -9 10 19 0 0 0 0 0 0 0 0 10 16 12 12 5 -25 -22 -18 13 15 7 5 -2 8 -22 -11 11 14 1 9 7 -6 10 -3 -7 1 2 -2 -14 -31 -17 -15 12 1 -2 3 -2 -11 -7 3 -3 -1 -1 1 0 0 2 1 0 0 0 0 0 0 0 0 13 8 4 -1 -1 4 8 13 8 4 -1 -6 -6 -1 4 8 4 -1 -6 -10 -10 -6 -1 4 -1 -6 -10 -15 -15 -10 -6 -1 -1 -6 -10 -15 -15 -10 -6 -1 4 -1 -6 -10 -10 -6 -1 4 8 4 -1 -6 -6 -1 4 8 13 8 4 -1 -1 4 8 13 6 -5 -7 -1 5 -6 0 -2 -1 -9 -10 4 -8 -9 -18 -2 9 -6 1 -7 0 -1 -8 -5 2 3 -2 -6 -13 -2 1 2 5 -8 -2 -8 -4 -6 -4 5 -5 1 4 -1 -4 -5 -5 -10 4 1 3 1 0 2 7 3 -3 2 2 1 1 0 0 0 1 -4 0 -12 -15 -14 6 -11 18 3 4 2 3 -10 -6 8 9 2 7 3 3 1 -11 -2 5 2 2 -2 -2 1 -2 1 2 0 1 -3 -3 -5 -2 -1 0 -3 -4 -2 -3 -3 -2 -2 0 -3 -4 -4 -2 -3 -1 -2 0 1 0 2 0 0 0 -1 8 5 -2 -8 -5 8 0 1 2 -3 -12 2 -9 -8 -2 -1 3 -2 -2 8 0 -1 -12 -5 -4 -3 4 6 -6 -3 -4 -8 1 -6 -2 -4 -3 -11 -3 -20 1 1 -2 -4 -8 -7 -5 -14 4 6 0 -1 0 -6 1 -2 3 2 0 1 2 -1 0 -2 0 -6 9 27 -10 17 -6 -12 -1 -5 2 11 -1 1 -33 -31 1 0 2 0 2 6 5 5 1 0 0 0 0 0 2 3 1 0 0 -1 -1 -1 0 1 0 0 0 0 0 0 -1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 5 3 3 -3 -5 -3 -1 2 14 6 2 -12 -16 1 -4 3 15 5 7 -2 2 -2 -6 1 2 1 -5 4 7 1 -4 -4 -2 -3 2 3 9 8 -7 -10 0 -1 6 -5 13 2 0 -9 0 0 0 0 0 0 0 0 -6 -1 -2 -1 0 -2 -1 -3 -3 1 1 3 2 3 -1 -2 -1 3 -1 0 1 0 4 -2 -2 9 2 3 0 7 19 1 -1 3 8 1 -1 -2 5 2 -4 4 -13 6 8 -10 -8 2 -5 0 1 7 1 4 0 -6 -2 -4 -2 -3 -3 -3 -4 -3 0 -3 0 2 -1 0 -1 0 1 -2 -1 -1 0 1 -1 -3 4 1 -1 -3 0 3 0 4 3 11 1 2 0 2 9 -1 2 0 -1 0 -2 -3 0 -6 -2 -3 2 5 2 2 -6 -1 -3 -14 0 1 3 2 -3 -4 -3 1 -11 2 2 6 -1 -3 2 3 1 1 -1 0 1 0 3 6 5 3 1 2 2 1 3 3 4 -1 0 5 3 0 2 1 3 -1 0 1 1 -1 2 3 5 3 2 2 3 2 3 4 1 1 -3 1 4 -2 2 2 3 -1 -2 -3 -1 -1 0 2 8 -3 -6 3 7 -19 -2 -1 1 -2 2 1 0 -1 0 -2 1 1 3 7 1 3 -2 2 1 2 6 8 5 9 0 3 2 7 8 8 7 12 -1 0 7 17 10 9 9 5 -2 4 8 14 8 14 5 1 -1 3 -8 4 0 2 0 1 0 0 3 -8 4 0 0 -1 0 -1 0 0 0 -1 0 0 -1 0 1 0 0 1 1 0 0 1 1 1 1 4 2 -2 -2 1 4 0 2 5 2 -4 -2 -2 4 1 -3 4 -6 -11 -3 0 4 5 5 13 -1 -10 -3 -3 4 -4 -6 3 -8 -21 -1 0 5 2 -2 21 -3 -16 0 0 0 0 0 0 0 0 -3 -3 -5 6 -3 0 1 13 2 1 0 -1 -4 -10 7 11 -6 3 9 -6 -8 -1 0 4 -11 1 -4 3 1 5 2 -1 -10 -4 1 10 6 -2 -1 -3 -3 -1 -1 2 2 0 1 0 0 0 0 0 0 0 0 0 3 9 3 0 4 2 -6 2 3 2 -2 -3 -5 -4 2 3 2 -4 6 -4 -4 4 -3 -1 4 -4 -9 2 -2 -6 -5 3 4 -13 -3 -3 1 -4 -15 -4 0 -2 3 0 0 -1 -3 3 3 3 0 -1 -1 0 1 3 5 1 1 0 1 0 1 3 5 1 11 -2 0 -5 0 2 3 17 -3 -2 -6 -1 4 3 3 -4 -3 -8 -12 0 4 -3 -1 -1 -5 -2 -6 1 0 2 -1 -12 -3 -3 -2 -3 -1 0 -1 0 1 1 0 -3 -2 -2 0 -3 -1 1 1 0 4 0 0 1 1 2 0 0 -1 0 -4 -3 -9 -1 2 -4 -11 18 -7 0 -1 1 2 4 0 3 -2 -3 -2 -3 3 2 -4 2 -5 -2 -5 -5 -3 -4 -2 0 -2 -3 -1 -3 1 -2 -3 0 -2 -1 -3 0 2 -2 -1 0 -1 -8 -6 -5 0 -2 -1 1 0 2 1 0 0 0 -2 0 2 1 -3 4 -1 2 0 0 3 -2 10 -12 -7 -1 0 0 3 0 -10 -16 -10 -15 -6 -3 13 -4 -6 -15 -9 -6 -5 -4 1 -7 -2 -5 -6 -8 -4 -10 -2 -1 -2 -3 -7 -7 -2 -6 2 3 -2 -3 -1 -5 0 -1 3 2 0 0 1 -1 0 -1 3 1 -7 -5 5 -23 1 16 5 4 -4 3 4 -5 6 18 4 -1 -7 -5 -8 -13 -1 6 4 2 -1 0 1 -6 5 8 0 0 -3 -3 -3 -5 -1 5 0 -1 -2 0 -1 -3 -2 1 0 -1 0 1 0 -1 -1 0 0 1 1 1 0 0 0 0""")

def lpad(t, n, c=' '):
  t = str(t)
  return max(n - len(t), 0) * c + t

# Returns  2 if w0.txt wins both games
# Returns -2 if w1.txt wins both games
def thread_main(fens):
  command = [
    "./selfplay",
    "weights", "w1.txt", "w0.txt",
    "nodes", "500",
    "maxmoves", "200",
  ]
  for fen in fens:
    command.append("fen")
    command += fen.split(' ')
  stdout = subprocess.check_output(command).decode()
  return [int(x) for x in stdout.strip().split('\n')]

kFoo = ['early', 'late', 'clipped', 'lonelyKing']

# The basic approach is to run a command to optimize some variables (e.g. see below) and then run
#
# $ python3 selfplay_w.py w0.txt weights.txt
#
# If the results are positive and significant run
#
# $ mv w0.txt weights.txt
#
# And commit the new weights.txt
#
# If the results are negative, it's possible the code below misinterpretted its data. You can view graphs of
# the data in optimages/... and make manual adjustments if necessary.
#
# Already run:
#
# 0.125 ± 0.029
# $ python3 genetic.py --num_trials 4096 --range 20 --step 5 --stage 0 \
# --variables OUR_PAWNS,THEIR_PAWNS,OUR_KNIGHTS,THEIR_KNIGHTS,OUR_BISHOPS,THEIR_BISHOPS,OUR_ROOKS,THEIR_ROOKS,OUR_QUEENS,THEIR_QUEENS
#
# 0.023 ± 0.009
# $ python3 genetic.py --num_trials 4096 --range 20 --step 5 --stage 1 \
# --variables OUR_PAWNS,THEIR_PAWNS,OUR_KNIGHTS,THEIR_KNIGHTS,OUR_BISHOPS,THEIR_BISHOPS,OUR_ROOKS,THEIR_ROOKS,OUR_QUEENS,THEIR_QUEENS
#
# 0.004 ± 0.002 (95.8%)
# $ python3 genetic.py --num_trials 4096 --range 10 --step 2 --stage 1 \
# --variables KING_ON_BACK_RANK,KING_ON_CENTER_FILE,KING_ACTIVE,THREATS_NEAR_KING_2,THREATS_NEAR_KING_3
#
# 
# 0.0022 ± 0.0111 (57.8%)
# $ python3 genetic.py --num_trials 4096 --range 10 --step 2 --stage 0 \
# --variables KING_ON_BACK_RANK,KING_ON_CENTER_FILE,KING_ACTIVE,THREATS_NEAR_KING_2,THREATS_NEAR_KING_3
#
# 0.0193 ± 0.0114 (95.5%)
# $ python3 genetic.py --num_trials 4096 --range 10 --step 2 --stage 2 \
# --variables ISOLATED_PAWNS,DOUBLED_PAWNS,DOUBLE_ISOLATED_PAWNS,PAWNS_CENTER_16,PAWNS_CENTER_4,ADVANCED_PASSED_PAWNS_2,ADVANCED_PASSED_PAWNS_3,ADVANCED_PASSED_PAWNS_4
#
# $ python3 -i genetic.py --num_trials 40000 --range 120 --stage 2 \
# --variables ROOK_CHECKS,QUEEN_CHECKS,IN_DOUBLE_CHECK,IN_TRIVIAL_CHECK

"""
N = (2 * range + 1) / (step * trialsPerPoint * duplicates)

step = range
duplicates = 1
trialsPerPoint = 256

while step > 1 and (2 * range + 1) / (step * trialsPerPoint * duplicates) < N:
  step -= 1

while (2 * range + 1) / (step * trialsPerPoint * duplicates) < N:
  duplicates += 1


"""

def dot(*A):
  return np.linalg.multi_dot(A)

def is_scalar(x):
  return isinstance(x, (np.floating, np.integer, float, int))

def linear_regression(X, Y, prior, noise):
  """
  Computes the mean and variance of "W" given the model "Y = X W + noise"

  Runs in O(n^3) time

  See page 14 of https://arxiv.org/pdf/1910.03558.pdf

      X: A n-by-d matrix of features
      Y: A n-by-1 matrix of labels
  prior: Either a d-long vector representing the variance of your prior over each
         weight, or a d-by-d covariance matrix representing your prior over the
         weights
  noise: Either a n-long vector representing the variance of the noise corrupting
         each label (assumed to be independent) or an n-by-n matrix representing
         the covariance matrix of the noise
   What: The maximum likelihood estimate for W
   Wvar: The covariance matrix of the posterior over What
  """
  n, d = X.shape
  if is_scalar(prior):
    prior = np.ones(d) * prior
  if is_scalar(noise):
    noise = np.ones(n) * noise
  if len(prior.shape) == 1:
    prior = np.diag(prior)
  if len(noise.shape) == 1:
    noise = np.diag(noise)
  if len(Y.shape) == 1:
    Y = Y.reshape((n, 1))
  assert prior.shape == (d, d)
  assert noise.shape == (n, n)
  assert Y.shape == (n, 1)
  tmp = np.linalg.inv(dot(X, prior, X.T) + noise)
  What = dot(prior, X.T, tmp, Y)
  Wvar = prior - dot(prior, X.T, tmp, X, prior)
  return What, Wvar

if __name__ == '__main__':
  mp.set_start_method('spawn')
  parser = argparse.ArgumentParser()
  parser.add_argument("--num_trials", type=int, default=8000, help="Number of duplicate-games to play per generation")
  parser.add_argument("--num_workers", type=int, default=4, help="Number of processes to use")
  parser.add_argument("--variables", type=str, default='')
  parser.add_argument("--timeout", type=float, default=1.0)
  parser.add_argument("--range", type=int, default=5)
  parser.add_argument("--trials_per_point", type=int, default=1000)
  parser.add_argument("--stage", type=int, required=True, help="Integer indicating weight type (early,late,clipped,lonelyKing)")
  args = parser.parse_args()
  assert args.stage >= 0 and args.stage < 4
  if args.variables == '':
    args.variables = ','.join(varnames)
  args.variables = args.variables.split(',')

  kTimeout = args.trials_per_point / args.num_workers / 5 * args.timeout

  kNumSteps = 2
  numDuplicates = 1

  while kNumSteps < 2 * args.range + 1 and args.trials_per_point * kNumSteps * numDuplicates < args.num_trials:
    kNumSteps += 1

  deltas = np.concatenate([np.linspace(-args.range, args.range, kNumSteps).astype(np.int32)])
  deltas = set(deltas)
  if 0 in deltas:
    deltas.remove(0)
  deltas = np.array(list(deltas), dtype=np.int64)

  while args.trials_per_point * len(deltas) * (numDuplicates + 0.5) < args.num_trials:
    numDuplicates += 1

  deltas = np.concatenate([deltas] * numDuplicates)

  print(f"steps = {len(deltas)}; duplicates = {numDuplicates}; trials = {args.trials_per_point * kNumSteps * numDuplicates}; timeout = %.2f secs" % kTimeout)

  for vn in args.variables:
    assert vn in varnames

  os.system('cp weights.txt w0.txt')
  w0 = load_weights("w0.txt").astype(np.float64)
  assert len(w0) == len(varnames) * 4

  for varname in args.variables:
    varidx = varnames.index(varname) + len(varnames) * args.stage

    save_weights(w0, 'w0.txt')

    px = [0.0]
    py = [0.0]
    pz = [0.0]

    for stepsize in tqdm(deltas):
      w1 = w0.copy()
      w1[varidx] += stepsize

      save_weights(w1, 'w1.txt')

      fens = [ play_random(chess.Board(), 4) for _ in range(args.trials_per_point) ]

      # We split fens into equal-sized lists (one for each worker).
      tmp = [[] for _ in range(args.num_workers)]
      for i, fen in enumerate(fens):
        tmp[i % args.num_workers].append(fen)
      fens = tmp

      t0 = time.time()
      with mp.Pool(processes=args.num_workers) as pool:
        # r = p.map(thread_main, fens)
        res = pool.map_async(thread_main, fens)
        try:
          r = res.get(timeout=kTimeout)
        except mp.context.TimeoutError:
          print('timeout')
          continue
      t1 = time.time()

      # flatten
      r = sum(r, [])
      r = np.array(r) / 2

      avg = r.mean()
      stderr = r.std() / np.sqrt(r.shape[0])  # Don't hate me.

      px.append(stepsize)
      py.append(avg)
      pz.append(stderr)

    px = np.array(px)
    py = np.array(py)
    pz = np.array(pz)

    # Since we know f(0) = 0, we can omit the contant term from this quadratic regression.
    X = np.zeros((px.shape[0], 2))
    X[:,0] = px
    X[:,1] = px * px

    model = OLS(py, X)
    results = model.fit()

    w = results.params
    stderr = results.bse

    print(w - stderr)
    print(w + stderr)

    slopeAt0 = w[0]
    maximum = -w[0] / (2 * w[1]) if w[1] < 0 else None
    if maximum is not None and maximum > -args.range and maximum < args.range:
      delta = max(-args.range, min(args.range, round(maximum)))
      scoreIncrease = w[0] * delta + w[1] * delta * delta
    else:
      # Linear regression
      w = float(np.linalg.lstsq(X[:,0:1], py, rcond=-1)[0])
      if w > 0:
        delta = round(np.percentile(deltas, 75))
      else:
        delta = round(np.percentile(deltas, 25))
      scoreIncrease = w * delta
    if scoreIncrease > 0:
      print(f"w0[{varname}] = {w0[varidx]} + {delta}; increases points by %.4f" % scoreIncrease)
    else:
      print(f'no improvement found for "{varname}"')
      delta = None
      scoreIncrease = None


    plt.figure()
    plt.scatter(px, py)
    curvex = np.array(list(sorted(set(px))))
    if isinstance(w, float):
      plt.plot(curvex, curvex * w)
    else:
      plt.plot(curvex, curvex * w[0] + curvex * curvex * w[1])

    plt.grid()
    if not os.path.exists("optimages"):
      os.mkdir("optimages")
    plt.savefig(os.path.join("optimages", f"{varname}_{args.stage}.png"))

    if delta is not None:
      w0[varidx] += delta


