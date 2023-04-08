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
]

def play_random(board, n):
  if n <= 0:
    return board.fen()
  move = random.choice(list(board.legal_moves))
  board.push(move)
  return play_random(board, n - 1)

def load_weights(fn):
  with open(fn, 'r') as f:
    weights = [float(x) for x in re.split(r'\s+', f.read()) if len(x.strip()) > 0]
  weights = np.array(weights)
  assert len(weights) == len(varnames) * 4
  return weights.astype(np.int32)

def save_weights(w, fn):
  w = np.round(w).reshape(4, len(varnames)).astype(np.int32)
  with open(fn, 'w+') as f:
    for row in w:
      f.write(' '.join([str(x) for x in row]) + '\n')

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

if __name__ == '__main__':
  mp.set_start_method('spawn')
  parser = argparse.ArgumentParser()
  parser.add_argument("--num_trials", type=int, default=4096, help="Number of duplicate-games to play per generation")
  parser.add_argument("--num_workers", type=int, default=4, help="Number of processes to use")
  parser.add_argument("--variables", type=str, default='')
  parser.add_argument("--timeout", type=float, default=1.0)
  parser.add_argument("--range", type=int, default=5)
  parser.add_argument("--trials_per_point", type=int, default=2000)
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

  while args.trials_per_point * len(deltas) * numDuplicates < args.num_trials:
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

    # Since we know f(0) = 0, we can omit the contant term from this quadratic regression.
    X = np.zeros((px.shape[0], 2))
    X[:,0] = px
    X[:,1] = px * px
    w = np.linalg.lstsq(X, py, rcond=-1)[0]
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


