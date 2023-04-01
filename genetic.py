"""
Running this gave us a bonus of 0.050 ± 0.021 (p = 0.008) so seems good!
Running 20 iterations on just material weight gave us 0.072 ± 0.019!

# TODO: PCA on variables before mutating?
"""

import argparse
import os
import random
import re
import subprocess
import time

from multiprocessing import Pool

import chess
import numpy as np

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
]

# mediumValueVars = [
#   "OUR_PAWNS",
#   "OUR_KNIGHTS",
#   "OUR_BISHOPS",
#   "OUR_ROOKS",
#   "OUR_QUEENS",
#   "THEIR_PAWNS",
#   "THEIR_KNIGHTS",
#   "THEIR_BISHOPS",
#   "THEIR_ROOKS",
#   "THEIR_QUEENS",
#   "IN_CHECK",
#   "KING_ON_BACK_RANK",
#   "KING_ACTIVE",
#   "THREATS_NEAR_KING_2",
#   "THREATS_NEAR_KING_3",
#   "ADVANCED_PASSED_PAWNS_2",
#   "ADVANCED_PASSED_PAWNS_3",
#   "ADVANCED_PASSED_PAWNS_4",
#   "PAWN_MINOR_CAPTURES",
#   "PAWN_MAJOR_CAPTURES",
#   "BISHOP_PAIR",
#   "BLOCKADED_BISHOPS",
#   "SCARY_BISHOPS",
#   "SCARIER_BISHOPS",
#   "BLOCKADED_ROOKS",
#   "SCARY_ROOKS",
#   "INFILTRATING_ROOKS",
#   "KNIGHT_MAJOR_CAPTURES",
#   "OUR_HANGING_PAWNS",
#   "OUR_HANGING_KNIGHTS",
#   "OUR_HANGING_BISHOPS",
#   "OUR_HANGING_ROOKS",
#   "OUR_HANGING_QUEENS",
#   "THEIR_HANGING_PAWNS",
#   "THEIR_HANGING_KNIGHTS",
#   "THEIR_HANGING_BISHOPS",
#   "THEIR_HANGING_ROOKS",
#   "THEIR_HANGING_QUEENS",
#   "LONELY_KING_IN_CENTER",
#   "LONELY_KING_AWAY_FROM_ENEMY_KING",
#   "PAWN_V_LONELY_KING",
#   "KNIGHTS_V_LONELY_KING",
#   "BISHOPS_V_LONELY_KING",
#   "ROOK_V_LONELY_KING",
#   "QUEEN_V_LONELY_KING",
#   "OUR_MATERIAL_THREATS",
#   "THEIR_MATERIAL_THREATS",
#   "LONELY_KING_ON_EDGE",
#   "OUTPOSTED_KNIGHTS",
#   "OUTPOSTED_BISHOPS",
#   "OUR_HANGING_PAWNS_2",
#   "OUR_HANGING_KNIGHTS_2",
#   "OUR_HANGING_BISHOPS_2",
#   "OUR_HANGING_ROOKS_2",
#   "OUR_HANGING_QUEENS_2",
#   "THEIR_HANGING_PAWNS_2",
#   "THEIR_HANGING_KNIGHTS_2",
#   "THEIR_HANGING_BISHOPS_2",
#   "THEIR_HANGING_ROOKS_2",
#   "THEIR_HANGING_QUEENS_2",
#   "QUEEN_THREATS_NEAR_KING",
#   "MISSING_FIANCHETTO_BISHOP",
#   "BISHOP_PAWN_DISAGREEMENT",
# ]

highValueVars = [
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
]

valid_indices = [varnames.index(v) for v in highValueVars]

# valid_indices = np.arange(len(varnames))

# valid_indices = [
#   varnames.index("PAWN_MOVES"),
#   varnames.index("KNIGHT_MOVES"),
#   varnames.index("BISHOP_MOVES"),
#   varnames.index("ROOK_MOVES"),
#   varnames.index("QUEEN_MOVES"),
#   varnames.index("PAWN_MOVES_ON_THEIR_SIDE"),
#   varnames.index("KNIGHT_MOVES_ON_THEIR_SIDE"),
#   varnames.index("BISHOP_MOVES_ON_THEIR_SIDE"),
#   varnames.index("ROOK_MOVES_ON_THEIR_SIDE"),
#   varnames.index("QUEEN_MOVES_ON_THEIR_SIDE"),
#   varnames.index('NUM_BAD_SQUARES_FOR_PAWNS'),
#   varnames.index('NUM_BAD_SQUARES_FOR_MINORS'),
#   varnames.index('NUM_BAD_SQUARES_FOR_ROOKS'),
#   varnames.index('NUM_BAD_SQUARES_FOR_QUEENS'),
# ]

valid_indices = np.concatenate([
  valid_indices,
  [i + len(varnames) for i in valid_indices],
  [i + len(varnames) * 2 for i in valid_indices],
  # [i + len(varnames) * 3 for i in valid_indices],  # Ignore lonely king
])

def play_random(board, n):
  if n <= 0:
    return board.fen()
  move = random.choice(list(board.legal_moves))
  board.push(move)
  return play_random(board, n - 1)

def analyze(weights_filename, fen, moves):
  command = ["./a.out", "loadweights", weights_filename, "mode", "analyze", "nodes", "10000", "fen", *fen.split(' '), "moves", *moves]
  stdout = subprocess.check_output(command).decode()
  matches = re.findall(r"\d+ : [^ ]+", stdout)
  try:
    return matches[-1].split(' ')[2], command
  except IndexError as e:
    print(command)
    print(' '.join(command))
    print(stdout)
    raise e

def play(fen0, weights_filename1, weights_filename2):
  board = chess.Board(fen0)
  moves = []
  mover, waiter = weights_filename1, weights_filename2
  while not board.can_claim_draw() and not board.is_stalemate() and not board.is_checkmate():
    move, cmd = analyze(mover, fen0, moves)
    moves.append(move)
    if move == 'a8a8':
      break
    try:
      board.push_uci(move)
    except (ValueError) as e:
      print('error', ' '.join(cmd))
      print(fen0, moves)
      print(board)
      print(board.fen())
      print('')
      raise e
    mover, waiter = waiter, mover
    if len(moves) > 250:
      break

  if board.is_checkmate():
    if waiter == weights_filename1:
      return 1
    else:
      return -1
  else:
    return 0

def moves2fen(*moves):
  b = chess.Board()
  for move in moves:
    b.push_uci(move)
  return b.fen()

def load_weights(fn):
  with open(fn, 'r') as f:
    weights = [float(x) for x in re.split(r'\s+', f.read()) if len(x.strip()) > 0]
  weights = np.array(weights)
  assert len(weights) == len(varnames) * 4
  return weights.astype(np.int32)

def save_weights(w, fn):
  w = w.reshape(4, len(varnames)).astype(np.int32)
  with open(fn, 'w+') as f:
    for row in w:
      f.write(' '.join([str(x) for x in row]) + '\n')

def lpad(t, n, c=' '):
  t = str(t)
  return max(n - len(t), 0) * c + t

# Returns  2 if w1.txt wins both games
# Returns -2 if w0.txt wins both games
def thread_main(fen):
  return play(fen, 'w1.txt', 'w0.txt') - play(fen, 'w0.txt', 'w1.txt')

kFoo = ['early', 'late', 'clipped', 'lonelyKing']

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--num_trials", type=int, default=40, help="Number of duplicate-games to play per generation")
  parser.add_argument("--threshold", type=float, default=2.0, help="Number of z-scores required to step in a direction")
  parser.add_argument("--lr", type=float, default=0.1)
  args = parser.parse_args()
  assert args.threshold <= args.num_trials * 2


  os.system("./a.out saveweights w0.txt")

  for it in range(100_000):
    t0 = time.time()
    w0 = load_weights("w0.txt")
    assert len(w0) == len(varnames) * 4

    # Create w1 by mutating one variable.
    w1 = w0.copy()
    varidx = random.choice(valid_indices)
    lr = max(1, round(abs(w1[varidx]) * args.lr))
    step = random.randint(0, 1) * (2 * lr) - lr
    w1[varidx] += step

    save_weights(w1, 'w1.txt')

    fens = [ play_random(chess.Board(), 4) for _ in range(args.num_trials) ]

    with Pool(4) as p:
      r = p.map(thread_main, fens)
    r = np.array(r)

    avg = r.mean()
    stderr = r.std() / np.sqrt(r.shape[0])  # Don't hate me.

    t1 = time.time()

    print('%.1f secs' % (t1 - t0))
    print('%.3f %.3f' % (avg - stderr * args.threshold, avg + stderr * args.threshold))
    if avg >= stderr * args.threshold:
      print(f"iteration {lpad(it, 6)}; score {lpad(avg, 3)}; {varnames[varidx % len(varnames)]}_{kFoo[varidx // len(varnames)]} += {step}")
      os.system(f"mv w1.txt w0.txt")
    else:
      print(f"iteration {lpad(it, 6)}; score {lpad(avg, 3)}; {varnames[varidx % len(varnames)]}_{kFoo[varidx // len(varnames)]} unchanged")

