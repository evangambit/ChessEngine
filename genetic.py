"""
Running this gave us a bonus of 0.050 ± 0.021 (p = 0.008) so seems good!
Running 20 iterations on just material weight gave us 0.072 ± 0.019!

# TODO: PCA on variables before mutating?
"""

import subprocess
import os
import random
import re

import chess
import numpy as np

varnames = [
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

def analyze(player, fen, moves):
  command = [player, "mode", "analyze", "nodes", "10000", "fen", *fen.split(' '), "moves", *moves]
  # command = [player, "mode", "analyze", "time", "10", "fen", *fen.split(' '), "moves", *moves]
  stdout = subprocess.check_output(command).decode()
  matches = re.findall(r"\d+ : [^ ]+", stdout)
  try:
    return matches[-1].split(' ')[2], command
  except IndexError as e:
    print(command)
    print(' '.join(command))
    print(stdout)
    raise e

def play(fen0, player1, player2):
  board = chess.Board(fen0)
  moves = []
  mover, waiter = player1, player2
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
    if waiter == player1:
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

kFoo = ['early', 'late', 'clipped', 'lonelyKing']

os.system("sh build.sh ; mv ./a.out ./old")

with open(os.path.join('src', 'Evaluator.h'), 'r') as f:
  text = f.read()

i = text.index("const int32_t kEarlyB0 =")
j = text.index("template<Color US>\nstruct Threats")

prefix = text[:i]
suffix = text[j:]

feature_text_with_placeholders = re.sub(r"-?\d+,", "$$,", text[i:j])

variables0 = np.array([int(x[:-1]) for x in re.findall(r"-?\d+,", text[i:j])], dtype=np.float64)

assert len(variables0) == len(varnames) * 4

for _ in range(10_000):
  # variables1 = variables0 + np.random.normal(0, 1, variables0.shape)
  variables1 = variables0.copy()

  # Mutate one variable.
  varidx = random.choice(valid_indices)
  lr = abs(variables1[varidx]) // 10 + 1
  variables1[varidx] += random.randint(0, 1) * (2 * lr) - lr

  t = feature_text_with_placeholders
  for v in variables1:
    idx = t.index('$$')
    t = t[:idx] + str(int(round(v))) + t[idx+2:]

  with open(os.path.join('src', 'Evaluator.h'), 'w') as f:
    f.write(prefix + t + suffix)

  try:
    os.system("sh build.sh ; mv ./a.out ./new")
  except KeyboardInterrupt:
    exit(0)

  fens = [ play_random(chess.Board(), 4) for _ in range(40) ]

  r = 0
  for fen in fens:
    r += play(fen, './new', './old')
    r -= play(fen, './old', './new')

  print(r, f'"{varnames[varidx % len(varnames)]} ({kFoo[varidx // len(varnames)]})" -> {variables0[varidx]} to {variables1[varidx]}')
  if r >= 5:
    variables0 = variables1.copy()
    os.system("mv ./new ./old")
    os.system(f"cp {os.path.join('src', 'Evaluator.h')} Evaluator.h.best")



