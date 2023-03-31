"""
Running this gave us a bonus of 0.050 ± 0.021 (p = 0.008) so seems good!
"""

import subprocess
import os
import random
import re

import chess
import numpy as np

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
			return 0.5
		else:
			return -0.5
	else:
		return 0

def moves2fen(*moves):
	b = chess.Board()
	for move in moves:
		b.push_uci(move)
	return b.fen()

os.system("sh build.sh ; mv ./a.out ./old")

with open(os.path.join('src', 'Evaluator.h'), 'r') as f:
	text = f.read()

i = text.index("const int32_t kEarlyB0 =")
j = text.index("std::string EFSTR[]")

prefix = text[:i]
suffix = text[j:]

feature_text_with_placeholders = re.sub(r"-?\d+,", "$$,", text[i:j])

variables0 = np.array([int(x[:-1]) for x in re.findall(r"-?\d+,", text[i:j])], dtype=np.float64)

for _ in range(100):
	# variables1 = variables0 + np.random.normal(0, 1, variables0.shape)
	variables1 = variables0.copy()
	variables1[np.random.randint(0, len(variables1))] += np.random.normal(0, 1) * 4

	t = feature_text_with_placeholders
	for v in variables1:
		idx = t.index('$$')
		t = t[:idx] + str(int(round(v))) + t[idx+2:]

	with open(os.path.join('src', 'Evaluator.h'), 'w') as f:
		f.write(prefix + t + suffix)

	os.system("sh build.sh ; mv ./a.out ./new")

	fens = [ play_random(chess.Board(), 4) for _ in range(20) ]

	r = 0
	for fen in fens:
		r += play(fen, './new', './old')
		r -= play(fen, './old', './new')

	if r > 0:
		print('1')
		variables0 = variables1.copy()
		os.system("mv ./new ./old")
		os.system(f"cp {os.path.join('src', 'Evaluator.h')} Evaluator.h.best")
	else:
		print('0')



