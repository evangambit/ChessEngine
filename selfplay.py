import random
import re
import subprocess
import chess
import sys
import time
import numpy as np
from scipy import stats
from multiprocessing import Pool

from simple_stockfish import Stockfish

def f(player, fen, moves):
	command = [player, "mode", "analyze", "nodes", "10000", "fen", *fen.split(' '), "moves", *moves]
	# command = [player, "mode", "analyze", "time", "5", "fen", *fen.split(' '), "moves", *moves]
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
	isPlayer1White = ' w ' in fen0
	board = chess.Board(fen0)
	moves = []
	mover, waiter = player1, player2
	while not board.can_claim_draw() and not board.is_stalemate() and not board.is_checkmate():
		move, cmd = f(mover, fen0, moves)
		moves.append(move)
		if move == 'a8a8':
			print('a8a8', ' '.join(cmd))
			print(board)
			print(board.fen())
			print('')
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
			print('long', ' '.join(cmd))
			print(fen0, moves)
			print(board)
			print(board.fen())
			print('')
			break

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
	return (play(fen, sys.argv[1], sys.argv[2]) - play(fen, sys.argv[2], sys.argv[1])) / 2.0

if __name__ == '__main__':
	t0 = time.time()
	fens = [play_random(chess.Board(), 4) for _ in range(200)]
	with Pool(2) as p:
		r = p.map(thread_main, fens)
	r = np.array(r, dtype=np.float64).reshape(-1)

	stderr = r.std(ddof=1) / np.sqrt(r.shape[0])
	avg = r.mean()

	dt = time.time() - t0
	if dt < 60:
		print('%.1f secs' % dt)
	else:
		print('%.1f min' % (dt / 60.0))
	print('%.3f ± %.3f' % (avg, stderr))
	print(stats.norm.cdf(avg / stderr))

