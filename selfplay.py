import random
import re
import subprocess
import chess
import sys
import numpy as np
from scipy import stats
from multiprocessing import Pool

def f(player, fen, moves):
	command = [player, "mode", "analyze", "fen", *fen.split(' '), "time", "40", "moves", *moves]
	output = subprocess.check_output(command).decode().strip()
	return re.findall(r"\d+ : [^ ]+", output)[-1].split(' ')[-1], command

def play(fen0, player1, player2):
	board = chess.Board(fen0)
	moves = []
	mover, waiter = player1, player2
	while not board.can_claim_draw() and not board.is_stalemate() and not board.is_checkmate():
		move, cmd = f(mover, fen0, moves)
		moves.append(move)
		if move == 'a8a8':
			print(board)
			print(board.fen())
			break
		try:
			board.push_uci(move)
		except (ValueError) as e:
			print(fen0, moves)
			print(board.fen())
			print(board)
			print(' '.join(cmd))
			raise e
		mover, waiter = waiter, mover

	if board.is_checkmate():
		if waiter == player1:
			return 1
		else:
			return -1
	else:
		return 0


def play_random(board, n):
	if n <= 0:
		return board.fen()
	move = random.choice(list(board.legal_moves))
	board.push(move)
	return play_random(board, n - 1)

def thread_main(fen):
	r = [
		play(fen, sys.argv[1], sys.argv[2]),
		-play(fen, sys.argv[2], sys.argv[1]),
	]
	return r

if __name__ == '__main__':
	fens = [play_random(chess.Board(), 4) for _ in range(20)]
	with Pool(2) as p:
		r = p.map(thread_main, fens)
	r = np.array(r, dtype=np.float64).reshape(-1)

	t = r.mean() / (r.std(ddof=1) / np.sqrt(r.shape[0]))

	print(int(r.sum()), r.shape[0])
	print(stats.t.cdf(t, 1))
