"""
Prints a positive number if (main1, weights1) is better than (main2, weights2)

python3 selfplay.py ./main1 weights1.txt ./main2 weights2.txt

If only the weights are different, prefer selfplay_w.py.
"""

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

def f(player, fen, moves):
	numTime = 200
	if player[1] == 'None':
		command = ["/usr/local/bin/gtimeout", str(numTime / 1000), player[0], "mode", "analyze", "time", str(numTime), "fen", *fen.split(' '), "moves", *moves]
	else:
		command = ["/usr/local/bin/gtimeout", str(numTime / 1000), player[0], "loadweights", player[1], "mode", "analyze", "time", str(numTime), "fen", *fen.split(' '), "moves", *moves]
	# command = [player, "mode", "analyze", "time", "5", "fen", *fen.split(' '), "moves", *moves]
	try:
		stdout = subprocess.check_output(command).decode()
	except Exception as e:
		stdout = e.stdout.decode()
	# stdout = subprocess.check_output(command).decode()
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
	player1, player2 = (sys.argv[1], sys.argv[2]), (sys.argv[3], sys.argv[4])
	return (play(fen, player1, player2) - play(fen, player2, player1)) / 2.0

def create_fen_batch(n):
	return [play_random(chess.Board(), 4) for _ in range(n)]

if __name__ == '__main__':
	mp.set_start_method('spawn')
	t0 = time.time()
	numWorkers = 8
	batches = [[]]
	for i in range(0, 64, numWorkers):
		batches.append(create_fen_batch(numWorkers))

	R = []
	with mp.Pool(numWorkers) as pool:
		for batch in tqdm(batches):
			try:
				r = pool.map_async(thread_main, batch).get(timeout=120)
				R += r
			except mp.context.TimeoutError:
				print('timeout')
				pass
	r = np.array(R, dtype=np.float64).reshape(-1)

	stderr = r.std(ddof=1) / np.sqrt(r.shape[0])
	avg = r.mean()

	dt = time.time() - t0
	if dt < 60:
		print('%.1f secs' % dt)
	else:
		print('%.1f min' % (dt / 60.0))
	print('%.4f ± %.4f' % (avg, stderr))
	print(stats.norm.cdf(avg / stderr))

