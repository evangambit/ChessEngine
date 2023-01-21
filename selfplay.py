import re
import subprocess
import chess
import sys
import numpy as np
from scipy import stats
from multiprocessing import Pool

def f(player, fen, moves):
	command = [player, "mode", "analyze", "fen", *fen.split(' '), "time", "30", "moves", *moves]
	output = subprocess.check_output(command).decode().strip()
	return re.findall(r"\d+ : [^ ]+", output)[-1].split(' ')[-1]

def play(fen0, player1, player2):
	board = chess.Board(fen0)
	moves = []
	mover, waiter = player1, player2
	while not board.can_claim_draw() and not board.is_stalemate() and not board.is_checkmate():
		fen = board.fen()
		move = f(mover, fen, moves)
		moves.append(move)
		if move == 'a8a8':
			print(board)
			break
		board.push_uci(move)
		mover, waiter = waiter, mover

	if board.is_checkmate():
		if waiter == player1:
			print(' 1', board.fen())
			return 1
		else:
			print('-1', board.fen())
			return -1
	else:
		print(' 0', board.fen())
		return 0


board = chess.Board()

# Italian game

fens = [
	# Italian game
	'r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3',

	# Ruy Lopez
	'r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3',

	# Classical Sicilian
	'r1bqkb1r/pp2pppp/2np1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 3 6',

	# Smith-Morra Gambit
	'rnbqkbnr/pp1ppppp/8/8/4P3/2N5/PP3PPP/R1BQKBNR b KQkq - 0 4',

	# King's Gambit accepted
	'rnbqkbnr/pppp1ppp/8/8/4Pp2/8/PPPP2PP/RNBQKBNR w KQkq - 0 3',

	# Queen's Gambit accepted
	'rnbqkbnr/ppp1pppp/8/8/2pP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3',

	# Old Benoni
	'rnbqkbnr/pp1ppppp/8/2pP4/8/8/PPP1PPPP/RNBQKBNR b KQkq - 0 2',

	# King's Indian Defense
	'rnbqk2r/ppppppbp/5np1/8/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 2 4',

	# Scandinavian Defence
	'rnb1kbnr/ppp1pppp/8/q7/8/2N5/PPPP1PPP/R1BQKBNR w KQkq - 2 4',

	# Albin Countergambit
	'rnbqkbnr/ppp2ppp/8/3pp3/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3',

	# Borg defense
	'rnbqkbnr/pppppp2/7p/6p1/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 0 3',

	# Bird's Opening
	'rnbqkbnr/ppp1pppp/8/3p4/5P2/8/PPPPP1PP/RNBQKBNR w KQkq - 0 2',

	# Hungarian Opening
	'rnbqkb1r/ppp1pppp/5n2/3p4/8/5NP1/PPPPPPBP/RNBQK2R b KQkq - 3 3',

	# French Defense
	'rnbqkbnr/ppp2ppp/4p3/3p4/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 0 3',

	# Pirc Defense
	'rnbqkb1r/ppp1pppp/3p1n2/8/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 1 3',
]

def thread_main(fen):
	a, b = 0, 0
	a += play(fen, sys.argv[1], sys.argv[2])
	a -= play(fen, sys.argv[2], sys.argv[1])
	return a, b

if __name__ == '__main__':
	with Pool(5) as p:
		r = p.map(thread_main, fens)
	r = np.array(r, dtype=np.float64).reshape(-1)

	t = r.mean() / (r.std(ddof=1) / np.sqrt(r.shape[0]))

	print(r.sum(), r.shape[0])
	print(stats.t.cdf(0.5, 1))
