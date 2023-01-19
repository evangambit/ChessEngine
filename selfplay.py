import re
import subprocess
import chess
import sys

def f(player, fen):
	command = [player, "mode", "analyze", "fen", *fen.split(' '), "time", "10"]
	output = subprocess.check_output(command).decode().strip()
	return re.findall(r"\d+ : [^ ]+", output)[-1].split(' ')[-1]

def play(fen0, player1, player2):
	r = 0
	board = chess.Board()
	mover, waiter = player1, player2
	while not board.can_claim_draw() and not board.is_stalemate() and not board.is_checkmate():
		fen = board.fen()
		move = f(mover, fen)
		if move == 'a8a8':
			print(board)
			break
		board.push_uci(move)
		mover, waiter = waiter, mover

	if board.is_checkmate():
		if waiter == player1:
			r += 1
		else:
			r -= 1

	board = chess.Board()
	mover, waiter = player2, player1
	while not board.can_claim_draw() and not board.is_stalemate() and not board.is_checkmate():
		fen = board.fen()
		move = f(mover, fen)
		if move == 'a8a8':
			print(board)
			break
		board.push_uci(move)
		mover, waiter = waiter, mover

	if board.is_checkmate():
		if waiter == player1:
			r += 1
		else:
			r -= 1

	return r


board = chess.Board()
play(board.fen(), sys.argv[1], sys.argv[2])

# Italian game

fens = [
	# Italian game
	'r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3',

	# Ruy Lopez
	'r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3'

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
]

for fen in fens:
	print(play(fen, sys.argv[1], sys.argv[2]))
