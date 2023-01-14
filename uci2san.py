import chess
import sys

board = chess.Board()
args = sys.argv[1:]

t = ''
for i, arg in enumerate(args):
	moves = list(board.legal_moves)
	moves = [m for m in moves if m.uci() == arg]
	assert len(moves) == 1
	if i % 2 == 0:
		t += ' ' + str(i//2 + 1) + '.'
	t += ' ' + board.san(moves[0])
	board.push(moves[0])

print(t[1:])

