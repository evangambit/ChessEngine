import chess
import sys

board = chess.Board('rnb1kbnr/ppp1pppp/8/q7/8/2N5/PPPP1PPP/R1BQKBNR w KQkq - 2 4')
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

