import chess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("moves", nargs='*')
parser.add_argument("--fen", "-f", type=str, default='rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')
args = parser.parse_args()


board = chess.Board(args.fen)

t = ''
for i, arg in enumerate(args.moves):
	moves = list(board.legal_moves)
	moves = [m for m in moves if m.uci() == arg]
	assert len(moves) == 1, moves
	if i % 2 == 0:
		t += ' ' + str(i//2 + 1) + '.'
	t += ' ' + board.san(moves[0])
	board.push(moves[0])

print("[FEN \"" + args.fen + "\"]")
print(t[1:])

