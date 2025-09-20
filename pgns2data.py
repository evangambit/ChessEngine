import io
import json
import math
import os

"""
python pgns2data.py > data/lichess/pos.txt
awk 'BEGIN{srand();} {printf "%06d %s\n", rand()*1000000, $0;}' data/lichess/pos.txt | sort -n | cut -c8- > data/lichess/pos.shuf.txt
rm data/lichess/x*
./make_tables data/lichess/pos.shuf.txt data/lichess/x
"""

import chess.pgn as pgn
import chess
import zstandard as zstd

from tqdm import tqdm

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

kNumToMake = 1_000_000

def get_data():
    with zstd.open(os.path.join('data', 'lichess_db_eval.jsonl.zst'), 'rb') as f:
        text_stream = io.TextIOWrapper(f, encoding='utf-8')
        for line_idx, line in enumerate(tqdm(text_stream, total=288_977_307)):
            if line_idx % (288_977_307 // kNumToMake) != 0:
                continue
            obj = json.loads(line)
            board = chess.Board(obj['fen'])
            obj['evals'].sort(key=lambda e: -len(e['pvs']))
            e = [e for e in obj['evals'] if e['depth'] > 10]
            if len(e) == 0:
                continue
            e = e[0]
            for variation in e['pvs']:
                line = variation['line'].split(' ')
                if len(line) < 2:
                    continue
                try:
                    board.push_uci(line[0])
                except chess.IllegalMoveError:
                    continue
                try:
                    san = board.san(board.parse_uci(line[1]))
                    if '+' not in san and '#' not in san and 'x' not in san:
                        if 'cp' in variation:
                            yield board.fen(), int(sigmoid(variation['cp'] / 100) * 1000)
                        else:
                            yield board.fen(), 1000 if variation['mate'] > 0 else 0
                except chess.IllegalMoveError:
                    pass
                board.pop()

for fen, eval in get_data():
    print(f'{fen}|{eval}')
