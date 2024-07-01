import zstandard as zstd

# lichess_db_standard_rated_2019-01.pgn.zst

import time
import json
import struct
import argparse
import os
import random
import subprocess
import sqlite3
import re

from scipy.special import logit

from multiprocessing import Process, Queue
 
import chess
from chess import pgn

from uci_player import UciPlayer

import numpy as np

from tqdm import tqdm

def get_table_name(args):
  r = f"make_train2_d{args.depth}_n{args.noise}"
  return r

def shard_counter_to_name(n):
  return (hex(n)[2:]).rjust(4, '0')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("pgnfiles", nargs='*')
  parser.add_argument("--odds", type=int, default=1)
  parser.add_argument("--nodes_per_shard", type=int, default=10)
  parser.add_argument("--num_shards", type=int, default=10)
  parser.add_argument("--output", type=str, required=True)
  args = parser.parse_args()

  assert args.odds >= 1
  assert args.nodes_per_shard >= 1
  assert args.num_shards >= 1

  node_counter = 0
  shard_counter = 0

  tstart = time.time()
  outfile = open(args.output + f'.{shard_counter_to_name(shard_counter)}', 'w')
  for pgnfile in args.pgnfiles:
    with zstd.open(args.pgnfiles[0], 'r') as f:
      while True:
        game = chess.pgn.read_game(f)
        if game is None:
          break
        for i, node in enumerate(game.mainline()):
          e = re.findall(r"%eval ([^\]]+)", node.comment)
          if e and random.randint(1, args.odds) == 1:
            fen = node.board().fen()
            outfile.write(str(e[0]) + ':' + fen + '\n')
            node_counter += 1
            if node_counter >= args.nodes_per_shard:
              node_counter = 0
              shard_counter += 1
              outfile.close()
              print('Done with shard', shard_counter, 'in', time.time() - tstart, 'seconds')
              tstart = time.time()
              if shard_counter >= args.num_shards:
                break
              outfile = open(args.output + f'.{shard_counter_to_name(shard_counter)}', 'w')
        if shard_counter >= args.num_shards:
          break
      if shard_counter >= args.num_shards:
        break
    if shard_counter >= args.num_shards:
      break

