import numpy as np

import argparse
import math
import random
import re
import sqlite3
import subprocess
import time

class WeightFile:
  def __init__(self, file):
    with open(file, 'r') as f:
      lines = f.read().split('\n')

    self.values = []
    self.comments = []
    for i, line in enumerate(lines):
      if "//" in line:
        self.comments.append(line[line.index('//'):])
        line = line[:line.index('//')]
      else:
        self.comments.append('')

      self.values.append([int(x) for x in re.findall(r"\-?\d+", line)])

    self.numFloats = [len(v) for v in self.values]
    self.cdf = np.array([0] + np.cumsum(self.numFloats).tolist())

  def save(self, filename):
    lines = []
    for comment, values in zip(self.comments, self.values):
      lines.append(' '.join(str(x).rjust(8) for x in values))
      if comment != '':
        lines[-1] +=  ' ' + comment
      if lines[-1].startswith(' //'):
        lines[-1] = lines[-1][1:]
    with open(filename, 'w+') as f:
      f.write('\n'.join(lines))


  def __len__(self):
    return self.cdf[-1]

  def __getitem__(self, i):
    if i >= len(self):
      raise IndexError('')
    idx = (self.cdf > i).argmax() - 1
    offset = i - self.cdf[idx]
    return self.values[idx][offset]

  def __setitem__(self, i, v):
    if i >= len(self):
      raise IndexError('')
    idx = (self.cdf > i).argmax() - 1
    offset = i - self.cdf[idx]
    self.values[idx][offset] = v

def evaluate(A, fn, args):
  A = A[:args.numSamples]

  with open('/tmp/fens.txt', 'w+') as f:
    f.write('\n'.join(a[0] for a in A))

  out = subprocess.check_output(['./main', 'fens', '/tmp/fens.txt', 'depth', str(args.depth), 'loadweights', fn]).decode().split('\n')
  if out[-1] == '':
    out.pop()

  assert len(out) == len(A) * 2

  L = []
  for gt, pred in zip(A, out[1::2]):
    pred = re.findall(r"^PV 1: .\d+ (\S+)", pred)[0]
    moves = gt[1].split(' ')
    scores = [int(x) for x in gt[2].split(' ')]
    if pred in moves:
      L.append(abs(scores[moves.index(pred)] - scores[0]))
    else:
      L.append(abs(scores[-1] - scores[0]))

  L = np.array(L, dtype=np.float64).clip(0, 200) / 100.0

  return L

if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument("--delta", type=int, required=True)
  parser.add_argument("--numSamples", "-n", type=int, default=1000)
  parser.add_argument("--depth", "-d", type=int, default=1)
  args = parser.parse_args()

  wf = WeightFile('weights.txt')

  conn = sqlite3.connect("db.sqlite3")
  c = conn.cursor()
  c.execute(f"""SELECT fen, moves, scores FROM tmp_d10""")

  A = c.fetchall()

  for i in range(1, len(wf)):
    random.shuffle(A)
    L0 = evaluate(A, 'weights.txt', args)

    wf[i] = wf[i] + args.delta
    wf.save('w0.txt')
    L1 = evaluate(A, 'w0.txt', args)

    D1 = L1 - L0
    z = D1.mean() / (D1.std() / math.sqrt(D1.size - 1))
    print(i, '+', ('%.3f' % z).rjust(7))

    wf[i] = wf[i] - args.delta * 2
    wf.save('w0.txt')
    L2 = evaluate(A, 'w0.txt', args)

    D2 = L2 - L0
    z = D2.mean() / (D2.std() / math.sqrt(D2.size - 1))
    print(i, '-', ('%.3f' % z).rjust(7))

    wf[i] = wf[i] + args.delta


