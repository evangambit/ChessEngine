import matplotlib.pyplot as plt
import numpy as np

from scipy import stats

import argparse
import math
import random
import re
import shutil
import sqlite3
import subprocess
import time

from collections import Counter

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
      lines.append(' '.join(str(x).rjust(6) for x in values))
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
  print('evaluate', len(A), fn)
  assert len(A) % args.num_threads == 0, ("I'm lazy", len(A), args.num_threads)
  P = []
  batch_size = len(A) // args.num_threads
  for thread in range(args.num_threads):
    fenfn = f'/tmp/fens{thread}.txt'
    with open(fenfn, 'w+') as f:
      f.write('\n'.join(a[0] for a in A[thread*batch_size:(thread+1)*batch_size]))
    P.append(subprocess.Popen([
      './main', 'fens', fenfn, 'depth', str(args.depth), 'loadweights', fn
    ], shell=False, stdout=subprocess.PIPE))

  stdouts = []
  for p in P:
    stdout, _ = p.communicate()
    stdouts.append(stdout.decode())
  stdout = ''.join(stdouts)

  out = stdout.split('\n')

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
  parser.add_argument("--num_samples", "-n", type=int, default=20000)
  parser.add_argument("--depth", "-d", type=int, default=1)
  parser.add_argument("--num_threads", type=int, default=4)
  parser.add_argument("--zscore", type=float, default=2.0)
  args = parser.parse_args()

  conn = sqlite3.connect("db.sqlite3")
  c = conn.cursor()
  c.execute(f"""SELECT fen, moves, scores FROM tmp_sd10_d{args.depth}_margin30""")

  A = c.fetchall()
  random.shuffle(A)
  A = A[:int(args.num_samples * 1.1)]

  # # Compute eigenvectors
  # with open('/tmp/fens.txt', 'w+') as f:
  #   f.write('\n'.join(a[0] for a in A))

  # lines = subprocess.check_output(['./main', 'fens', '/tmp/fens.txt', 'mode', 'printvec-cpu']).decode().split('\n')
  # if lines[-1] == '':
  #   lines.pop()

  # assert len(lines) == len(A) * 2

  # bad_indices = []
  # X = []
  # for i, (vec, a) in enumerate(zip(lines[1::2], A)):
  #   if vec.startswith('PRINT FEATURE VEC FAIL'):
  #     bad_indices.append(i)
  #   else:
  #     X.append(np.array([int(x) for x in vec.split(' ')], dtype=np.float64))

  # for i in bad_indices[::-1]:
  #   del A[i]

  # A = A[:args.num_samples]
  # X = np.array(X)[:args.num_samples]

  # avg = X.mean(0)
  # std = X.std(0)

  # Z = (X - avg) / (std + 1e-4)

  # cov = (X.T @ X) / X.shape[0] + np.eye(X.shape[1]) * 1e-4

  # D, V = np.linalg.eigh(cov)

  # I = np.argsort(-D)
  # D, V = D[I], V[:, I]

  # # Orthogonalized X = X @ V / np.sqrt(D)

  assert len(A) == args.num_samples

  t0 = time.time()
  L0 = []
  for j in range(0, args.num_samples, 20000):
    L0 = np.concatenate([L0, evaluate(A[j:j+20000], 'weights.txt', args)])
  t1 = time.time()
  print("%.4f (%.4f secs)" % (L0.mean(), t1 - t0))

  wf = WeightFile('weights.txt')

  for i in range(1, len(wf)):
    wf = WeightFile('weights.txt')
    v0 = wf[i]

    wf[i] = v0 + args.delta
    wf.save('w0.txt')
    L1 = []

    for j in range(0, args.num_samples, 20000):
      L1 = np.concatenate([L1, evaluate(A[j:j+20000], 'w0.txt', args)])
      D = L0[:L1.size] - L1
      avg = D.mean()
      std = D.std() / math.sqrt(D.size - 1)
      if abs(avg) > std * args.zscore:
        break

    print('')
    print('%.6f %.6f (z = %.3f)' % (avg, std, avg / std if std > 0.0 else 0.0))
    if avg > std * args.zscore:
      print(f'accept wf[{i}] {v0} -> {v0 + args.delta}')
      L0 = np.concatenate([L1, evaluate(A[j+20000:], 'w0.txt', args)])
      shutil.copyfile('w0.txt', 'weights.txt')
    elif avg < std * -args.zscore:
      print(f'reject wf[{i}] {v0} -> {v0 + args.delta}')
    else:
      print(f'null   wf[{i}] {v0} -> {v0 + args.delta}')


