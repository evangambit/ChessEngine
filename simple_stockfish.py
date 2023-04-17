import re
import subprocess

class Stockfish:
  def __init__(self, path):
    self._p = subprocess.Popen(path, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    self.command("uci")

  def command(self, text):
    self._p.stdin.write((text + '\n').encode())
    self._p.stdin.flush()

  def __del__(self):
    self._p.terminate()

  def analyze(self, fen, moves = [], depth = None, nodes = None, from_whites_perspective = False):
    assert (depth is None) != (nodes is None)
    if len(moves) == 0:
      self.command(f"position fen {fen}")
    else:
      self.command(f"position fen {fen} moves {' '.join(moves)}")
    if depth is None:
      self.command(f"go nodes {nodes}")
    else:
      self.command(f"go depth {depth}")
    lines = []
    while True:
      line = self._p.stdout.readline().decode()
      if line == '':
        raise RuntimeError('')
      lines.append(line.strip())
      if line.startswith('bestmove '):
        break

    lines = [l for l in lines if re.match(r"^info depth \d+ seldepth.+$", l)]
    lines = [l for l in lines if 'lowerbound' not in l]
    lines = [l for l in lines if 'upperbound' not in l]
    if len(lines) == 0:
      raise RuntimeError('')

    line = lines[-1]
    parts = line.split(' ')[1:]

    r = {}
    i = 0
    while i < len(parts):
      if parts[i] == 'pv':
        r['pv'] = parts[i+1:]
        break
      if parts[i] == 'score':
        assert parts[i + 1] in ['cp', 'mate']
        r['score'] = parts[i+1:i+3]
        i += 3
        continue
      r[parts[i]] = parts[i + 1]
      i += 2

    intkeys = set(['depth', 'seldepth', 'multipv', 'nodes', 'nps', 'hashfull', 'tbhits', 'time'])
    for k in r:
      if k in intkeys:
        r[k] = int(r[k])

    r['score'][1] = int(r['score'][1])

    if from_whites_perspective and ' b ' in fen:
      r['score'][1] *= -1

    return r
    

