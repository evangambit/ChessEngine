import re
import subprocess

class UciPlayer:
  def __init__(self, path, weights = None):
    self.name = (path, weights)
    self._p = subprocess.Popen(path, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    self.allin = []
    self.allout = []
    self.command("uci")
    if weights is not None and weights.lower() != 'none':
      self.command(f"loadweights {weights}")

  def __del__(self):
    self._p.terminate()

  def command(self, text):
    self.allin.append(text)
    self._p.stdin.write((text + '\n').encode())
    self._p.stdin.flush()

  def setoption(self, name : str, value : str = None):
    if value is None:
      self.command(f"setoption name {name}")
    else:
      self.command(f"setoption name {name} value {value}")

  def set_multipv(self, value : int):
    self.setoption("MultiPV", value)

  @staticmethod
  def _parse_line(line):
    parts = line.split(' ')[1:]

    r = {}
    i = 0
    while i < len(parts):
      if parts[i] == 'pv':
        r['pv'] = parts[i+1:]
        break
      if parts[i] == 'score':
        assert parts[i + 1] in ['cp', 'mate']
        r['score'] = [parts[i+1], int(parts[i+2])]
        i += 3
        continue
      if parts[i] == 'wdl':
        r['wdl'] = [int(x) for x in parts[i+1:i+4]]
        i += 4
        continue
      if parts[i] == 'upperbound':  # ignore this for now
        assert parts[i + 1] == 'nodes'
        i += 1
        continue
      if i + 1 >= len(parts):
        print(parts)
      r[parts[i]] = parts[i + 1]
      i += 2

    intkeys = set(['depth', 'seldepth', 'multipv', 'nodes', 'nps', 'hashfull', 'tbhits', 'time'])
    for k in r:
      if k in intkeys:
        r[k] = int(r[k])

    return r

  def go(self, fen, depth):
    self.command(f"position fen {fen}")
    self.command(f"go depth {depth}")
    lines = []
    while True:
      line = self._p.stdout.readline().decode()
      self.allout.append(line)
      lines.append(line.strip())
      if line.startswith('bestmove '):
        break

    assert 'bestmove ' in lines[-1] # e.g. "bestmove h6h7 ponder a2a3"
    lines = [line for line in lines if re.match(rf"info depth {depth}.+", line)]

    R = []
    for line in lines:
      R.append(UciPlayer._parse_line(line))

    return R

if __name__ == '__main__':
  import chess
  player = UciPlayer('/usr/local/bin/stockfish')
  player.setoption('UCI_ShowWDL', 'true')
  r = player.go(chess.Board().fen(), depth=20)

