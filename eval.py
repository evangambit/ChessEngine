import re
import time
import subprocess
import sys
import sqlite3
import numpy as np
from multiprocessing import Pool

def lpad(t, n, c=' '):
	t = str(t)
	return max(0, n - len(t)) * c + t

def thread_main(inputs):
	fen, bestmove = inputs
	command = [sys.argv[1], "mode", "analyze", "fen", *fen.split(' '), "time", "30"]
	output = subprocess.check_output(command).decode().strip()
	result = re.findall(r"\d+ : [^ ]+", output)[-1].split(' ')[-1]
	return result == bestmove

if __name__ == '__main__':
	conn = sqlite3.connect("db.sqlite3")
	c = conn.execute("SELECT fen, bestmove FROM TacticsTable LIMIT 3000")
	A = c.fetchall()
	t0 = time.time()
	with Pool(5) as p:
		r = np.array(p.map(thread_main, A))
	t1 = time.time()
	print(
		'%.3f' % r.mean(),
		'%.3f' % (r.std() / np.sqrt(r.shape[0])),
		'%.3f' % (t1 - t0)
	)

# 0.503 0.016 30.382
# 0.510 0.016 19.618
