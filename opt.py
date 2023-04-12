import random

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

class Model:
	def run(self, w1, w2, num_trials = 20):
		w1 /= np.maximum(1.0, np.sqrt((w1**2).sum()))
		w2 /= np.maximum(1.0, np.sqrt((w2**2).sum()))
		d = w1.shape[0]
		x = np.random.normal(0, 1, (num_trials, d))
		x += np.ones(d) * 0.05
		r = (x @ w1 - x @ w2)
		return r.mean(), r.std() / np.sqrt(num_trials)

model = Model()

best = np.ones(100)
best /= np.maximum(1.0, np.sqrt((best**2).sum()))

L = []

np.random.seed(0)
for _ in range(4):
	w0 = np.random.normal(0, 1, best.shape[0])
	lr = 1 / best.shape[0]
	for batchSize in [20, 80]:
		for _ in range(10000):
			L.append((w0 * best).sum())

			w1 = w0 + np.random.normal(0, 0.1, w0.shape)
			m, s = model.run(w1, w0, num_trials = batchSize)
			z = m / s
			w0 += max(0, 0.5 * z) * (w1 - w0) * lr

	print((w0 * best).sum())
# plt.plot(uniform_filter1d(L, 10)); plt.grid(); plt.show()

"""
D, lr
3, 4
"""