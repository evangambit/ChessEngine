import random; random.seed(0)

import json

import matplotlib.pyplot as plt
import numpy as np
import torch
import chess

from torch import nn, optim
import torch.utils.data as tdata
from tqdm import tqdm
from scipy.ndimage import uniform_filter1d as gf

kSquares = [
	chess.A8,
	chess.A7,
	chess.A6,
	chess.A5,
	chess.A4,
	chess.A3,
	chess.A2,
	chess.A1,
	chess.B8,
	chess.B7,
	chess.B6,
	chess.B5,
	chess.B4,
	chess.B3,
	chess.B2,
	chess.B1,
	chess.C8,
	chess.C7,
	chess.C6,
	chess.C5,
	chess.C4,
	chess.C3,
	chess.C2,
	chess.C1,
	chess.D8,
	chess.D7,
	chess.D6,
	chess.D5,
	chess.D4,
	chess.D3,
	chess.D2,
	chess.D1,
	chess.E8,
	chess.E7,
	chess.E6,
	chess.E5,
	chess.E4,
	chess.E3,
	chess.E2,
	chess.E1,
	chess.F8,
	chess.F7,
	chess.F6,
	chess.F5,
	chess.F4,
	chess.F3,
	chess.F2,
	chess.F1,
	chess.G8,
	chess.G7,
	chess.G6,
	chess.G5,
	chess.G4,
	chess.G3,
	chess.G2,
	chess.G1,
	chess.H8,
	chess.H7,
	chess.H6,
	chess.H5,
	chess.H4,
	chess.H3,
	chess.H2,
	chess.H1,
]

def loss_fn(yhat, y, temp = 3.0):
	yhat = torch.sigmoid(yhat / temp)
	y = torch.sigmoid(y / temp)
	return nn.functional.mse_loss(yhat, y)

def board2fn(fen: str):
	"""
	side-to-move's home row is x.reshape((8, 8, 12))][0, :, :]
	"""
	board = chess.Board(fen)
	is_black = ' b ' in fen
	pieces = 'PNBRQKpnbrqk'
	x = np.zeros((64, 12), dtype=np.int8)
	for i in kSquares:
		p = str(board.piece_at(i))
		if p == 'None':
			continue
		if is_black:
			p = p.swapcase()
		x[i, pieces.index(p)] = 1
	if ' b ' in fen:
		x = x.reshape((8, 8, 12))[::-1, :, :]
	return x.reshape((-1,))

with open('out.txt', 'r') as f:
	lines = f.read().split('\n')

lines = [json.loads(line) for line in tqdm(lines) if line != '']
random.shuffle(lines)

X = []
for line in tqdm(lines):
	X.append(board2fn(line['fen']))

X = np.array(X, dtype=np.int8)
Y = np.array([float(line['scores'][0]) / 100.0 for line in lines], dtype=np.float32)
Z = X.reshape((-1, 64, 12)).sum(1)

Xth = torch.tensor(X, dtype=torch.float32)
Yth = torch.tensor(Y, dtype=torch.float32)

class Model(nn.Module):
	def __init__(self):
		super().__init__()
		self.early = nn.Linear(8 * 8 * 12, 1)
		self.late = nn.Linear(8 * 8 * 12, 1)
	def forward(self, x, t):
		return self.early(x).squeeze() * (1.0 - t) + self.late(x).squeeze() * t

model = Model()
opt = optim.AdamW(model.parameters(), lr=3e-3)
dataset = tdata.TensorDataset(Xth, Yth)

counts2time = torch.tensor([0.0, 1.0, 1.0, 2.0, 4.0, 0.0, 0.0, 1.0, 1.0, 2.0, 4.0, 0.0])

L = []
for bs in tqdm((2**np.linspace(5, 10, 11)).astype(np.int64)):
	dataloader = tdata.DataLoader(dataset, batch_size=512, drop_last=True, shuffle=True)
	for epoch in range(2):
		for x, y in dataloader:
			counts = x.reshape((-1, 8, 8, 12)).sum((1,2))
			time = 1.0 - (counts @ counts2time).clip(0, 22) / 22.0
			yhat = model(x, time)
			loss = loss_fn(y, yhat.squeeze())
			opt.zero_grad()
			loss.backward()
			opt.step()
			L.append(float(loss))
	print(sum(L[-50:]) / 50.0)

early = model.early.weight.detach().numpy().squeeze().reshape((8, 8, 12))
late = model.late.weight.detach().numpy().squeeze().reshape((8, 8, 12))

for w in [early, late]:
	print('====' * 8)
	for i in range(6):
		a = w[:, :, i] - w[::-1, :, i + 6]
		a -= a.mean()
		print((a[::-1] * 10).astype(np.int32))


