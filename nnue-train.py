import os
import re

import chess
from chess import pgn

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

import torch
from torch import nn, optim
import torch.utils.data as tdata

from collections import defaultdict

tables = np.load('tables.npy').astype(np.float32)
Y = np.load('Y.npy').astype(np.float32)
turns = np.load('turns.npy').astype(np.float32)

Y = Y[:,0]

Y *= turns.squeeze()

# tables = tables[:, 768:]

assert Y[4] < 0

def soft_clip(x, k = 4.0):
  x = nn.functional.leaky_relu(x + k) - k
  x = k - nn.functional.leaky_relu(k - x)
  return x

class LossFn(nn.Module):
  def forward(self, yhat, y):
    yhat = soft_clip(yhat)
    r = yhat - y
    return (r**2).mean()

class Model(nn.Module):
  def __init__(self):
    super().__init__()
    self.seq = nn.Sequential(
      nn.Linear(tables.shape[1], 64),
      nn.ReLU(),
      nn.Linear(64, 64),
      nn.ReLU(),
      nn.Linear(64, 1, bias=False)
    )
    for layer in self.seq:
      if isinstance(layer, nn.Linear):
        nn.init.xavier_normal_(layer.weight)
        if layer.bias is not None:
          nn.init.zeros_(layer.bias)

  def forward(self, x):
    return self.seq(x)

def forever(loader):
  while True:
    for batch in loader:
      yield batch

model = Model()
opt = optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.1)

Xth = torch.tensor(tables, dtype=torch.float32)
Yth = torch.tensor(Y, dtype=torch.float32)

dataset = tdata.TensorDataset(Xth, Yth)

loss_fn = LossFn()

L = []

dataloader = tdata.DataLoader(dataset, batch_size=2048, shuffle=True, drop_last=True)
for x, y in dataloader:
  yhat = model(x)
  loss = loss_fn(yhat.squeeze(), y / 100)
  opt.zero_grad()
  loss.backward()
  opt.step()
  L.append(float(loss))
  print(sum(L[-100:]) / 100)


# windowSize = 2048

# # for bs in 2**(np.linspace(4, 14, 11)):
# for bs in 2**(np.linspace(5, 11, 7)):
#   bs = int(bs)
#   dataloader = tdata.DataLoader(dataset, batch_size=bs, shuffle=True, drop_last=True)
#   l = []
#   for x, y in forever(dataloader):
#     yhat = model(x)
#     loss = loss_fn(yhat.squeeze(), y / 100)

#     # Penalty for activations larger than 8
#     # penalty = sum(torch.relu(torch.abs(layer) - 8.0).mean() for layer in hidden_layers)

#     opt.zero_grad()
#     loss.backward()
#     opt.step()
#     l.append(float(loss))
#     L.append(float(loss))
#     if len(l) == windowSize:
#       firstHalf = np.array(l[:windowSize//2]).mean()
#       secondHalf = np.array(l[windowSize//2:]).mean()
#       l = []
#       print('%i %.4f %.4f' % (bs, firstHalf, secondHalf))
#       if firstHalf < secondHalf:
#         break

w1 = model.seq[0].weight.detach().numpy()
w2 = model.seq[2].weight.detach().numpy()
w3 = model.seq[4].weight.detach().numpy()
b1 = model.seq[0].bias.detach().numpy()
b2 = model.seq[2].bias.detach().numpy()

# save
with open('nnue.bin', 'wb') as f:
  f.write(w1.T.tobytes())
  f.write(w2.T.tobytes())
  f.write(w3.T.tobytes())
  f.write(b1.tobytes())
  f.write(b2.tobytes())

# 2048 1.4365 1.4369
