import os
 
import chess
from chess import pgn

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
import torch.utils.data as tdata

varnames = [
  "OUR_PAWNS",
  "OUR_KNIGHTS",
  "OUR_BISHOPS",
  "OUR_ROOKS",
  "OUR_QUEENS",
  "THEIR_PAWNS",
  "THEIR_KNIGHTS",
  "THEIR_BISHOPS",
  "THEIR_ROOKS",
  "THEIR_QUEENS",
  "IN_CHECK",
  "KING_ON_BACK_RANK",
  "KING_ON_CENTER_FILE",
  "KING_ACTIVE",
  "THREATS_NEAR_KING_2",
  "THREATS_NEAR_KING_3",
  "OUR_PASSED_PAWNS",
  "THEIR_PASSED_PAWNS",
  "ISOLATED_PAWNS",
  "DOUBLED_PAWNS",
  "DOUBLE_ISOLATED_PAWNS",
  "PAWNS_CENTER_16",
  "PAWNS_CENTER_4",
  "ADVANCED_PASSED_PAWNS_2",
  "ADVANCED_PASSED_PAWNS_3",
  "ADVANCED_PASSED_PAWNS_4",
  "PAWN_MINOR_CAPTURES",
  "PAWN_MAJOR_CAPTURES",
  "PROTECTED_PAWNS",
  "PROTECTED_PASSED_PAWNS",
  "BISHOPS_DEVELOPED",
  "BISHOP_PAIR",
  "BLOCKADED_BISHOPS",
  "SCARY_BISHOPS",
  "SCARIER_BISHOPS",
  "BLOCKADED_ROOKS",
  "SCARY_ROOKS",
  "INFILTRATING_ROOKS",
  "KNIGHTS_DEVELOPED",
  "KNIGHT_MAJOR_CAPTURES",
  "KNIGHTS_CENTER_16",
  "KNIGHTS_CENTER_4",
  "KNIGHT_ON_ENEMY_SIDE",
  "OUR_HANGING_PAWNS",
  "OUR_HANGING_KNIGHTS",
  "OUR_HANGING_BISHOPS",
  "OUR_HANGING_ROOKS",
  "OUR_HANGING_QUEENS",
  "THEIR_HANGING_PAWNS",
  "THEIR_HANGING_KNIGHTS",
  "THEIR_HANGING_BISHOPS",
  "THEIR_HANGING_ROOKS",
  "THEIR_HANGING_QUEENS",
  "LONELY_KING_IN_CENTER",
  "LONELY_KING_AWAY_FROM_ENEMY_KING",
  "TIME",
  "PAWN_PM",
  "KNIGHT_PM",
  "BISHOP_PM",
  "ROOK_PM",
  "QUEEN_PM",
  "KING_PM",
  "KPVK_OPPOSITION",
  "KPVK_IN_FRONT_OF_PAWN",
  "KPVK_OFFENSIVE_KEY_SQUARES",
  "KPVK_DEFENSIVE_KEY_SQUARES",
  "SQUARE_RULE",
  "ADVANCED_PAWNS_1",
  "ADVANCED_PAWNS_2",
  "OPEN_ROOKS",
  "ROOKS_ON_THEIR_SIDE",
  "KING_IN_FRONT_OF_PASSED_PAWN",
  "KING_IN_FRONT_OF_PASSED_PAWN2",
  "OUR_MATERIAL_THREATS",
  "THEIR_MATERIAL_THREATS",
  "LONELY_KING_ON_EDGE",
]

class PCA:
  def __init__(self, X, dims = None):
    # self.std = X.std(0) + 1e-5
    cov = (X.T @ X) / X.shape[0] + np.eye(X.shape[1]) * 0.01
    D, V = np.linalg.eigh(cov)
    assert D.min() > 1e-5, D.min()
    I = np.argsort(-D)
    self.D = D[I].copy()
    self.V = V[:,I].copy()
    if dims is not None:
      self.D = self.D[:dims]
      self.V = self.V[:, :dims]

  def forward(self, x):
    x = x @ self.V
    # x = x / np.sqrt(self.D)
    return x

  def backward(self, x):
    # x = x * np.sqrt(self.D)
    x = x @ self.V.T
    return x


def lpad(t, n, c=' '):
  t = str(t)
  return max(n - len(t), 0) * c + t

X = np.load(os.path.join('traindata', f'x.any_d10_q1_n3.npy')).astype(np.float64)
Y = np.load(os.path.join('traindata', f'y.any_d10_q1_n3.npy')).astype(np.float64)
F = np.load(os.path.join('traindata', f'f.any_d10_q1_n3.npy'))
cat =  np.concatenate
# X = cat([X, np.load(os.path.join('traindata', 'x.endgame_d10_q1_n0.npy')).astype(np.float64)], 0)
# Y = cat([Y, np.load(os.path.join('traindata', 'y.endgame_d10_q1_n0.npy')).astype(np.float64)], 0)
# F = cat([F, np.load(os.path.join('traindata', 'f.endgame_d10_q1_n0.npy'))], 0)

T = X[:,varnames.index('TIME')].copy()

pca = PCA(X, X.shape[1] - 2)

X = pca.forward(X)

def soft_clip(x):
  x = nn.functional.leaky_relu(x + 5.0) - 5.0
  x = 5.0 - nn.functional.leaky_relu(5.0 - x)
  return x

class LossFn(nn.Module):
  def forward(self, yhat, y):
    yhat = soft_clip(yhat)
    r = yhat - y
    return (r**2).mean()

class Model(nn.Module):
  def __init__(self):
    super().__init__()
    self.w = nn.ModuleDict()
    self.w["early"] = nn.Linear(X.shape[1], 1)
    self.w["late"] = nn.Linear(X.shape[1], 1)
    self.w["clipped"] = nn.Linear(X.shape[1], 1)
    self.w["scale"] = nn.Linear(X.shape[1], 1)
    for k in self.w:
      nn.init.zeros_(self.w[k].weight)
      nn.init.zeros_(self.w[k].bias)

  def forward(self, x, t):
    t = t.clip(0, 18)
    early = self.w["early"](x).squeeze() * (18 - t) / 18
    late = self.w["late"](x).squeeze() * t / 18
    clipped = self.w["clipped"](x).squeeze().clip(-1, 1)
    scale = self.w["scale"](x).squeeze()
    return (early + late + clipped) * (torch.sigmoid(scale) + 0.2)

model = Model()
opt = optim.AdamW(model.parameters(), lr=3e-3)

Xth = torch.tensor(X, dtype=torch.float32)
Yth = torch.tensor(Y / 100, dtype=torch.float32)
Tth = torch.tensor(T, dtype=torch.float32)

bs = 50_000
maxlr = 0.1
duration = 40

dataset = tdata.TensorDataset(Xth, Tth, Yth)
dataloader = tdata.DataLoader(dataset, batch_size=bs, shuffle=True)

loss_fn = LossFn()

L = []

for lr in cat([np.linspace(maxlr / 100, maxlr, duration // 2), np.linspace(maxlr, maxlr / 100, duration // 2)]):
  for pg in opt.param_groups:
    pg['lr'] = lr
  for x, t, y in dataloader:
    loss = loss_fn(model(x, t), y)
    loss = loss + (model.w["scale"](x)**2).mean() * 0.2
    opt.zero_grad()
    loss.backward()
    opt.step()
    L.append(float(loss))
  print(lr, L[-1])

for k in model.w:
  w = model.w[k].weight.detach().numpy()
  w = pca.backward(w)
  with torch.no_grad():
    model.w[k] = nn.Linear(w.shape[1], w.shape[0])
    model.w[k].weight *= 0
    model.w[k].weight += torch.tensor(w)

print(L[-1])

W = {}
for k in model.w:
  W[k] = model.w[k].weight.detach().numpy().squeeze()

print([lpad(k, 5) for k in W])
for i in range(W['early'].shape[0]):
  A = [lpad(round(W[k][i] * 100), 5) for k in W]
  A.append(varnames[i])
  print(*A)

datatype = 'int32_t'

for name in model.w:
  linear = model.w[name]
  name = name[0].upper() + name[1:]
  w = linear.weight.squeeze().cpu().detach().numpy()
  b = linear.bias.squeeze().cpu().detach().numpy()
  w, b = w.astype(np.float32), b.astype(np.float32)
  w, b = np.round(w * 100).astype(np.int32), np.round(b * 100).astype(np.int32)
  if len(b.shape) == 1:
    print(f'const {datatype} k' + name + f'B0[' + str(w.shape[0]) + '] = {')
    for i in range(0, len(b), 6):
      print(','.join(lpad(x, 4) for x in b[i:i+6]) + ',')
    print('};')
  else:
    print(f'const {datatype} k' + name + f'B0 = ' + str(b) + ';')
  if len(w.shape) == 1:
    print(f'const {datatype} k' + name + f'W0[' + str(w.shape[0]) + '] = {')
    for i in range(0, len(w), 6):
      print(','.join(lpad(x, 4) for x in w[i:i+6]) + ',')
    print('};')
  else:
    print(f'const {datatype} k' + name + f'W0[' + str(w.shape[0]) + '*' + str(w.shape[1]) + '] = {')
    for i in range(w.shape[0]):
        print(','.join(str(x) for x in w[i]) + ',')
    print('};')

