import os
 
import chess
from chess import pgn

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
import torch.utils.data as tdata

# class MonoFunc(nn.Module):
#   """
#   Represents a 1D monotonic function from x=low to x=high.

#   For x<low and x>high the function is constant.
#   """
#   def __init__(self, low, high, num_steps):
#     super(MonoFunc, self).__init__()
#     self.pivots = np.linspace(low, high, num_steps + 1)[:-1]
#     self.step_size = self.pivots[1] - self.pivots[0]
#     self.b = nn.Parameter(torch.tensor(self.pivots, dtype=torch.float32), requires_grad=False)
#     self.w = nn.Parameter(torch.zeros(num_steps, dtype=torch.float32), requires_grad=True)
#     self.bias = nn.Parameter(torch.zeros(1, dtype=torch.float32), requires_grad=True)
#   def forward(self, x):
#     shape = (1,) * len(x.shape) + (len(self.b),)
#     x = x.reshape(x.shape + (1,))
#     b = self.b.reshape(shape)
#     w = torch.exp(self.w.reshape(shape))
#     r = (((x - b).clip(0, self.step_size) + b) * w).sum(-1)
#     return r + self.bias

class MonoFunc(nn.Module):
  def __init__(self):
    super(MonoFunc, self).__init__()
    self.w = nn.Parameter(torch.zeros(4), requires_grad=True)
    # self.b = nn.Parameter(torch.zeros(4), requires_grad=True)
    self.u = nn.Parameter(torch.zeros(4), requires_grad=True)
    self.bias = nn.Parameter(torch.zeros(1, dtype=torch.float32), requires_grad=True)
  def forward(self, x):
    shape = (len(self.w),) + (1,) * len(x.shape)
    x = x.reshape((1,) + x.shape)
    w = self.w.reshape(shape)
    # b = self.b.reshape(shape)
    x = x * torch.exp(w)
    # x = x + b
    u = nn.functional.softmax(self.u, 0)
    r = x[0] * u[0]
    r += torch.sigmoid(x[1]) * u[1]
    r += torch.tanh(x[2]) * u[2]
    r += (x[3]**3) * u[3]
    return r + self.bias

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
  "PASSED_PAWNS",
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
  "KPVK_OPPOSITION",
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
  "OUTPOSTED_KNIGHTS",
  "OUTPOSTED_BISHOPS",
  "PAWN_MOVES",
  "KNIGHT_MOVES",
  "BISHOP_MOVES",
  "ROOK_MOVES",
  "QUEEN_MOVES",
  "PAWN_MOVES_ON_THEIR_SIDE",
  "KNIGHT_MOVES_ON_THEIR_SIDE",
  "BISHOP_MOVES_ON_THEIR_SIDE",
  "ROOK_MOVES_ON_THEIR_SIDE",
  "QUEEN_MOVES_ON_THEIR_SIDE",
  "KING_HOME_QUALITY",
  "BISHOPS_BLOCKING_KNIGHTS",
  "OUR_HANGING_PAWNS_2",
  "OUR_HANGING_KNIGHTS_2",
  "OUR_HANGING_BISHOPS_2",
  "OUR_HANGING_ROOKS_2",
  "OUR_HANGING_QUEENS_2",
  "THEIR_HANGING_PAWNS_2",
  "THEIR_HANGING_KNIGHTS_2",
  "THEIR_HANGING_BISHOPS_2",
  "THEIR_HANGING_ROOKS_2",
  "THEIR_HANGING_QUEENS_2",
  "QUEEN_THREATS_NEAR_KING",
  "MISSING_FIANCHETTO_BISHOP",
  "NUM_BAD_SQUARES_FOR_PAWNS",
  "NUM_BAD_SQUARES_FOR_MINORS",
  "NUM_BAD_SQUARES_FOR_ROOKS",
  "NUM_BAD_SQUARES_FOR_QUEENS",
  "IN_TRIVIAL_CHECK",
  "IN_DOUBLE_CHECK",
  "THREATS_NEAR_OUR_KING",
  "THREATS_NEAR_THEIR_KING",
  "NUM_PIECES_HARRASSABLE_BY_PAWNS",
  "PAWN_CHECKS",
  "KNIGHT_CHECKS",
  "BISHOP_CHECKS",
  "ROOK_CHECKS",
  "QUEEN_CHECKS",
  "BACK_RANK_MATE_THREAT_AGAINST_US",
  "BACK_RANK_MATE_THREAT_AGAINST_THEM",
  "OUR_KING_HAS_0_ESCAPE_SQUARES",
  "THEIR_KING_HAS_0_ESCAPE_SQUARES",
  "OUR_KING_HAS_1_ESCAPE_SQUARES",
  "THEIR_KING_HAS_1_ESCAPE_SQUARES",
  "OUR_KING_HAS_2_ESCAPE_SQUARES",
  "THEIR_KING_HAS_2_ESCAPE_SQUARES",
  "OPPOSITE_SIDE_KINGS_PAWN_STORM",
]

class PCA:
  def __init__(self, X, reg = 0.0, scale = True):
    X = X.reshape(-1, X.shape[-1])
    cov = (X.T @ X) / X.shape[0] + np.eye(X.shape[1]) * reg
    D, V = np.linalg.eigh(cov)
    assert D.min() > 1e-5, D.min()
    I = np.argsort(-D)
    self.scale = scale
    self.D = D[I].copy()
    self.V = V[:,I].copy()

  def points_forward(self, x):
    s = x.shape
    x = x.reshape(-1, s[-1])
    x = x @ self.V
    if self.scale:
      x = x / np.sqrt(self.D)
    return x.reshape(s)

  def points_backward(self, x):
    s = x.shape
    x = x.reshape(-1, s[-1])
    if self.scale:
      x = x * np.sqrt(self.D)
    x = x @ self.V.T
    return x.reshape(s)

  def slope_backward(self, w):
    D, V = np.sqrt(self.D), self.V
    if isinstance(w, nn.Parameter):
      D = torch.sqrt(torch.tensor(D, dtype=torch.float32))
      V = torch.tensor(V, dtype=torch.float32)
    s = w.shape
    w = w.reshape(1, -1)
    if self.scale:
      w = w / D
    w = w @ V.T
    return w.reshape(s)

def lpad(t, n, c=' '):
  t = str(t)
  return max(n - len(t), 0) * c + t

def logit(x):
  return torch.log(x / (1.0 - x))

# X = np.load(os.path.join('traindata', f'x.make_train_any_d10_n0.npy')).astype(np.float64)
# Y = np.load(os.path.join('traindata', f'y.make_train_any_d10_n0.npy')).astype(np.float64)
cat = np.concatenate
# X = cat([X, np.load(os.path.join('traindata', f'x.make_train_any_d10_n1.npy')).astype(np.float64)], 0)
# Y = cat([Y, np.load(os.path.join('traindata', f'y.make_train_any_d10_n1.npy')).astype(np.float64)], 0)

X = np.load(os.path.join('traindata', f'x.make_train_any_d6_n0.npy')).astype(np.float64)
Y = np.load(os.path.join('traindata', f'y.make_train_any_d6_n0.npy')).astype(np.float64)

T = X[:,varnames.index('TIME')].copy()

pca = PCA(X, 0.001, scale=False)

X = pca.points_forward(X)

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
    self.w = nn.ModuleDict()
    self.w["early"] = nn.Linear(X.shape[1], 1)
    self.w["late"] = nn.Linear(X.shape[1], 1)
    self.w["clipped"] = nn.Linear(X.shape[1], 1)
    for k in self.w:
      nn.init.zeros_(self.w[k].weight)
      nn.init.zeros_(self.w[k].bias)

  def forward(self, x, t):
    t = t.clip(0, 18)
    early = self.w["early"](x).squeeze() * (18 - t) / 18
    late = self.w["late"](x).squeeze() * t / 18
    clipped = self.w["clipped"](x).squeeze().clip(-1, 1)
    return early + late + clipped

def forever(loader):
  while True:
    for batch in loader:
      yield batch

model = Model()
opt = optim.AdamW(model.parameters(), lr=3e-3)

Xth = torch.tensor(X, dtype=torch.float32)
Yth = torch.tensor(Y, dtype=torch.float32)
Tth = torch.tensor(T, dtype=torch.float32)

bs = 25_000
maxlr = 0.03
duration = 60

dataset = tdata.TensorDataset(Xth, Tth, Yth)

loss_fn = LossFn()

L = []

windowSize = 500

for bs in 2**(np.linspace(4, 14, 11)):
  bs = int(bs)
  dataloader = tdata.DataLoader(dataset, batch_size=bs, shuffle=True, drop_last=True)

  l = []
  for x, t, y in forever(dataloader):
    pwin = ((y[:,0] + 1) + (y[:,1] + 1) / 2) / (y.sum(1) + 3)
    y = logit(pwin)
    yhat = model(x, t)
    loss = loss_fn(yhat, y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    l.append(float(loss))
    L.append(float(loss))
    if len(l) == windowSize:
      firstHalf = np.array(l[:windowSize//2]).mean()
      secondHalf = np.array(l[windowSize//2:]).mean()
      l = []
      print('%i %.4f %.4f' % (bs, firstHalf, secondHalf))
      if firstHalf < secondHalf:
        break

for k in model.w:
  w = model.w[k].weight.detach().numpy()
  w = pca.slope_backward(w)
  with torch.no_grad():
    model.w[k] = nn.Linear(w.shape[1], w.shape[0])
    model.w[k].weight *= 0
    model.w[k].weight += torch.tensor(w)

print(sum(L[-100:]) / 100)

W = {}
for k in model.w:
  W[k] = model.w[k].weight.detach().numpy().squeeze()

K = ['early', 'late', 'clipped']

print(' '.join(K))
for i in range(len(varnames)):
  print(' '.join(lpad(round(W[k][i] * 100), 5) for k in K) + '  ' + varnames[i])

for k in K:
  print(lpad(round(float(model.w[k].bias.detach().numpy() * 100)), 6) + f'  // {k} bias')
  for i in range(W[k].shape[0]):
    print(lpad(round(W[k][i] * 100), 6) + f'  // {k} {varnames[i]}')
