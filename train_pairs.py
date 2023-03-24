from collections import defaultdict
from itertools import chain
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
  "NUM_TARGET_SQUARES",
  "TIME",
  "KPVK_OPPOSITION",
  "KPVK_IN_FRONT_OF_PAWN",
  "KPVK_OFFENSIVE_KEY_SQUARES",
  "KPVK_DEFENSIVE_KEY_SQUARES",
  "SQUARE_RULE",
  "ADVANCED_PAWNS_1",
  "ADVANCED_PAWNS_2",
  "OPEN_ROOKS",
  "ROOKS_ON_THEIR_SIDE",
  "KING_CASTLED",
  "CASTLING_RIGHTS",
  "KING_IN_FRONT_OF_PASSED_PAWN",
  "KING_IN_FRONT_OF_PASSED_PAWN2",
  "PAWN_V_LONELY_KING",
  "KNIGHTS_V_LONELY_KING",
  "BISHOPS_V_LONELY_KING",
  "ROOK_V_LONELY_KING",
  "QUEEN_V_LONELY_KING",
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
]

def varnames2mask(names):
  r = np.zeros(len(varnames))
  for name in names:
    r[varnames.index(name)] = 1.0
  return r

kEarlyBlacklist = torch.tensor(varnames2mask([
  "LONELY_KING_IN_CENTER",
  "LONELY_KING_AWAY_FROM_ENEMY_KING",
  "KPVK_OPPOSITION",
  "KPVK_IN_FRONT_OF_PAWN",
  "KPVK_OFFENSIVE_KEY_SQUARES",
  "KPVK_DEFENSIVE_KEY_SQUARES",
  "SQUARE_RULE",
  "LONELY_KING_ON_EDGE",
]).reshape(1, -1), dtype=torch.float32)

kPositiveList = torch.tensor(varnames2mask([
  "OUR_PAWNS",
  "OUR_KNIGHTS",
  "OUR_BISHOPS",
  "OUR_ROOKS",
  "OUR_QUEENS",
  "OUR_PASSED_PAWNS",
  "PAWNS_CENTER_16",
  "PAWNS_CENTER_4",
  "ADVANCED_PASSED_PAWNS_2",
  "ADVANCED_PASSED_PAWNS_3",
  "ADVANCED_PASSED_PAWNS_4",
  "PAWN_MINOR_CAPTURES",
  "PAWN_MAJOR_CAPTURES",
  "PROTECTED_PASSED_PAWNS",
  "BISHOP_PAIR",
  "KNIGHT_MAJOR_CAPTURES",
  "KNIGHTS_CENTER_16",
  "THEIR_HANGING_PAWNS",
  "THEIR_HANGING_KNIGHTS",
  "THEIR_HANGING_BISHOPS",
  "THEIR_HANGING_ROOKS",
  "THEIR_HANGING_QUEENS",
  "SQUARE_RULE",
  "ADVANCED_PAWNS_1",
  "ADVANCED_PAWNS_2",
  "OUR_MATERIAL_THREATS",
  "OUTPOSTED_KNIGHTS",
  "PAWN_MOVES",
  "KNIGHT_MOVES",
  "BISHOP_MOVES",
  "ROOK_MOVES",
  "QUEEN_MOVES",
]).reshape(1, -1), dtype=torch.float32)

kNegativeList = torch.tensor(varnames2mask([
  "THEIR_PAWNS",
  "THEIR_KNIGHTS",
  "THEIR_BISHOPS",
  "THEIR_ROOKS",
  "THEIR_QUEENS",
  "IN_CHECK",
  "THEIR_PASSED_PAWNS",
  "ISOLATED_PAWNS",
  "DOUBLED_PAWNS",
  "DOUBLE_ISOLATED_PAWNS",
  "BLOCKADED_BISHOPS",
  "OUR_HANGING_PAWNS",
  "OUR_HANGING_KNIGHTS",
  "OUR_HANGING_BISHOPS",
  "OUR_HANGING_ROOKS",
  "OUR_HANGING_QUEENS",
  "THEIR_MATERIAL_THREATS",
]).reshape(1, -1), dtype=torch.float32)

def lpad(t, n, c=' '):
  t = str(t)
  return max(n - len(t), 0) * c + t

X = np.load(os.path.join('traindata', f'x.pair.any_d10_q1_n0.npy')).astype(np.float64)
Y = np.load(os.path.join('traindata', f'y.pair.any_d10_q1_n0.npy')).astype(np.float64) / 100.0
F = np.load(os.path.join('traindata', f'fen.pair.any_d10_q1_n0.npy'))
S = np.load(os.path.join('traindata', f'turn.pair.any_d10_q1_n0.npy')).astype(np.float64) * 2.0 - 1.0
A = np.load(os.path.join('traindata', f'pm.pair.any_d10_q1_n0.npy')).astype(np.float32)
cat =  np.concatenate
# X = cat([X, np.load(os.path.join('traindata', 'x.pair.endgame_d10_q1_n0.npy')).astype(np.float64)], 0)
# Y = cat([Y, np.load(os.path.join('traindata', 'y.pair.endgame_d10_q1_n0.npy')).astype(np.float64)], 0)
# F = cat([F, np.load(os.path.join('traindata', 'fen.pair.endgame_d10_q1_n0.npy'))], 0)
# S = cat([S, np.load(os.path.join('traindata', 'turn.pair.endgame_d10_q1_n0.npy')).astype(np.float64) * 2.0 - 1.0], 0)
# A = cat([A, np.load(os.path.join('traindata', 'pm.pair.endgame_d10_q1_n0.npy')).astype(np.float32)], 0)

T = X[:,:,varnames.index('TIME')].copy()

tmp = []
for vn in ["OUR_KNIGHTS", "OUR_BISHOPS", "OUR_ROOKS", "OUR_QUEENS"]:
  tmp.append(varnames.index(vn))
tmp = torch.tensor(tmp, dtype=torch.int64)
numOurPieces = X[:,:,tmp].sum(2)

tmp = []
for vn in ["THEIR_KNIGHTS", "THEIR_BISHOPS", "THEIR_ROOKS", "THEIR_QUEENS"]:
  tmp.append(varnames.index(vn))
tmp = torch.tensor(tmp, dtype=torch.int64)
numTheirPieces = X[:,:,tmp].sum(2)

tmp = []
for vn in ["OUR_QUEENS", "THEIR_QUEENS"]:
  tmp.append(varnames.index(vn))
tmp = torch.tensor(tmp, dtype=torch.int64)
numQueens = X[:,:,tmp].sum(2)


X[:,:,varnames.index('NUM_TARGET_SQUARES')] *= 0.0
X[:,:,varnames.index('PAWN_V_LONELY_KING')] *= 0.0
X[:,:,varnames.index('KNIGHTS_V_LONELY_KING')] *= 0.0
X[:,:,varnames.index('BISHOPS_V_LONELY_KING')] *= 0.0
X[:,:,varnames.index('ROOK_V_LONELY_KING')] *= 0.0
X[:,:,varnames.index('QUEEN_V_LONELY_KING')] *= 0.0

class PCA:
  def __init__(self, X, reg = 0.0):
    X = X.reshape(-1, X.shape[-1])
    cov = (X.T @ X) / X.shape[0] + np.eye(X.shape[1]) * reg
    D, V = np.linalg.eigh(cov)
    assert D.min() > 1e-5, D.min()
    I = np.argsort(-D)
    self.D = D[I].copy()
    self.V = V[:,I].copy()

  def points_forward(self, x):
    s = x.shape
    x = x.reshape(-1, s[-1])
    x = x @ self.V
    x = x / np.sqrt(self.D)
    return x.reshape(s)

  def points_backward(self, x):
    s = x.shape
    x = x.reshape(-1, s[-1])
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
    w = w / D
    w = w @ V.T
    return w.reshape(s)

def test_pca():
  n, d = 20, 4

  np.random.seed(0)
  x = np.random.normal(0, 1, (n, d)); x -= x.mean(0)
  y = np.random.normal(0, 1, (n, 1)); y -= y.mean(0)
  w = np.linalg.lstsq(x, y, rcond=0.0)[0]

  pca = PCA(x, reg=0.0)

  transformed_x = pca.points_forward(x)
  transformed_w = np.linalg.lstsq(transformed_x, y, rcond=0.0)[0]

  assert np.abs(x - pca.points_backward(transformed_x)).mean() < 1e-7
  assert np.abs(w - pca.slope_backward(transformed_w)).mean() < 1e-7

test_pca()

pca = PCA(X, reg=0.001)

X = pca.points_forward(X)

def soft_clip(x):
  x = nn.functional.leaky_relu(x + 3.0) - 3.0
  x = 3.0 - nn.functional.leaky_relu(3.0 - x)
  return x

def score_loss_fn(yhat, y):
  return ((soft_clip(yhat) - soft_clip(y))**2).mean()

class Model(nn.Module):
  def __init__(self):
    super().__init__()
    self.w = nn.ModuleDict()
    self.w["early"] = nn.Linear(X.shape[-1], 1)
    self.w["late"] = nn.Linear(X.shape[-1], 1)
    self.w["clipped"] = nn.Linear(X.shape[-1], 1)
    # self.w["scale"] = nn.Linear(X.shape[-1], 1)
    self.w["lonelyKing"] = nn.Linear(X.shape[-1], 1)
    for k in self.w:
      nn.init.zeros_(self.w[k].weight)
      nn.init.zeros_(self.w[k].bias)

  def forward(self, x, t, numOurPieces, numTheirPieces, numQueens):
    # t = t.clip(0, 18)
    early = self.w["early"](x).squeeze() * (18 - t) / 18
    late = self.w["late"](x).squeeze() * t / 18
    r = early + late
    if 'clipped' in self.w:
      r = r + self.w["clipped"](x).squeeze().clamp_(-1.0, 1.0)
      # r = r + self.w["clipped"](x).squeeze() * (numQueens == 0).to(torch.float32)
    if 'lonelyKing' in self.w:
      r = r + self.w["lonelyKing"](x).squeeze() * (1 - (numOurPieces != 0).to(torch.float32) * (numTheirPieces != 0).to(torch.float32))
    if 'scale' in self.w:
      r = r * (torch.sigmoid(self.w["scale"](x).squeeze()) + 0.2)
    return r

class PieceMapModel(nn.Module):
  def __init__(self):
    super().__init__()
    k = 12 * 64
    self.w = nn.ModuleDict()
    self.w["early"] = nn.Linear(k, 1, bias=False)
    self.w["late"] = nn.Linear(k, 1, bias=False)
    nn.init.zeros_(self.w["early"].weight)
    nn.init.zeros_(self.w["late"].weight)

  def forward(self, x, t):
    t = t / 18
    x = x.reshape(x.shape[0], 2, 12 * 64)
    early = self.w["early"](x).squeeze() * (1 - t)
    late = self.w["late"](x).squeeze() * t
    return early + late

model = Model()
pmModel = PieceMapModel()

Xth = torch.tensor(X, dtype=torch.float32)
Yth = torch.tensor(Y, dtype=torch.float32)
Tth = torch.tensor(T, dtype=torch.float32)
Sth = torch.tensor(S, dtype=torch.float32)
Ath = torch.tensor(A, dtype=torch.float32)
numOurPieces = torch.tensor(numOurPieces, dtype=torch.float32)
numTheirPieces = torch.tensor(numTheirPieces, dtype=torch.float32)
numQueens = torch.tensor(numQueens, dtype=torch.float32)

kAlpha = 0.5  # higher -> more copying stockfish
bs = min(Xth.shape[0], 50_000)
maxlr = 0.3
minlr = maxlr / 100
duration = 30

dataset = tdata.TensorDataset(Xth, Tth, Sth, Yth, Ath, numOurPieces, numTheirPieces, numQueens)
dataloader = tdata.DataLoader(dataset, batch_size=bs, shuffle=True, drop_last=True)

cross_entropy_loss_fn = nn.CrossEntropyLoss()

metrics = defaultdict(list)

for includePieceMaps in [False, True]:
  if includePieceMaps:
    opt = optim.AdamW(pmModel.parameters(), lr=3e-3, weight_decay=0.001)
  else:
    opt = optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1.0)
  for lr in cat([np.linspace(minlr, maxlr, int(round(duration * 0.1))), np.linspace(maxlr, minlr, int(round(duration * 0.9)))]):
    for pg in opt.param_groups:
      pg['lr'] = lr
    for x, t, s, y, a, nop, ntp, nq in dataloader:
      yhat = model(x, t, nop, ntp, nq) * s
      if includePieceMaps:
        yhat = yhat.detach() + pmModel(a, t)
      y = y * s
      b = (y[:,0] < y[:,1]) * 1
      loss = cross_entropy_loss_fn(nn.functional.softmax(yhat, 1), b) * (1 - kAlpha)
      loss = loss + score_loss_fn(yhat, y) * kAlpha

      w1 = pca.slope_backward(model.w['early'].weight)
      # w2 = pca.slope_backward(model.w['late'].weight)
      # w3 = pca.slope_backward(model.w['clipped'].weight)
      loss = loss + torch.abs(w1 * kEarlyBlacklist).mean()
      # loss = loss + torch.abs(torch.relu(-(w1 + w3)) * kPositiveList).mean()
      # loss = loss + torch.abs(torch.relu(-(w2 + w3)) * kPositiveList).mean()
      # loss = loss + torch.abs(torch.relu(w1 + w3) * kNegativeList).mean()
      # loss = loss + torch.abs(torch.relu(w2 + w3) * kNegativeList).mean()

      if "scale" in model.w:
        loss = loss + (model.w["scale"](x)**2).mean() * 0.2
      opt.zero_grad()
      loss.backward()
      opt.step()
      metrics['loss'].append(float(loss))
      metrics['error'].append(float((yhat.argmax(1) != b).to(torch.float32).mean()))
    print('%.4f %.4f %.4f' % (lr, sum(metrics['loss'][-4:]) / 4, sum(metrics['error'][-4:]) / 4))

for k in model.w:
  w = model.w[k].weight.detach().numpy()
  w = pca.slope_backward(w)
  with torch.no_grad():
    model.w[k] = nn.Linear(w.shape[1], w.shape[0])
    model.w[k].weight *= 0
    model.w[k].weight += torch.tensor(w)

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
    print(f'const {datatype} k' + name + 'W0[EF::NUM_EVAL_FEATURES] = {')
    for i in range(0, len(w), 6):
      print(','.join(lpad(x, 4) for x in w[i:i+6]) + ',')
    print('};')
  else:
    print(f'const {datatype} k' + name + f'W0[' + str(w.shape[0]) + '*' + str(w.shape[1]) + '] = {')
    for i in range(w.shape[0]):
        print(','.join(str(x) for x in w[i]) + ',')
    print('};')

print('====' * 6)

kPieceName = 'PNBRQKpnbrqk'
w = np.round(pmModel.w['early'].weight.reshape(12, 8, 8).detach().numpy() * 100).astype(np.int64)
for i, pn in enumerate(kPieceName):
  if pn == pn.upper():
    print('// Early White ' + pn.upper())
  else:
    print('// Early Black' + pn.upper())
  for j in range(8):
    print(','.join([lpad(x, n=4) for x in w[i,j]]) + ',')

print('====' * 6)

w = np.round(pmModel.w['late'].weight.reshape(12, 8, 8).detach().numpy() * 100).astype(np.int64)
for i, pn in enumerate(kPieceName):
  if pn == pn.upper():
    print('// Late White ' + pn.upper())
  else:
    print('// Late Black' + pn.upper())
  for j in range(8):
    print(','.join([lpad(x, n=4) for x in w[i,j]]) + ',')


