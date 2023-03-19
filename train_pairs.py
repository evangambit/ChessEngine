from collections import defaultdict
import os
 
import chess
from chess import pgn

import numpy as np

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
]

def lpad(t, n, c=' '):
  t = str(t)
  return max(n - len(t), 0) * c + t

X = np.load(os.path.join('traindata', f'x.pair.any_d10_q1_n0.npy')).astype(np.float64)
Y = np.load(os.path.join('traindata', f'y.pair.any_d10_q1_n0.npy')).astype(np.float64) / 100.0
F = np.load(os.path.join('traindata', f'fen.pair.any_d10_q1_n0.npy'))
S = np.load(os.path.join('traindata', f'turn.pair.any_d10_q1_n0.npy')).astype(np.float64) * 2.0 - 1.0
cat =  np.concatenate
# X = cat([X, np.load(os.path.join('traindata', 'x.endgame_d10_q1_n0.npy')).astype(np.float64)], 0)
# Y = cat([Y, np.load(os.path.join('traindata', 'y.endgame_d10_q1_n0.npy')).astype(np.float64)], 0)
# F = cat([F, np.load(os.path.join('traindata', 'f.endgame_d10_q1_n0.npy'))], 0)

T = X[:,:,varnames.index('TIME')].copy()

X[:,:,varnames.index('NUM_TARGET_SQUARES')] *= 0.0

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
    self.w["early"] = nn.Linear(X.shape[-1], 1)
    self.w["late"] = nn.Linear(X.shape[-1], 1)
    self.w["clipped"] = nn.Linear(X.shape[-1], 1)
    # self.w["scale"] = nn.Linear(X.shape[-1], 1)
    self.w["lonelyKing"] = nn.Linear(X.shape[-1], 1)
    for k in self.w:
      nn.init.zeros_(self.w[k].weight)
      nn.init.zeros_(self.w[k].bias)

    self.ai = []
    for vn in ["OUR_KNIGHTS", "OUR_BISHOPS", "OUR_ROOKS", "OUR_QUEENS"]:
      self.ai.append(varnames.index(vn))
    self.ai = torch.tensor(self.ai, dtype=torch.int64)

    self.bi = []
    for vn in ["THEIR_KNIGHTS", "THEIR_BISHOPS", "THEIR_ROOKS", "THEIR_QUEENS"]:
      self.bi.append(varnames.index(vn))
    self.bi = torch.tensor(self.bi, dtype=torch.int64)

  def forward(self, x, t):
    # t = t.clip(0, 18)
    early = self.w["early"](x).squeeze() * (18 - t) / 18
    late = self.w["late"](x).squeeze() * t / 18
    clipped = self.w["clipped"](x).squeeze().clamp_(-1.0, 1.0)
    # scale = 
    ourPieces = x[:,:, model.ai].sum(2)
    theirPieces = x[:,:, model.bi].sum(2)
    lonely = self.w["lonelyKing"](x).squeeze() * (1 - (ourPieces != 0).to(torch.float32) * (theirPieces != 0).to(torch.float32))
    r = early + late + clipped + lonely
    if 'scale' in self.w:
      r = r * (torch.sigmoid(self.w["scale"](x).squeeze()) + 0.2)
    return r

def soft_clip(x):
  x = nn.functional.leaky_relu(x + 5.0) - 5.0
  x = 5.0 - nn.functional.leaky_relu(5.0 - x)
  return x

model = Model()
opt = optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.1)

Xth = torch.tensor(X, dtype=torch.float32)
Yth = torch.tensor(Y, dtype=torch.float32)
Tth = torch.tensor(T, dtype=torch.float32)
Sth = torch.tensor(S, dtype=torch.float32)

bs = 50_000
maxlr = 0.3
duration = 60

dataset = tdata.TensorDataset(Xth, Tth, Sth, Yth)
dataloader = tdata.DataLoader(dataset, batch_size=bs, shuffle=True)

cross_entropy_loss_fn = nn.CrossEntropyLoss()

metrics = defaultdict(list)

for lr in cat([np.linspace(maxlr / 100, maxlr, duration // 2), np.linspace(maxlr, maxlr / 100, duration // 2)]):
  for pg in opt.param_groups:
    pg['lr'] = lr
  for x, t, s, y in dataloader:
    yhat = model(x, t) * s
    y = y * s
    b = (y[:,0] < y[:,1]) * 1
    loss = cross_entropy_loss_fn(nn.functional.softmax(yhat, 1), b)
    loss = loss + ((soft_clip(yhat) - soft_clip(y))**2).mean() * 0.3
    if "scale" in model.w:
      loss = loss + (model.w["scale"](x)**2).mean() * 0.2
    opt.zero_grad()
    loss.backward()
    opt.step()
    metrics['loss'].append(float(loss))
    metrics['error'].append(float((yhat.argmax(1) != b).to(torch.float32).mean()))
  print('%.4f %.4f %.4f' % (lr, metrics['loss'][-1], metrics['error'][-1]))

for k in model.w:
  w = model.w[k].weight.detach().numpy()
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
    print(f'const {datatype} k' + name + f'W0[' + str(w.shape[0]) + '] = {')
    for i in range(0, len(w), 6):
      print(','.join(lpad(x, 4) for x in w[i:i+6]) + ',')
    print('};')
  else:
    print(f'const {datatype} k' + name + f'W0[' + str(w.shape[0]) + '*' + str(w.shape[1]) + '] = {')
    for i in range(w.shape[0]):
        print(','.join(str(x) for x in w[i]) + ',')
    print('};')

kPieceName = 'PNBRQKpnbrqk'
A = np.zeros((len(F) * 2, 12, 64))
for i in range(len(F)):
  if i % 5000 == 0:
    print(i, len(F))
  fens = F[i].split(':')
  for j in range(2):
    board = chess.Board(fens[j])
    tiles = [line.split(' ') for line in str(board).split('\n')]
    for y in range(8):
      for x in range(8):
        if tiles[y][x] != '.':
          A[i * 2 + j, kPieceName.index(tiles[y][x]), y * 8 + x] = 1

A = torch.tensor(A.reshape(len(F), 2, 12 * 64), dtype=torch.float32)

class Model2(nn.Module):
  def __init__(self):
    super().__init__()
    k = 12 * 64
    self.w = nn.ModuleDict()
    self.w["early"] = nn.Linear(k, 1)
    self.w["late"] = nn.Linear(k, 1)

  def forward(self, x, t):
    early = self.w["early"](x).squeeze() * (18 - t) / 18
    late = self.w["late"](x).squeeze() * t / 18
    return early + late

dataset = tdata.TensorDataset(Xth, Tth, Sth, Yth, A)
dataloader = tdata.DataLoader(dataset, batch_size=bs, shuffle=True)

bs = 50_000
maxlr = 0.1
duration = 30

pmModel = Model2()
opt = optim.AdamW(pmModel.parameters(), lr=3e-3, weight_decay=1.0)

for lr in cat([np.linspace(maxlr / 100, maxlr, duration // 2), np.linspace(maxlr, maxlr / 100, duration // 2)]):
  for pg in opt.param_groups:
    pg['lr'] = lr
  for x, t, s, y, a in dataloader:
    yhat = model(x, t) * s
    yhat = yhat + pmModel(a, t)
    y = y * s
    b = (y[:,0] < y[:,1]) * 1
    loss = cross_entropy_loss_fn(nn.functional.softmax(yhat, 1), b)
    loss = loss + ((soft_clip(yhat) - soft_clip(y))**2).mean() * 0.3
    if "scale" in model.w:
      loss = loss + (model.w["scale"](x)**2).mean() * 0.2
    opt.zero_grad()
    loss.backward()
    opt.step()
    metrics['loss'].append(float(loss))
    metrics['error'].append(float((yhat.argmax(1) != b).to(torch.float32).mean()))
  print('%.4f %.4f %.4f' % (lr, metrics['loss'][-1], metrics['error'][-1]))


w = np.round(pmModel.w['early'].weight.reshape(12, 8, 8).detach().numpy() * 100).astype(np.int64)
for i, pn in enumerate(kPieceName):
  if pn == pn.upper():
    print('// Early White ' + pn.upper())
  else:
    print('// Early Black' + pn.upper())
  for j in range(8):
    print(','.join([lpad(x, n=4) for x in w[i,j]]) + ',')

w = np.round(pmModel.w['late'].weight.reshape(12, 8, 8).detach().numpy() * 100).astype(np.int64)
for i, pn in enumerate(kPieceName):
  if pn == pn.upper():
    print('// Late White ' + pn.upper())
  else:
    print('// Late Black' + pn.upper())
  for j in range(8):
    print(','.join([lpad(x, n=4) for x in w[i,j]]) + ',')


