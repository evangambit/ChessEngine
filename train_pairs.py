"""
TODO: A special evaluation function for opposite-side castling?

"""

from collections import defaultdict
from itertools import chain
import os
 
import chess
from chess import pgn

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torch import nn, optim
import torch.utils.data as tdata

kIncludePiecemaps = False

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
  "OUR_KING_HAS_0_ESCAPE_SQUARES",
  "THEIR_KING_HAS_0_ESCAPE_SQUARES",
  "OUR_KING_HAS_1_ESCAPE_SQUARES",
  "THEIR_KING_HAS_1_ESCAPE_SQUARES",
  "OUR_KING_HAS_2_ESCAPE_SQUARES",
  "THEIR_KING_HAS_2_ESCAPE_SQUARES",
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
  "THEIR_HANGING_PAWNS_2",
  "THEIR_HANGING_KNIGHTS_2",
  "THEIR_HANGING_BISHOPS_2",
  "THEIR_HANGING_ROOKS_2",
  "THEIR_HANGING_QUEENS_2",
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
  "OUR_HANGING_PAWNS_2",
  "OUR_HANGING_KNIGHTS_2",
  "OUR_HANGING_BISHOPS_2",
  "OUR_HANGING_ROOKS_2",
  "OUR_HANGING_QUEENS_2",
]).reshape(1, -1), dtype=torch.float32)

def lpad(t, n, c=' '):
  t = str(t)
  return max(n - len(t), 0) * c + t

cat =  np.concatenate

X, Y, F, S = [], [], [], []
if kIncludePiecemaps:
  A = []
for fn in [
  "x.pair.any_d8_q1_n1.npy",
  "x.pair.any_d8_q1_n2.npy",
  "x.pair.any_d8_q1_n3.npy",
  "x.pair.any_d8_q1_n5.npy",
  ]:
  if len(X) == 0:
    X = np.load(os.path.join('traindata', fn))
    Y = np.load(os.path.join('traindata', fn.replace('x', 'y')))
    F = np.load(os.path.join('traindata', fn.replace('x', 'fen')))
    S = np.load(os.path.join('traindata', fn.replace('x', 'turn')))
    if kIncludePiecemaps:
      A = np.load(os.path.join('traindata', fn.replace('x', 'pm')))
  else:
    X = cat([X, np.load(os.path.join('traindata', fn)).astype(np.float64)], 0)
    Y = cat([Y, np.load(os.path.join('traindata', fn.replace('x', 'y'))).astype(np.float64)], 0)
    F = cat([F, np.load(os.path.join('traindata', fn.replace('x', 'fen')))], 0)
    S = cat([S, np.load(os.path.join('traindata', fn.replace('x', 'turn'))).astype(np.float64) * 2.0 - 1.0], 0)
    if kIncludePiecemaps:
      A = cat([A, np.load(os.path.join('traindata', fn.replace('x', 'pm'))).astype(np.float32)], 0)

X = X.astype(np.float32)
Y = Y.astype(np.float32)

assert X.shape[-1] == len(varnames)

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

# Remove positions where one side only has a king.
isLonelyKing = (1 - (numOurPieces != 0) * (numTheirPieces != 0)).astype(bool)
I = isLonelyKing.sum(1) == 0

X, Y, F, S, T = X[I], Y[I], F[I], S[I], T[I]
if kIncludePiecemaps:
  A = A[I]

tmp = []
for vn in ["OUR_QUEENS", "THEIR_QUEENS"]:
  tmp.append(varnames.index(vn))
tmp = torch.tensor(tmp, dtype=torch.int64)
numQueens = X[:,:,tmp].sum(2)

X[:,:,varnames.index('KPVK_OFFENSIVE_KEY_SQUARES')] *= 0.0
X[:,:,varnames.index('KPVK_DEFENSIVE_KEY_SQUARES')] *= 0.0
X[:,:,varnames.index('SQUARE_RULE')] *= 0.0



# X = X[:,:,:12]
# varnames = varnames[:12]
# kEarlyBlacklist = kEarlyBlacklist[0,:12]
# kPositiveList = kPositiveList[0,:12]
# kNegativeList = kNegativeList[0,:12]

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
    self.dropout = nn.Dropout(p=0.01)
    for k in self.w:
      nn.init.zeros_(self.w[k].weight)
      nn.init.zeros_(self.w[k].bias)

  def forward(self, x, t, numQueens):
    # t = t.clip(0, 18)
    x = torch.cat([x[:,:11], self.dropout(x[:, 11:])], 1)
    early = self.w["early"](x).squeeze() * (18 - t) / 18
    late = self.w["late"](x).squeeze() * t / 18
    r = early + late
    if 'clipped' in self.w:
      r = r + self.w["clipped"](x).squeeze().clamp_(-1.0, 1.0)
    if 'scale' in self.w:
      r = r * (torch.sigmoid(self.w["scale"](x).squeeze()) + 0.2)
    return r

class PieceMapModel(nn.Module):
  def __init__(self):
    super().__init__()
    k = 6 * 64
    self.w = nn.ModuleDict()
    self.w["early"] = nn.Linear(k, 1, bias=False)
    self.w["late"] = nn.Linear(k, 1, bias=False)
    self.dropout = nn.Dropout(p=0.01)
    nn.init.zeros_(self.w["early"].weight)
    nn.init.zeros_(self.w["late"].weight)

  def forward(self, x, t):
    t = t / 18
    x = x.reshape(x.shape[0], 2, 6 * 64)
    x = self.dropout(x)
    early = self.w["early"](x).squeeze() * (1 - t)
    late = self.w["late"](x).squeeze() * t
    return early + late

model = Model()
pmModel = PieceMapModel()
stdModel = nn.Linear(X.shape[-1], 1)

Xth = torch.tensor(X, dtype=torch.float32)
Yth = torch.tensor(Y, dtype=torch.float32)
Tth = torch.tensor(T, dtype=torch.float32)
Sth = torch.tensor(S, dtype=torch.float32)
if kIncludePiecemaps:
  Ath = torch.tensor(A, dtype=torch.float32)
numQueens = torch.tensor(numQueens, dtype=torch.float32)

if kIncludePiecemaps:
  PmEarlySampleSizeTh = torch.tensor((A * (1.0 - T.reshape(T.shape + (1,1)) / 18)).sum((0, 1)), dtype=torch.float32)
  PmEarlySampleSizeTh = PmEarlySampleSizeTh.reshape((1, 1) + PmEarlySampleSizeTh.shape)
  PmLateSampleSizeTh = torch.tensor((A * T.reshape(T.shape + (1,1)) / 18).sum((0, 1)), dtype=torch.float32)
  PmLateSampleSizeTh = PmLateSampleSizeTh.reshape((1, 1) + PmLateSampleSizeTh.shape)

kAlpha = 0.9  # higher -> more copying stockfish
bs = min(Xth.shape[0], 10_000)
maxlr = 0.1
minlr = maxlr / 300
numIters = 20000

if kIncludePiecemaps:
  dataset = tdata.TensorDataset(Xth, Tth, Sth, Yth, Ath, numQueens)
else:
  dataset = tdata.TensorDataset(Xth, Tth, Sth, Yth, numQueens)
dataloader = tdata.DataLoader(dataset, batch_size=bs, shuffle=True, drop_last=True, num_workers=12)

def loop(iterable, n):
  it = 0
  while it < n:
    for batch in iterable:
      it += 1
      yield batch
      if it >= n:
        break

class PiecewiseFunction:
  def __init__(self, x, y):
    self.x = np.array(x, dtype=np.float64)
    self.y = np.array(y, dtype=np.float64)
  
  def __call__(self, x):
    if x <= self.x[0]:
      return self.y[0]
    if x >= self.x[-1]:
      return self.y[-1]
    high = np.searchsorted(self.x, x)
    low = high - 1
    t = (x - self.x[low]) / (self.x[high] - self.x[low])
    return self.y[low] * (1 - t) + self.y[high] * t

def cross_entropy_loss_fn(yhat, y):
  b = (y[:,0] < y[:,1]).to(torch.int64)
  scale = torch.abs(y[:,0].clamp(-5, 5) - y[:,1].clamp(-5, 5))
  loss = (nn.functional.cross_entropy(yhat, b, reduction='none') * scale).mean()

  error = (yhat.argmax(1) != b).to(torch.float32)
  cploss = (error * scale).mean()

  return loss, float(error.mean()), float(cploss)

metrics = defaultdict(list)

pf = PiecewiseFunction([0, numIters // 10, numIters], [minlr, maxlr, minlr])

if os.path.exists('model.pth'):
  s = torch.load('model.pth')
  if 'lonelyKing' not in model.w and 'lonelyKing' in s:
    del s['w.lonelyKing.bias']
    del s['w.lonelyKing.weight']
  model.load_state_dict(s, strict=False)
  if kIncludePiecemaps:
    pmModel.load_state_dict(torch.load('pmModel.pth'), strict=False)

opt = optim.AdamW(chain(pmModel.parameters(), model.parameters(), stdModel.parameters()), lr=3e-2, weight_decay=0.1)
for it, batch in tqdm(enumerate(loop(dataloader, numIters)), total=numIters):
  if len(batch) == 6:
    x, t, s, y, a, nq = batch
  else:
    x, t, s, y, nq = batch
  lr = pf(it)
  if it % 10 == 0:
    for pg in opt.param_groups:
      pg['lr'] = lr

  yhat = model(x, t, nq) * s
  if kIncludePiecemaps:
    yhat = yhat + pmModel(a, t)
  y = y * s
  loss, error, cpLoss = cross_entropy_loss_fn(nn.functional.softmax(yhat, 1), y)
  loss = loss * (1 - kAlpha)
  loss = loss + score_loss_fn(yhat, y) * kAlpha

  # stdhat = nn.functional.leaky_relu(stdModel(x)).squeeze()
  # loss = loss + score_loss_fn(stdhat, torch.abs(soft_clip(yhat) - soft_clip(y)))

  if kIncludePiecemaps:
    loss = loss + 10.0 * ((pmModel.w["early"].weight.reshape(PmEarlySampleSizeTh.shape)**2) / torch.sqrt(PmEarlySampleSizeTh + 1.0)).mean()
    loss = loss + 10.0 * ((pmModel.w["late"].weight.reshape(PmLateSampleSizeTh.shape)**2) / torch.sqrt(PmEarlySampleSizeTh + 1.0)).mean()

  w1 = pca.slope_backward(model.w['early'].weight)
  w2 = pca.slope_backward(model.w['late'].weight)
  w3 = pca.slope_backward(model.w['clipped'].weight)
  loss = loss + torch.abs(w1 * kEarlyBlacklist).mean() * 10.0
  for w, threshold in [(w1, 10.0), (w2, 10.0), (w3, 10.0)]:
    loss = loss + torch.relu(w - threshold).mean() * 10
    loss = loss + torch.relu(-threshold - w).mean() * 10
    loss = loss + ((w[0, varnames.index('OUR_PAWNS'  )] + w[0, varnames.index('THEIR_PAWNS'  )])**2).mean() * 1
    loss = loss + ((w[0, varnames.index('OUR_KNIGHTS')] + w[0, varnames.index('THEIR_KNIGHTS')])**2).mean() * 1
    loss = loss + ((w[0, varnames.index('OUR_BISHOPS')] + w[0, varnames.index('THEIR_BISHOPS')])**2).mean() * 1
    loss = loss + ((w[0, varnames.index('OUR_ROOKS'  )] + w[0, varnames.index('THEIR_ROOKS'  )])**2).mean() * 1
    loss = loss + ((w[0, varnames.index('OUR_QUEENS' )] + w[0, varnames.index('THEIR_QUEENS' )])**2).mean() * 1


  # loss = loss + torch.abs(torch.relu(-(w1 + w3)) * kPositiveList).mean()
  # loss = loss + torch.abs(torch.relu(-(w2 + w3)) * kPositiveList).mean()
  # loss = loss + torch.abs(torch.relu(-(w2 + w3 + w4)) * kPositiveList).mean()
  # loss = loss + torch.abs(torch.relu(w1 + w3) * kNegativeList).mean()
  # loss = loss + torch.abs(torch.relu(w2 + w3) * kNegativeList).mean()
  # loss = loss + torch.abs(torch.relu(w2 + w3 + w4) * kNegativeList).mean()

  if "scale" in model.w:
    loss = loss + (model.w["scale"](x)**2).mean() * 0.2

  opt.zero_grad()
  loss.backward()
  opt.step()

  metrics['loss'].append(float(loss))
  metrics['error'].append(error)
  metrics['cpLoss'].append(cpLoss)

  if it % 50 == 0:
    n = len(metrics['loss'][-4:])
    print('lr = %.4f; loss = %.4f; error = %.4f; cpLoss = %.4f' % (lr, sum(metrics['loss'][-4:]) / n, sum(metrics['error'][-4:]) / n, sum(metrics['cpLoss'][-4:]) / n))

torch.save(model.state_dict(), 'model.pth')
if kIncludePiecemaps:
  torch.save(pmModel.state_dict(), 'pmModel.pth')

for k in model.w:
  w = model.w[k].weight.detach().numpy()
  w = pca.slope_backward(w)
  with torch.no_grad():
    model.w[k] = nn.Linear(w.shape[1], w.shape[0])
    model.w[k].weight *= 0
    model.w[k].weight += torch.tensor(w.astype(np.float32))

W = {}
for k in model.w:
  W[k] = model.w[k].weight.detach().numpy().squeeze()

stdw = pca.slope_backward(stdModel.weight).detach().numpy().squeeze()
for i in range(len(varnames)):
  print(lpad(round(stdw[i] * 100), 5) + '  // ' + varnames[i])

K = ['early', 'late', 'clipped']

print(' '.join(K))
for i in range(len(varnames)):
  print(' '.join(lpad(round(W[k][i] * 100), 5) for k in K) + '  ' + varnames[i])

for k in K:
  print(lpad(round(float(model.w[k].bias.detach().numpy() * 100)), 6) + f'  // {k} bias')
  for i in range(W[k].shape[0]):
    print(lpad(round(W[k][i] * 100), 6) + f'  // {k} {varnames[i]}')

kPieceName = 'PNBRQK'

w = np.round(pmModel.w['early'].weight.reshape(6, 8, 8).detach().numpy() * 100).astype(np.int64)
w = cat([np.zeros((1, 8, 8), dtype=np.int64), w], 0)
for i, row in enumerate(w.reshape(-1, 8)):
  if i % 8 == 0:
    print('// ' + ('?' + kPieceName)[i//8])
  print(' '.join(lpad(x, 5) for x in row))

w = np.round(pmModel.w['late'].weight.reshape(6, 8, 8).detach().numpy() * 100).astype(np.int64)
w = cat([np.zeros((1, 8, 8), dtype=np.int64), w], 0)
for i, row in enumerate(w.reshape(-1, 8)):
  if i % 8 == 0:
    print('// ' + ('?' + kPieceName)[i//8])
  print(' '.join(lpad(x, 5) for x in row))
