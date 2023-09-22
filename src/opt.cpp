// g++ src/opt.cpp src/game/*.cpp -std=c++20 -O3 -DNDEBUG -o opt

#include <cassert>
#include <cstdint>
#include <ctime>
#include <cstdlib>

#include <string>
#include <vector>
#include <iostream>
#include <algorithm>
#include <random>
#include <fstream>
#include <deque>

#include "game/geometry.h"
#include "game/utils.h"
#include "game/Position.h"
#include "game/movegen.h"
#include "game/movegen/sliding.h"
#include "game/Evaluator.h"
#include "game/search.h"

using namespace ChessEngine;

std::chrono::time_point<std::chrono::steady_clock> current_time() {
  return std::chrono::steady_clock::now();
}

int64_t elapsed_ms(std::chrono::time_point<std::chrono::steady_clock> start, std::chrono::time_point<std::chrono::steady_clock> end) {
  std::chrono::duration<double> delta = end - start;
  return std::chrono::duration_cast<std::chrono::milliseconds>(delta).count();
}

double sigmoid(double x) {
  x /= 200.0;
  return 1.0 / (1.0 + std::exp(-x));
}

enum BoundType {
  LOWER,
  UPPER,
  EXACT
};

struct SimpleCacheResult {
  uint64_t hash;
  Evaluation eval;
  Move bestMove;
  Depth depth;
  BoundType boundType;
};

constexpr uint64_t kSimpleCacheSize = 8192;
struct SimpleSearchThread : public Thread {
  SimpleSearchThread(uint64_t id, const Position& pos, const Evaluator& e, const std::unordered_set<std::string>& moves)
  : Thread(id, pos, e, moves) {
    this->reset_cache();
  }
  SimpleCacheResult cache[kSimpleCacheSize];
  void reset_cache() {
    for (uint64_t i = 0; i < kSimpleCacheSize; ++i) {
      cache[i].bestMove = kNullMove;
      cache[i].depth = 0;
    }
  }
};

template<Color TURN>
static SearchResult<TURN> simple_search(
  Thinker *thinker,
  SimpleSearchThread *thread,
  const Depth depthRemaining,
  const Depth plyFromRoot,
  Evaluation alpha,
  Evaluation beta) {
  if (depthRemaining == 0) {
    return qsearch<TURN>(thinker, thread, 0, plyFromRoot, alpha, beta);
  }

  constexpr Color opposingColor = opposite_color<TURN>();
  constexpr ColoredPiece moverKing = coloredPiece<TURN, Piece::KING>();

  ExtMove moves[kMaxNumMoves];
  ExtMove *movesEnd;
  if (plyFromRoot == 0) {
    movesEnd = compute_legal_moves<TURN>(&thread->pos, moves);
  } else {
    movesEnd = compute_moves<TURN, MoveGenType::ALL_MOVES>(thread->pos, moves);
  }

  if (movesEnd - moves == 0) {
    const bool inCheck = can_enemy_attack<TURN>(thread->pos, lsb(thread->pos.pieceBitboards_[moverKing]));
    if (inCheck) {
      return SearchResult<TURN>(kCheckmate + plyFromRoot, kNullMove);
    } else {
      return SearchResult<TURN>(Evaluation(0), kNullMove);
    }
  }

  SimpleCacheResult& cr = thread->cache[thread->pos.hash_ % kSimpleCacheSize];
  if (cr.depth >= depthRemaining && cr.hash == thread->pos.hash_) {
    return SearchResult<TURN>(cr.eval, cr.bestMove);
  }

  for (ExtMove *move = moves; move < movesEnd; ++move) {
    move->score = kMoveOrderPieceValues[move->capture];
    move->score -= kMoveOrderPieceValues[move->piece];
    if (move->move == cr.bestMove) {
      move->score += 9999;
    }
  }

  // avg=10.6166 std=0.1013 val=185 delta=0 nodes=31120k 8539ms
  // avg=10.6166 std=0.1013 val=185 delta=0 nodes=31139k 8628ms

  std::sort(moves, movesEnd, [](ExtMove a, ExtMove b) {
    return a.score > b.score;
  });

  SearchResult<TURN> r(
    kMinEval + 1,
    kNullMove
  );

  for (ExtMove *move = moves; move < movesEnd; ++move) {
    make_move<TURN>(&thread->pos, move->move);
    SearchResult<TURN> a = flip(
      simple_search<opposingColor>(thinker, thread, depthRemaining - 1, plyFromRoot + 1, -beta, -alpha)
    );
    undo<TURN>(&thread->pos);
    if (a.score > r.score) {
      r.score = a.score;
      r.move = move->move;
      alpha = std::max(alpha, a.score);
      if (a.score >= beta) {
        break;
      }
    }
  }

  if (depthRemaining > cr.depth) {
    cr.eval = r.score;
    cr.bestMove = r.move;
    cr.depth = depthRemaining;
    cr.hash = thread->pos.hash_;
  }

  return r;
}

struct Datapoint {
  std::string fen;
  std::vector<std::string> ucis;
  std::vector<Evaluation> evals;
};

struct Result {
  double avgCpLoss;
  double stdCpLoss;
  uint64_t numNodes;
};

template<Color TURN>
std::string evaluate(SimpleSearchThread *threadObj, Thinker *thinker, int depth) {
  SearchResult<TURN> result = simple_search<TURN>(
    thinker,
    threadObj,
    1,
    0,
    kMinEval,
    kMaxEval
  );
  if (depth > 1) {
    result = simple_search<TURN>(
      thinker,
      threadObj,
      depth,
      0,
      kMinEval,
      kMaxEval
    );
  }
  return result.move.uci();
}

Result evaluate(Thinker *thinker, const std::vector<Datapoint>& datapoints, int depth) {
  double num1 = 0.0;
  double num2 = 0.0;
  double den = 0.0;
  uint64_t nodes = 0;
  for (const auto& datapoint : datapoints) {
    Position pos(datapoint.fen);
    SimpleSearchThread threadObj(0, pos, thinker->evaluator, compute_legal_moves_set(&pos));

    std::string bestPredictedMove;
    if (pos.turn_ == Color::WHITE) {
      bestPredictedMove = evaluate<Color::WHITE>(&threadObj, thinker, depth);
    } else {
      bestPredictedMove = evaluate<Color::BLACK>(&threadObj, thinker, depth);
    }

    nodes += threadObj.nodeCounter;

    auto it = std::find(datapoint.ucis.begin(), datapoint.ucis.end(), bestPredictedMove);
    double delta;
    if (it != datapoint.ucis.end()) {
      delta = sigmoid(datapoint.evals[0]) - sigmoid(datapoint.evals[it - datapoint.ucis.begin()]);
    } else {
      delta = sigmoid(datapoint.evals[0]) - sigmoid(datapoint.evals[datapoint.evals.size() - 1]);
    }
    num1 += delta;
    num2 += delta * delta;
    den += 1.0;
  }

  double avg = num1 / den;
  double stdev = std::sqrt(num2 / den - avg * avg);
  stdev /= std::sqrt(datapoints.size());

  return Result{
    avg * 100.0,
    stdev * 100.0,
    nodes
  };
}

// ./opt $depth $idx delta1 delta2 ...
int main(int argc, char *argv[]) {
  initialize_geometry();
  initialize_zorbrist();
  initialize_movegen();

  std::deque<std::string> args;
  for (size_t i = 1; i < argc; ++i) {
    args.push_back(argv[i]);
  }

  std::ifstream myfile;
  myfile.open("train.txt");
  if (!myfile.is_open()) {
    std::cout << "Failed to open file" << std::endl;
    return 1;
  }

  std::vector<Datapoint> datapoints;

  std::string line;
  getline(myfile, line);
  while (line.size() > 0) {
    std::vector<std::string> parts = split(line, ':');
    if (parts.size() != 2) {
      throw std::runtime_error("weird line \"" + line + "\"");
    }

    datapoints.push_back(Datapoint{});
    Datapoint& datapoint = datapoints.back();
    datapoint.fen = parts[0];
    parts = split(parts[1], ' ');

    for (size_t i = 0; i < parts.size(); i += 2) {
      datapoint.ucis.push_back(parts[i + 0]);
      datapoint.evals.push_back(std::stoi(parts[i + 1]));
    }

    if (datapoints.size() >= 100'000) {
      break;
    }

    getline(myfile, line);
  }

  Thinker thinker;
  thinker.load_weights_from_file("weights.txt");
  int depth = std::stoi(argv[1]);
  int idx = std::stoi(argv[2]);
  const int oldValue = thinker.evaluator.earlyW[idx];

  for (int i = 3; i < argc; ++i) {
    int delta = std::stoi(argv[i]);
    thinker.evaluator.earlyW[idx] = oldValue + delta;

    auto t0 = current_time();
    auto e = evaluate(&thinker, datapoints, depth);
    auto t1 = current_time();

    printf("avg=%.4f std=%.4f val=%i delta=%i nodes=%lluk %llums\n", e.avgCpLoss, e.stdCpLoss, oldValue + delta, delta, e.numNodes / 1000, elapsed_ms(t0, t1));
  }

  return 0;
}
