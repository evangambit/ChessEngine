#include <cassert>
#include <cstdint>
#include <ctime>
#include <cstdlib>
#include <cmath>

#include <algorithm>
#include <deque>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <stdio.h>
#include <execinfo.h>
#include <signal.h>
#include <stdlib.h>
#include <unistd.h>
#include <signal.h>    
#include <stdio.h>
#include <stdlib.h>
#include <execinfo.h>
#include <unistd.h>

#include "game/geometry.h"
#include "game/utils.h"
#include "game/string_utils.h"
#include "game/Position.h"
#include "game/movegen.h"
#include "game/movegen/sliding.h"
#include "game/Evaluator.h"
#include "game/search.h"

using namespace ChessEngine;

Thinker gThinker;

struct Gradients {
  Gradients() {
    earlyW.resize(EF::NUM_EVAL_FEATURES);
    lateW.resize(EF::NUM_EVAL_FEATURES);
    clippedW.resize(EF::NUM_EVAL_FEATURES);
    std::fill_n(&earlyW[0], EF::NUM_EVAL_FEATURES, 0.0);
    std::fill_n(&lateW[0], EF::NUM_EVAL_FEATURES, 0.0);
    std::fill_n(&clippedW[0], EF::NUM_EVAL_FEATURES, 0.0);

    earlyW_2.resize(EF::NUM_EVAL_FEATURES);
    lateW_2.resize(EF::NUM_EVAL_FEATURES);
    clippedW_2.resize(EF::NUM_EVAL_FEATURES);
    std::fill_n(&earlyW_2[0], EF::NUM_EVAL_FEATURES, 0.0);
    std::fill_n(&lateW_2[0], EF::NUM_EVAL_FEATURES, 0.0);
    std::fill_n(&clippedW_2[0], EF::NUM_EVAL_FEATURES, 0.0);

    n = 0;
  }
  std::vector<double> earlyW;
  std::vector<double> lateW;
  std::vector<double> clippedW;

  std::vector<double> earlyW_2;
  std::vector<double> lateW_2;
  std::vector<double> clippedW_2;

  void post_process() {
    for (int i = 0; i < EF::NUM_EVAL_FEATURES; ++i) {
      this->earlyW[i] /= this->n;
      this->clippedW[i] /= this->n;
      this->lateW[i] /= this->n;

      this->earlyW_2[i] /= this->n;
      this->clippedW_2[i] /= this->n;
      this->lateW_2[i] /= this->n;

      this->earlyW_2[i] -= this->earlyW[i] * this->earlyW[i];
      this->clippedW_2[i] -= this->clippedW[i] * this->clippedW[i];
      this->lateW_2[i] -= this->lateW[i] * this->lateW[i];

      this->earlyW_2[i] = std::sqrt(this->earlyW_2[i] / n);
      this->clippedW_2[i] = std::sqrt(this->clippedW_2[i] / n);
      this->lateW_2[i] = std::sqrt(this->lateW_2[i] / n);
    }
  }

  friend std::ostream& operator<<(std::ostream& os, const Gradients& grads) {
    const double threshold = 3.0;
    for (int i = 0; i < EF::NUM_EVAL_FEATURES; ++i) {
      os << "   ##   " << EFSTR[i] << "   ##   " << std::endl;
      if (std::abs(grads.earlyW[i] / grads.earlyW_2[i]) > threshold) {
        os << "early " << grads.earlyW[i] << " +/- " << grads.earlyW_2[i] << std::endl;
      }
      if (std::abs(grads.clippedW[i] / grads.clippedW_2[i]) > threshold) {
        os << "clipped " << grads.clippedW[i] << " +/- " << grads.clippedW_2[i] << std::endl;
      }
      if (std::abs(grads.lateW[i] / grads.lateW_2[i]) > threshold) {
        os << "late " << grads.lateW[i] << " +/- " << grads.lateW_2[i] << std::endl;
      }
    }
    return os;
  }

  uint64_t n;
};

Evaluation forward(Position *pos) {
  Thread thread{0, *pos, gThinker.evaluator, compute_legal_moves_set(pos)};
  if (pos->turn_ == Color::WHITE) {
    return qsearch<Color::WHITE>(&gThinker, &thread, 0, 0, kMinEval, kMaxEval).score;
  } else {
    return qsearch<Color::BLACK>(&gThinker, &thread, 0, 0, kMinEval, kMaxEval).score;
  }
}

template<Color TURN>
void add_grad(Gradients *grads, Position position, const Evaluation y, const int32_t h) {
  Thread thread{0, position, gThinker.evaluator, compute_legal_moves_set(&position)};
  Evaluation yhat = forward(&position);

  for (int i = 0; i < EF::NUM_EVAL_FEATURES; ++i) {
    Evaluation yhat2;
    double g;

    thread.evaluator.earlyW[i] += h;
    yhat2 = forward(&position);
    thread.evaluator.earlyW[i] -= h;
    g = 2.0 * double(yhat - y) * double(yhat2 - yhat) / double(h);
    grads->earlyW[i] += g;
    grads->earlyW_2[i] += g * g;

    thread.evaluator.lateW[i] += h;
    yhat2 = forward(&position);
    thread.evaluator.lateW[i] -= h;
    g = 2.0 * double(yhat - y) * double(yhat2 - yhat) / double(h);
    grads->lateW[i] += g;
    grads->lateW_2[i] += g * g;

    thread.evaluator.clippedW[i] += h;
    yhat2 = forward(&position);
    thread.evaluator.clippedW[i] -= h;
    g = 2.0 * double(yhat - y) * double(yhat2 - yhat) / double(h);
    grads->clippedW[i] += g;
    grads->clippedW_2[i] += g * g;
  }

  grads->n++;
}

struct Triple {
  std::string fen;
  int32_t stockfish;
  int32_t prediction;
  double error;
};

double sign(double x) {
  if (x > 0.0) {
    return 1.0;
  } else if (x < 0.0) {
    return -1.0;
  } else {
    return 0.0;
  }
}

int main(int argc, char *argv[]) {
  initialize_geometry();
  initialize_zorbrist();
  initialize_movegen();

  gThinker.load_weights_from_file("weights.txt");

  std::vector<Triple> lines;
  {
    if (argc < 3) {
      std::cout << "usage: " << argv[0] << " <fens.txt> <n>" << std::endl;
      return 1;
    }
    std::ifstream infile(argv[1]);
    const int64_t n = std::stoi(argv[2]);
    std::string line;
    while (std::getline(infile, line)) {
      if (line == "") {
        continue;
      }

      std::vector<std::string> parts = split(line, ' ');
      if (parts.size() != 7) {
        throw std::runtime_error("");
      }
      Evaluation y = std::stoi(parts[6]);
      parts.pop_back();
      lines.push_back(Triple{join(parts, " "), y, 0});

      if (lines.size() >= n) {
        break;
      }
    }
  }

  for (auto &trip : lines) {
    Position pos(trip.fen);
    trip.prediction = forward(&pos);
  }

  for (size_t i = 0; i < lines.size(); ++i) {
    int wantToBeBigger = 0;
    int wantToBeSmaller = 0;
    for (size_t j = 0; j < lines.size(); ++j) {
      if (lines[j].stockfish > lines[i].stockfish && lines[j].prediction < lines[i].prediction) {
        wantToBeBigger++;
      } else if (lines[j].stockfish < lines[i].stockfish && lines[j].prediction > lines[i].prediction) {
        wantToBeSmaller++;
      }
    }
    lines[i].error = double(wantToBeBigger - wantToBeSmaller) / double(lines.size());
  }

  const int32_t h = 3;
  for (size_t i = 0; i < 10; ++i) {
    std::cout << i << std::endl;
    for (size_t idx = 0; idx < EF::NUM_EVAL_FEATURES; ++idx) {
      gThinker.evaluator.earlyW[idx] += h;
      double a = 0.0;
      double b = 0.0;
      for (const auto &line : lines) {
        Evaluation y = line.stockfish;
        Position position(line.fen);
        double g = 2.0 * double(line.error) * double(forward(&position) - line.prediction) / double(h);
        a += g;
        b += 1;
      }
      gThinker.evaluator.earlyW[idx] -= h;

      const double mean = a / b;
      gThinker.evaluator.earlyW[idx] -= sign(std::round(mean * 10.0));
    }
  }

  gThinker.save_weights_to_file("w.txt");

  return 0;
}
