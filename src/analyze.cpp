
// sqlite3 positions.db "select * from positions limit 4" > /tmp/positions.txt

/*

import numpy as np
Y = np.frombuffer(open('/tmp/y.bin', 'rb').read(), dtype=np.int16).reshape(-1, 3)
X = np.frombuffer(open('/tmp/x.bin', 'rb').read(), dtype=np.int8).reshape(Y.shape[0], -1)
Y = (Y[:,0] + Y[:,1] * 0.5 + 1.5) / (Y.sum(1) + 3)
logit = lambda x: np.log(x / (1 - x))
Y = logit(Y)
time = X[:,54]

*/

#include <cassert>
#include <cstdint>
#include <ctime>
#include <cstdlib>

#include <algorithm>
#include <deque>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "game/Position.h"
#include "game/Evaluator.h"
#include "game/string_utils.h"

using namespace ChessEngine;

Evaluator gEvaluator;

template<Color US>
void handle(Position &pos, std::fstream& fout) {
  gEvaluator.score<US>(pos);
  int8_t A[EF::NUM_EVAL_FEATURES];
  for (size_t i = 0; i < EF::NUM_EVAL_FEATURES; ++i) {
    A[i] = gEvaluator.features[i];
  }
  fout.write((char *)(&A[0]), sizeof(A));
}

int main(int argc, char *argv[]) {
  if (argc != 4) {
    std::cerr << "Usage: " << argv[0] << " <input> <output> <y_output>" << std::endl;
    return 1;
  }

  std::fstream fin(argv[1], std::ios::in);
  if (!fin.is_open()) {
    std::cerr << "Could not open file " << argv[1] << std::endl;
    return 1;
  }

  std::fstream fout(argv[2], std::ios::out | std::ios::binary);
  if (!fout.is_open()) {
    std::cerr << "Could not open file " << argv[2] << std::endl;
    return 1;
  }

  std::fstream yout(argv[3], std::ios::out | std::ios::binary);
  if (!fout.is_open()) {
    std::cerr << "Could not open file " << argv[3] << std::endl;
    return 1;
  }

  std::string line;
  size_t counter = 0;
  while (std::getline(fin, line)) {
    std::vector<std::string> tokens = split(line, '|');
    if (tokens.size() != 4) {
      break;
    }
    if ((++counter) % 100000 == 0) {
      std::cout << counter / 1000 << "k" << std::endl;
    }

    Position pos(tokens[0]);
    int16_t wins, losses;
    int16_t draws = std::stoi(tokens[2]);
    if (pos.turn_ == Color::WHITE) {
      wins = std::stoi(tokens[1]);
      losses = std::stoi(tokens[3]);
    } else {
      wins = std::stoi(tokens[3]);
      losses = std::stoi(tokens[1]);
    }

    if (pos.turn_ == Color::WHITE) {
      handle<Color::WHITE>(pos, fout);
    } else {
      handle<Color::BLACK>(pos, fout);
    }
    yout.write((char *)&wins, sizeof(wins));
    yout.write((char *)&draws, sizeof(draws));
    yout.write((char *)&losses, sizeof(losses));
  }

  return 0;
}