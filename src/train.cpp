// Production:
// g++ src/main.cpp src/game/*.cpp -std=c++20 -O3 -DNDEBUG -o main
// 
// Debug:
// g++ src/main.cpp src/game/*.cpp -std=c++20 -rdynamic -g1
//
// To Generate train.txt
// ./a.out mode printvec fens eval.txt > ./train.txt
//
// To evaluate changes
// ./a.out mode evaluate fens eval.txt depth 2

#include <cassert>
#include <cstdint>
#include <ctime>
#include <cstdlib>

#include <algorithm>
#include <chrono>
#include <deque>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "game/geometry.h"
#include "game/utils.h"
#include "game/Position.h"
#include "game/movegen.h"
#include "game/movegen/sliding.h"
#include "game/Evaluator.h"
#include "game/search.h"

const size_t numPartsOfFen = 6;

using namespace ChessEngine;

struct TrainingPosition {
  TrainingPosition(const std::vector<std::string>& parts) {
      this->fen = join(std::vector<std::string>(parts.begin(), parts.begin() + numPartsOfFen), " ");

      for (size_t i = numPartsOfFen; i < parts.size(); i += 2) {
        std::pair<std::string, Evaluation> pair;
        pair.first = parts[i];
        try {
          pair.second = std::stoi(parts[i + 1]);
        } catch (std::invalid_argument& err) {
          std::cout << "failed to parse \"" << parts[i + 1] << "\" into integer" << std::endl;
          throw err;
        }
        this->moves.push_back(pair);
      }
  }
  std::string fen;
  std::vector<std::pair<std::string, Evaluation>> moves;
};

template<Color TURN>
uint16_t _get_cp_diff(Thinker& thinker, Position *pos, const TrainingPosition& trainingPosition, const size_t maxDepth) {
  // thinker.reset_stuff();
  // SearchResult<Color::WHITE> result(0, kNullMove);
  // for (size_t depth = 1; depth <= maxDepth; ++depth) {
  //   if (pos.turn_ == Color::WHITE) {
  //     result = thinker.search<Color::WHITE, SearchTypeRoot>(
  //       &pos, depth, 0, kMinEval, kMaxEval, RecommendedMoves(), 0, 0
  //     );
  //   } else {
  //     result = flip(thinker.search<Color::BLACK, SearchTypeRoot>(
  //       &pos, depth, 0, kMinEval, kMaxEval, RecommendedMoves(), 0, 0
  //     ));
  //   }
  // }

  constexpr Color opposingColor = opposite_color<TURN>();

  ExtMove moves[kMaxNumMoves];
  ExtMove *end = compute_legal_moves<TURN>(pos, &moves[0]);

  SearchResult<TURN> bestChild(kMissingKing, kNullMove);
  for (ExtMove *move = moves; move < end; ++move) {
    make_move<TURN>(pos, move->move);
    SearchResult<TURN> childResult = flip(thinker.qsearch<opposingColor>(pos, 0, -kMaxEval, -bestChild.score));
    if (childResult.score > bestChild.score) {
      bestChild.score = childResult.score;
      bestChild.move = move->move;
    }
    undo<TURN>(pos);
  }

  const int32_t bestStockfishScore = trainingPosition.moves[0].second;
  Evaluation scoreIfMissing = 0;
  for (size_t i = 0; i < trainingPosition.moves.size(); ++i) {
    int32_t b = trainingPosition.moves[i].second;
    Evaluation e = std::min<Evaluation>(std::abs(bestStockfishScore - b), 500);
    if (trainingPosition.moves[i].first == bestChild.move.uci()) {
      return e;
    }
    scoreIfMissing = std::max(scoreIfMissing, e);
  }
  return scoreIfMissing;
}

uint16_t get_cp_diff(Thinker& thinker, const TrainingPosition& trainingPosition, const size_t maxDepth) {
  Position pos(trainingPosition.fen);
  if (pos.turn_ == Color::WHITE) {
    return _get_cp_diff<Color::WHITE>(thinker, &pos, trainingPosition, maxDepth);
  } else {
    return _get_cp_diff<Color::BLACK>(thinker, &pos, trainingPosition, maxDepth);
  }
}

struct Shuffler {
  Shuffler() : n(0) {}
  template<class T>
  void shuffle(std::vector<T>& vec) {
    std::shuffle(vec.begin(), vec.end(), std::default_random_engine(++n));
  }
  size_t n;
};
Shuffler gShuffler;

struct Trainer {
  Trainer(std::vector<TrainingPosition>& trainingPositions) : trainingPositions(trainingPositions) {
    // Clearing a large cache between trials is very slow, so we make the cache smaller.
    this->thinker.set_cache_size(1);
    this->thinker.load_weights_from_file("weights.txt");
  }

  std::pair<double, double> evaluate(const size_t batchSize) {
    const size_t depth = 1;

    uint64_t t0 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

    uint64_t num = 0;
    uint64_t num2 = 0;
    uint64_t den = 0;
    for (size_t i = 0; i < std::min(batchSize, this->trainingPositions.size()); ++i) {
      uint64_t x = get_cp_diff(this->thinker, this->trainingPositions[i], depth);
      num += x;
      num2 += x * x;
      den += 1;
    }

    uint64_t t1 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    std::cout << t1 - t0 << "ms" << std::endl;

    double avg = double(num) / double(den);
    double stdvar = (double(num2) / double(den) - avg * avg) / den;

    return std::make_pair(avg, std::sqrt(stdvar));
  }

  void step() {
    int stepsize = rand() % 2 ? 50 : -50;
    const size_t batchSize = 1000;

    gShuffler.shuffle(this->trainingPositions);

    std::pair<double, double> before = this->evaluate(batchSize);

    size_t featureIdx = rand() % EF::NUM_EVAL_FEATURES;
    this->thinker.evaluator.clippedW[featureIdx] += stepsize;
    std::pair<double, double> after = this->evaluate(batchSize);

    double diff = after.first - before.first;
    double stderr = std::sqrt(after.second * 2 + before.second * 2);

    std::cout << EFSTR[featureIdx] << " += " << stepsize << std::endl;
    std::cout << diff / stderr << std::endl;

    if (diff / stderr < 2.0) {
      this->thinker.evaluator.clippedW[featureIdx] -= stepsize;
    } else {
      this->thinker.save_weights_to_file("www.txt");
    }
  }

  Thinker thinker;
  std::vector<TrainingPosition>& trainingPositions;
};

int main(int argc, char *argv[]) {
  initialize_geometry();
  initialize_zorbrist();
  initialize_movegen();

  // TODO: load analyzed positions
  // "<FEN> <uci1> <score1> <uci2> <score2> <uci3> <score3>"

  std::vector<TrainingPosition> trainingPositions;
  {
    const std::string filename = "train.txt";
    std::ifstream myfile;
    myfile.open(filename);
    if (!myfile.is_open()) {
      std::cout << "Error opening file \"" << filename << "\"" << std::endl;
      return 1;
    }

    std::string line;
    getline(myfile, line);
    while (true) {
      std::vector<std::string> parts = split(line, ' ');
      if (parts.size() < numPartsOfFen + 4) {
        break;
      }
      if ((parts.size() - numPartsOfFen) % 2 != 0) {
        break;
      }
      trainingPositions.push_back(TrainingPosition(parts));

      getline(myfile, line);
    }

    std::cout << "Loaded " << trainingPositions.size() << " positions" << std::endl;

    myfile.close();
  }

  Trainer trainer(trainingPositions);
  // trainer.thinker.evaluator.zero_();
  // trainer.thinker.pieceMaps.zero_();

  while (true) {
    trainer.step();
  }


  return 0;
}
