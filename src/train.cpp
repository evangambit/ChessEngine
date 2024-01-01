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
  TrainingPosition(const std::vector<std::string>& fenParts, const std::vector<std::string>& parts) {
      this->fen = join(fenParts, " ");

      for (size_t i = 0; i < parts.size(); i += 2) {
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

template<class A, class B>
std::ostream& operator<<(std::ostream& stream, const std::pair<A, B>& pair) {
  return stream << "(" << pair.first << ", " << pair.second << ")" << std::endl;
}

std::ostream& operator<<(std::ostream& stream, const std::vector<std::pair<std::string, Evaluation>>& vec) {
  if (vec.size() == 0) {
    return stream << "{}";
  }
  stream << "{" << vec[0];
  for (size_t i = 1; i < vec.size(); ++i) {
    stream << ", " << vec[i];
  }
  return stream << "}";
}

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

  Thread thread(
    0,
    *pos,
    thinker.evaluator,
    compute_legal_moves_set(pos)
  );

  SearchResult<TURN> bestChild(kMissingKing, kNullMove);
  for (ExtMove *move = moves; move < end; ++move) {
    make_move<TURN>(pos, move->move);
    SearchResult<TURN> childResult = flip(qsearch<opposingColor>(&thinker, &thread, 0, 0, -kMaxEval, -bestChild.score));
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

  void evaluate(int32_t *A, const size_t batchSize, int32_t s) {
    const size_t depth = 1;

    uint64_t t0 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

    for (size_t i = 0; i < std::min(batchSize, this->trainingPositions.size()); ++i) {
      A[i] += get_cp_diff(this->thinker, this->trainingPositions[i], depth) * s;
    }

    uint64_t t1 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    // std::cout << t1 - t0 << "ms" << std::endl;
  }

  void step() {
    int stepsize = rand() % 2 ? 50 : -50;
    const size_t batchSize = 20000;

    gShuffler.shuffle(this->trainingPositions);

    int32_t *A = new int32_t[batchSize];
    std::fill_n(A, batchSize, 0);

    this->evaluate(A, batchSize, 1);

    size_t featureIdx = rand() % (EF::THEIR_QUEENS + 1);
    this->thinker.evaluator.earlyW[featureIdx] += stepsize;

    this->evaluate(A, batchSize, -1);

    double a = 0.0;
    double a2 = 0.0;
    for (size_t i = 0; i < batchSize; ++i) {
      a += A[i];
      a2 += A[i] * A[i];
    }

    a /= batchSize;
    a2 /= batchSize;

    double stderr = std::sqrt((a2 - a) / (batchSize - 1));

    std::cout << EFSTR[featureIdx] << " += " << stepsize << std::endl;
    std::cout << a << " / " << stderr << " = " << a / stderr << std::endl;

    if (stderr < 1e-6 || a / stderr < 1.0) {
      this->thinker.evaluator.earlyW[featureIdx] -= stepsize;
    } else {
      this->thinker.save_weights_to_file("www.txt");
    }

    delete[] A;
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
      std::vector<std::string> parts = split(line, ':');
      if (parts.size() != 2) {
        break;
      }
      std::vector<std::string> fenParts = split(parts[0], ' ');
      parts = split(parts[1], ' ');
      if (fenParts.size() != numPartsOfFen) {
        break;
      }
      if (parts.size() % 2 != 0) {
        break;
      }
      trainingPositions.push_back(TrainingPosition(fenParts, parts));

      getline(myfile, line);
    }

    std::cout << "Loaded " << trainingPositions.size() << " positions" << std::endl;

    myfile.close();
  }

  Trainer trainer(trainingPositions);
  trainer.thinker.load_weights_from_file("weights.txt");
  // trainer.thinker.evaluator.zero_();
  // trainer.thinker.pieceMaps.zero_();

  while (true) {
    trainer.step();
  }


  return 0;
}
