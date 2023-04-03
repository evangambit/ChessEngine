// Production:
// g++ src/selfplay.cpp src/game/*.cpp -std=c++20 -O3 -DNDEBUG -o selfplay -o selfplay
// 
// Debug:
// g++ src/selfplay.cpp src/game/*.cpp -std=c++20 -std=c++20 -rdynamic -g1 -o selfplay

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

#include "game/geometry.h"
#include "game/utils.h"
#include "game/Position.h"
#include "game/movegen.h"
#include "game/movegen/sliding.h"
#include "game/Evaluator.h"
#include "game/search.h"

using namespace ChessEngine;

Thinker thinker1;
Thinker thinker2;

bool make_move(Thinker *thinker, Position *pos, size_t nodeLimit) {
  thinker->reset_stuff();
  SearchResult<Color::WHITE> results(Evaluation(0), kNullMove);
  for (size_t depth = 1; depth < 99; ++depth) {
    results = thinker->search(pos, Depth(depth), results);
    if (thinker->nodeCounter >= nodeLimit) {
      break;
    }
  }
  if (results.move == kNullMove) {
    return false;
  }
  if (pos->turn_ == Color::WHITE) {
    make_move<Color::WHITE>(pos, results.move);
  } else {
    make_move<Color::BLACK>(pos, results.move);
  }
  return true;
}

bool is_material_draw(const Position& pos) {
  if (pos.pieceBitboards_[ColoredPiece::WHITE_PAWN]) {
    return false;
  }
  if (pos.pieceBitboards_[ColoredPiece::BLACK_PAWN]) {
    return false;
  }
  if (pos.pieceBitboards_[ColoredPiece::WHITE_QUEEN]) {
    return false;
  }
  if (pos.pieceBitboards_[ColoredPiece::BLACK_QUEEN]) {
    return false;
  }
  if (pos.pieceBitboards_[ColoredPiece::WHITE_ROOK]) {
    return false;
  }
  if (pos.pieceBitboards_[ColoredPiece::BLACK_ROOK]) {
    return false;
  }
  bool canAnyoneWin = false;
  canAnyoneWin |= std::popcount(pos.pieceBitboards_[ColoredPiece::WHITE_BISHOP]) > 1;
  canAnyoneWin |= std::popcount(pos.pieceBitboards_[ColoredPiece::BLACK_BISHOP]) > 1;
  canAnyoneWin |= std::popcount(pos.pieceBitboards_[ColoredPiece::WHITE_KNIGHT]) > 2;
  canAnyoneWin |= std::popcount(pos.pieceBitboards_[ColoredPiece::BLACK_KNIGHT]) > 2;
  canAnyoneWin |= std::popcount(pos.pieceBitboards_[ColoredPiece::WHITE_BISHOP]) == 1 && std::popcount(pos.pieceBitboards_[ColoredPiece::WHITE_KNIGHT]) > 0;
  canAnyoneWin |= std::popcount(pos.pieceBitboards_[ColoredPiece::BLACK_BISHOP]) == 1 && std::popcount(pos.pieceBitboards_[ColoredPiece::BLACK_KNIGHT]) > 0;
  return !canAnyoneWin;
}

int play(Thinker *thinkerWhite, Thinker *thinkerBlack, const std::string& fen, size_t nodeLimit) {
  Position pos(fen);
  while (true) {
    Thinker *thinker;
    if (pos.turn_ == Color::WHITE) {
      thinker = thinkerWhite;
    } else {
      thinker = thinkerBlack;
    }
    if (!make_move(thinker, &pos, nodeLimit)) {
      break;
    }
    if (pos.is_draw() || is_material_draw(pos)) {
      break;
    }
  }
  if (!pos.pieceBitboards_[ColoredPiece::WHITE_KING]) {
    throw std::runtime_error("missing white king");
  }
  if (!pos.pieceBitboards_[ColoredPiece::BLACK_KING]) {
    throw std::runtime_error("missing black king");
  }
  if (pos.is_draw() || is_material_draw(pos)) {
    return 0;
  }
  if (can_enemy_attack<Color::BLACK>(pos, lsb(pos.pieceBitboards_[ColoredPiece::BLACK_KING]))) {
    return 1;
  } else if (can_enemy_attack<Color::WHITE>(pos, lsb(pos.pieceBitboards_[ColoredPiece::WHITE_KING]))) {
    return -1;
  }
  return 0;
}

int main(int argc, char *argv[]) {
  initialize_geometry();
  initialize_zorbrist();
  initialize_sliding();

  std::vector<std::string> args;
  for (size_t i = 1; i < argc; ++i) {
    args.push_back(argv[i]);
  }

  Thinker thinker1;
  Thinker thinker2;

  std::vector<std::string> fens;
  bool weightsLoaded = false;
  size_t nodeLimit = 5000;

  while (args.size() > 0) {
    if (args.size() >= 7 && args[0] == "fen") {
      std::vector<std::string> fenVec(args.begin() + 1, args.begin() + 7);
      args = std::vector<std::string>(args.begin() + 7, args.end());
      fens.push_back(join(fenVec, " "));
    } else if (args.size() >= 2 && args[0] == "nodes") {
      nodeLimit = std::stoi(args[1]);
      if (nodeLimit < 1) {
        throw std::invalid_argument("");
        exit(1);
      }
      args = std::vector<std::string>(args.begin() + 2, args.end());
    } else if (args.size() >= 3 && args[0] == "weights") {
      thinker1.evaluator.load_weights_from_file(args[1]);
      thinker2.evaluator.load_weights_from_file(args[2]);
      weightsLoaded = true;
      args = std::vector<std::string>(args.begin() + 3, args.end());
    } else {
      std::cout << "Cannot understand argument \"" << args[0] << "\"" << std::endl;
      return 1;
    }
  }

  if (fens.size() == 0) {
    fens.push_back("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
  }

  if (!weightsLoaded) {
    std::cout << "missing weights files" << std::endl;
    return 1;
  }

  // Prints 1 if thinker1 wins
  // Prints -1 if thinker2 wins
  for (const auto& fen : fens) {
    std::cout << play(&thinker1, &thinker2, fen, nodeLimit) << std::endl;
    std::cout << -play(&thinker2, &thinker1, fen, nodeLimit) << std::endl;
  }

  return 0;
}
