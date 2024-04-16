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

  // Do a preliminary search at depth=1 to guarantee a result.
  GoCommand goCommand;
  goCommand.pos = *pos;
  goCommand.depthLimit = 1;
  goCommand.moves = compute_legal_moves_set(pos);
  SearchResult<Color::WHITE> results = search(thinker, goCommand, nullptr, [](Position *position, VariationHead<Color::WHITE> results, size_t depth, double secs) {});

  // Do the "real" search with the given node limit.
  goCommand.nodeLimit = nodeLimit;
  goCommand.depthLimit = 32;
  results = search(thinker, goCommand, nullptr, [](Position *position, VariationHead<Color::WHITE> results, size_t depth, double secs) {});

  if (results.move == kNullMove) {
    return false;
  }
  if (goCommand.moves.count(results.move.uci()) == 0) {
    std::cout << "\"" << results.move.uci() << "\" not a legal move" << std::endl;
    exit(1);
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

int play(Thinker *thinkerWhite, Thinker *thinkerBlack, const std::string& fen, const size_t nodeLimit, const size_t maxMoves) {
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
    if (isStalemate(&pos, thinkerWhite->evaluator)) {
      break;
    }
    if (isCheckmate(&pos)) {
      break;
    }
    if (pos.history_.size() >= maxMoves) {
      break;
    }
  }
  if (!pos.pieceBitboards_[ColoredPiece::WHITE_KING]) {
    throw std::runtime_error("missing white king");
  }
  if (!pos.pieceBitboards_[ColoredPiece::BLACK_KING]) {
    throw std::runtime_error("missing black king");
  }
  if (pos.history_.size() >= maxMoves) {
    return 0;
  }
  if (isStalemate(&pos, thinkerWhite->evaluator)) {
    return 0;
  }
  if (isCheckmate(&pos)) {
    if (pos.turn_ == Color::WHITE) {
      return -1;
    } else {
      return 1;
    }
  }
  return 0;
}

int main(int argc, char *argv[]) {
  initialize_geometry();
  initialize_zorbrist();
  initialize_movegen();

  std::vector<std::string> args;
  for (size_t i = 1; i < argc; ++i) {
    args.push_back(argv[i]);
  }

  Thinker thinker1;
  Thinker thinker2;

  std::vector<std::string> fens;
  bool weightsLoaded = false;
  size_t nodeLimit = 5000;
  size_t maxMoves = 9999999;

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
      thinker1.load_weights_from_file(args[1]);
      thinker2.load_weights_from_file(args[2]);
      weightsLoaded = true;
      args = std::vector<std::string>(args.begin() + 3, args.end());
    } else if (args.size() >= 2 && args[0] == "maxmoves") {
      maxMoves = std::stoi(args[1]);
      args = std::vector<std::string>(args.begin() + 2, args.end());
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
    // thinker1.clear_tt(); thinker2.clear_tt();
    std::cout << play(&thinker1, &thinker2, fen, nodeLimit, maxMoves) << std::endl;
    // thinker1.clear_tt(); thinker2.clear_tt();
    std::cout << -play(&thinker2, &thinker1, fen, nodeLimit, maxMoves) << std::endl;
  }

  return 0;
}
