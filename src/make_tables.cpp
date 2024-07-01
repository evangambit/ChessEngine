#include "game/geometry.h"
#include "game/utils.h"
#include "game/Position.h"
#include "game/movegen.h"
#include "game/movegen/sliding.h"
#include "game/Evaluator.h"
#include "game/SquareEmbeddings.h"
#include "game/nnue.h"

using namespace ChessEngine;

void process(std::string line, std::ostream &outfile) {
  Position pos(line);

  int8_t pieceMaps[NnueFeatures::NF_NUM_FEATURES + 1];
  std::fill_n(pieceMaps, NnueFeatures::NF_NUM_FEATURES, 'a');
  for (int y = 0; y < 8; ++y) {
    for (int x = 0; x < 8; ++x) {
      ColoredPiece cp = pos.tiles_[y * 8 + x];
      if (cp == ColoredPiece::NO_COLORED_PIECE) {
        continue;
      }
      pieceMaps[(cp - 1) * 64 + y * 8 + x] = 'a' + 1;
    }
  }

  pieceMaps[NnueFeatures::NF_IS_WHITE_TURN] = 'a' + (pos.turn_ == Color::WHITE);
  pieceMaps[NnueFeatures::NF_IS_BLACK_TURN] = 'a' + (pos.turn_ == Color::BLACK);

  pieceMaps[NnueFeatures::NF_WHITE_KING_CASTLING] = 'a' + ((pos.currentState_.castlingRights & kCastlingRights_WhiteKing) > 0);
  pieceMaps[NnueFeatures::NF_WHITE_QUEEN_CASTLING] = 'a' + ((pos.currentState_.castlingRights & kCastlingRights_WhiteQueen) > 0);
  pieceMaps[NnueFeatures::NF_BLACK_KING_CASTLING] = 'a' + ((pos.currentState_.castlingRights & kCastlingRights_BlackKing) > 0);
  pieceMaps[NnueFeatures::NF_BLACK_QUEEN_CASTLING] = 'a' + ((pos.currentState_.castlingRights & kCastlingRights_BlackQueen) > 0);

  pieceMaps[NnueFeatures::NF_NUM_WHITE_PAWNS] = 'a' + std::popcount(pos.pieceBitboards_[ColoredPiece::WHITE_PAWN]);
  pieceMaps[NnueFeatures::NF_NUM_WHITE_KNIGHTS] = 'a' + std::popcount(pos.pieceBitboards_[ColoredPiece::WHITE_KNIGHT]);
  pieceMaps[NnueFeatures::NF_NUM_WHITE_BISHOPS] = 'a' + std::popcount(pos.pieceBitboards_[ColoredPiece::WHITE_BISHOP]);
  pieceMaps[NnueFeatures::NF_NUM_WHITE_ROOKS] = 'a' + std::popcount(pos.pieceBitboards_[ColoredPiece::WHITE_ROOK]);
  pieceMaps[NnueFeatures::NF_NUM_WHITE_QUEENS] = 'a' + std::popcount(pos.pieceBitboards_[ColoredPiece::WHITE_QUEEN]);

  pieceMaps[NnueFeatures::NF_NUM_BLACK_PAWNS] = 'a' + std::popcount(pos.pieceBitboards_[ColoredPiece::BLACK_PAWN]);
  pieceMaps[NnueFeatures::NF_NUM_BLACK_KNIGHTS] = 'a' + std::popcount(pos.pieceBitboards_[ColoredPiece::BLACK_KNIGHT]);
  pieceMaps[NnueFeatures::NF_NUM_BLACK_BISHOPS] = 'a' + std::popcount(pos.pieceBitboards_[ColoredPiece::BLACK_BISHOP]);
  pieceMaps[NnueFeatures::NF_NUM_BLACK_ROOKS] = 'a' + std::popcount(pos.pieceBitboards_[ColoredPiece::BLACK_ROOK]);
  pieceMaps[NnueFeatures::NF_NUM_BLACK_QUEENS] = 'a' + std::popcount(pos.pieceBitboards_[ColoredPiece::BLACK_QUEEN]);

  pieceMaps[NnueFeatures::NF_NUM_FEATURES] = 0;

  outfile << pieceMaps;
}

int main(int argc, char *argv[]) {
  initialize_geometry();
  initialize_zorbrist();
  initialize_movegen();

  uint64_t counter = 0;

  if (argc > 1) {
    std::ifstream infile(argv[1]);
    std::string line;
    if (infile.is_open()) {
      while (std::getline(infile, line)) {
        if (line == "") {
          continue;
        }
        process(line, std::cout);
      }
    }
  } else {
    std::string line;
    while (true) {
      std::getline(std::cin, line);
      if (line == "") {
        continue;
      }
      if (line == "exit") {
        break;
      }
      process(line, std::cout);
    }
  }

  return 0;
}
