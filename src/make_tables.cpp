#include "game/geometry.h"
#include "game/utils.h"
#include "game/Position.h"
#include "game/movegen.h"
#include "game/movegen/sliding.h"
#include "game/Evaluator.h"
#include "game/SquareEmbeddings.h"

using namespace ChessEngine;

void process(std::string line, std::ostream &outfile) {
  Position pos(line);

  int32_t pieceMaps[7 * 64];
  std::fill_n(pieceMaps, 7 * 64, 0);
  int sign = pos.turn_ == Color::WHITE ? 1 : -1;
  for (int y = 0; y < 8; ++y) {
    for (int x = 0; x < 8; ++x) {
      ColoredPiece cp = pos.tiles_[y * 8 + x];
      if (cp2color(cp) == Color::BLACK) {
        pieceMaps[(cp + ColoredPiece::WHITE_PAWN - ColoredPiece::BLACK_PAWN) * 64 + (7 - y) * 8 + x] -= sign;
      } else {
        pieceMaps[cp * 64 + y * 8 + x] += sign;
      }
    }
  }
  std::fill_n(pieceMaps, 64, 0);  // zero out "no piece" squares
  for (int i = 0; i < 7 * 64; ++i) {
    if (pieceMaps[i] < -1 || pieceMaps[i] > 1) {
      throw std::runtime_error("oops");
    }
    outfile << pieceMaps[i] + 1;
  }
}

int main(int argc, char *argv[]) {
  initialize_geometry();
  initialize_zorbrist();
  initialize_movegen();

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
