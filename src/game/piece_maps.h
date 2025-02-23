#ifndef PIECE_MAPS_h
#define PIECE_MAPS_h

#include "geometry.h"
#include "utils.h"

#include <cstdint>

#include <fstream>

namespace ChessEngine {

enum PieceMapType {
  PieceMapTypeEarly = 0,
  PieceMapTypeLate = 1,
  PieceMapTypeCount = 2,
};

constexpr size_t kSizeOfPieceMap = 13 * 64;

struct PieceMaps {
  PieceMaps() {
    for (size_t i = 0; i < PieceMapType::PieceMapTypeCount; ++i) {
      std::fill_n(&pieceMaps[0][0], kSizeOfPieceMap * PieceMapType::PieceMapTypeCount, 0);
    }
  }

  int32_t const *weights(ColoredPiece cp, SafeSquare sq) const;

  void save_weights_to_file(std::ostream& myfile) {
    for (size_t k = 0; k < PieceMapType::PieceMapTypeCount; ++k) {
      for (size_t i = 0; i <= ColoredPiece::WHITE_KING; ++i) {
        myfile << "// " << piece_to_string(Piece(i)) << std::endl;
        for (size_t j = 0; j < 64; ++j) {
          myfile << lpad(pieceMaps[i * 64 + j][k]);
          if (j % 8 == 7) {
            myfile << std::endl;
          }
        }
      }
    }
  }

  void load_weights_from_file(std::istream &myfile) {
    std::string line;
    std::vector<std::string> params;
    for (size_t k = 0; k < PieceMapType::PieceMapTypeCount; ++k) {
      for (size_t i = NO_COLORED_PIECE; i <= WHITE_KING; ++i) {
        getline(myfile, line);
        if (line.substr(0, 3) != "// ") {
          throw std::runtime_error("Unexpected weight format; expected \"// \" but got \"" + line + "\"");
        }
        for (size_t y = 0; y < 8; ++y) {
          getline(myfile, line);
          line = process_with_file_line(line);
          std::vector<std::string> parts = split(line, ' ');
          if (parts.size() != 8) {
            std::cout << line << std::endl;
            throw std::runtime_error("Expected 8 weights in piece-map row but got " + std::to_string(parts.size()));
          }
          for (size_t x = 0; x < 8; ++x) {
            if (i >= ColoredPiece::BLACK_PAWN) {
              continue;
            }
            SafeSquare sq = SafeSquare(y * 8 + x);
            int val = stoi(parts[x]);
            pieceMaps[i * 64 + sq][k] = val;
            if (i != ColoredPiece::NO_COLORED_PIECE) {
              pieceMaps[(i + ColoredPiece::BLACK_PAWN - ColoredPiece::WHITE_PAWN) * 64 + vertical_mirror(sq)][k] = -val;
            }
          }
        }
      }
    }
  }

  void zero_() {
    for (size_t i = 0; i < PieceMapType::PieceMapTypeCount; ++i) {
      for (size_t j = 0; j < 12 * 64; ++j) {
        pieceMaps[j][i] = 0;
      }
    }
  }

 private:
  int32_t pieceMaps[kSizeOfPieceMap][PieceMapType::PieceMapTypeCount];
};

const PieceMaps kZeroPieceMap = PieceMaps();

}  // namespace ChessEngine

#endif  // PIECE_MAPS_h
