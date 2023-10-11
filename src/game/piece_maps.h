#ifndef PIECE_MAPS_h
#define PIECE_MAPS_h

#include "geometry.h"
#include "utils.h"

#import "protos/weights.pb.h"

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

  int32_t const *weights(ColoredPiece cp, Square sq) const;

  void load_weights(const Weights& W) {
    for (size_t k = 0; k < PieceMapType::PieceMapTypeCount; ++k) {
      for (size_t i = NO_COLORED_PIECE; i <= WHITE_KING; ++i) {
        for (size_t y = 0; y < 8; ++y) {
          for (size_t x = 0; x < 8; ++x) {
            size_t idx1 = i * 64 + y * 8 + x;
            size_t idx2 = (i + ColoredPiece::BLACK_PAWN - ColoredPiece::WHITE_PAWN) * 64 + (7 - y) * 8 + x;
            if (k == PieceMapType::PieceMapTypeEarly) {
              pieceMaps[k][idx1] = W.earlypiecesquares(idx1);
              pieceMaps[k][idx2] = W.earlypiecesquares(idx1);
            } else {
              pieceMaps[k][idx1] = W.latepiecesquares(idx1);
              pieceMaps[k][idx2] = W.latepiecesquares(idx1);
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
