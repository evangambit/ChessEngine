#include "piece_maps.h"

#include <cassert>

namespace ChessEngine {

int32_t const *PieceMaps::weights(ColoredPiece cp, SafeSquare sq) const {
  assert(cp >= 0);
  assert(cp < 13);
  assert(sq >= 0);
  assert(sq < 64);
  return &pieceMaps[cp * 64 + sq][0];
}

}  // namespace ChessEngine