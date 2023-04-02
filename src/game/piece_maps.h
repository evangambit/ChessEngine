#ifndef PIECE_MAPS_h
#define PIECE_MAPS_h

#include "geometry.h"
#include "utils.h"

#include <cstdint>

namespace ChessEngine {

int32_t early_piece_map(ColoredPiece cp, Square sq);

int32_t late_piece_map(ColoredPiece cp, Square sq);

}  // namespace ChessEngine

#endif  // PIECE_MAPS_h
