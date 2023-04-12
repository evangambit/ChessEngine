#include "piece_maps.h"

#include <cassert>

namespace ChessEngine {

namespace {
const int32_t kEarlyPieceMap[13*64] = {
  // NO_COLORED_PIECE
   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,
// Early White P
   0,   0,   0,   0,   0,   0,   0,   0,
   3,   1,   3,   0,  -1,  -1,  -1,  -2,
  -5,  -7,   5,  -2,   7,  12,   5,  -9,
   4,   8,   3,   1,   9,  31,  25,  18,
  -3, -11,   0,  -7,  -6,   7,  -3,   3,
 -10,  -9,  -7, -10,  -2,  -2,  15,   5,
  -6,  -6, -11, -15,  -9,  18,  13,  11,
   0,   0,   0,   0,   0,   0,   0,   0,
// Early White N
 -13,  -8,  -4,   1,   1,  -4,  -8, -13,
  -8,  -4,   1,   6,   6,   1,  -4,  -8,
  -4,   1,   6,  10,  10,   6,   1,  -4,
   1,   6,  10,  15,  15,  10,   6,   1,
   1,   6,  10,  15,  15,  10,   6,   1,
  -4,   1,   6,  10,  10,   6,   1,  -4,
  -8,  -4,   1,   6,   6,   1,  -4,  -8,
 -13,  -8,  -4,   1,   1,  -4,  -8, -13,
// Early White B
   2,  -2,  -1,   0,  -1,  -1,  -1,   0,
  -5,   1,   2,  -3,   0,   1,  -4,  -9,
   1,   1,  -4,   5,   9,   7,   8,  13,
   0,   3,   5,   4,  12,   6,   7,   0,
   3,   2,   1,  10,   7,   3,  -4,   0,
  -5,  10,   6,  10,  15,  -4,   3,   2,
   0,  15,  16,   4,   4,   8,   9,   2,
   0,  16,  15,  -8,  -3,   1,   0,  -6,
// Early White R
   2,  -1,   1,   2,   0,   0,   1,   1,
   0,   4,   6,   4,   4,   3,   3,   2,
   1,   2,   2,   2,   6,   6,   4,   2,
  -4,  -3,   1,  -2,  -1,   3,   3,   1,
  -3,   0,   0,   0,   1,   0,   4,   4,
  -9,  -1,  -3,  -3,   4,   4,  14,   5,
 -14,  -1,  -2,  -4,  -4,   4,   1,  -9,
   3,   6,   5,  16,  21,  15,  -2,  13,
// Early White Q
  -1,  -1,   0,  -1,   1,   1,   0,   1,
  -2,  -1,   2,   1,   2,   8,   2,   8,
  -3,   1,   0,   5,   9,  13,  12,  23,
  -3,  -5,   1,   3,   8,   9,   8,  11,
   0,  -2,  -3,  -2,   6,   2,   5,  16,
  -3,   0,   3,  -3,   1,   7,  17,  10,
  -1,   3,   9,   5,  11,  12,   2,   2,
  -6,  -5,   1,   8,   4,  -3,  -1,  -2,
// Early White K
   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   1,   1,   0,
   0,   0,   0,   0,   0,   0,   0,  -1,
  -1,   0,   0,   0,  -1,   1,  -3,  -2,
   0,   1,  -1,  -2,  -3,  -6,   1,  -8,
   5,   5,   2,  -8,   2,   2,  37,  38,
   1,  18,  -4, -15,  16,  -9,  10,  19,
// Early BlackP
   0,   0,   0,   0,   0,   0,   0,   0,
  10,  16,  12,  12,   5, -25, -22, -18,
  13,  15,   7,   5,  -2,   8, -22, -11,
  11,  14,   1,   9,   7,  -6,  10,  -3,
  -7,   1,   2,  -2, -14, -31, -17, -15,
  12,   1,  -2,   3,  -2, -11,  -7,   3,
  -3,  -1,  -1,   1,   0,   0,   2,   1,
   0,   0,   0,   0,   0,   0,   0,   0,
// Early BlackN
  13,   8,   4,  -1,  -1,   4,   8,  13,
   8,   4,  -1,  -6,  -6,  -1,   4,   8,
   4,  -1,  -6, -10, -10,  -6,  -1,   4,
  -1,  -6, -10, -15, -15, -10,  -6,  -1,
  -1,  -6, -10, -15, -15, -10,  -6,  -1,
   4,  -1,  -6, -10, -10,  -6,  -1,   4,
   8,   4,  -1,  -6,  -6,  -1,   4,   8,
  13,   8,   4,  -1,  -1,   4,   8,  13,
// Early BlackB
   6,  -5,  -7,  -1,   5,  -6,   0,  -2,
  -1,  -9, -10,   4,  -8,  -9, -18,  -2,
   9,  -6,   1,  -7,   0,  -1,  -8,  -5,
   2,   3,  -2,  -6, -13,  -2,   1,   2,
   5,  -8,  -2,  -8,  -4,  -6,  -4,   5,
  -5,   1,   4,  -1,  -4,  -5,  -5, -10,
   4,   1,   3,   1,   0,   2,   7,   3,
  -3,   2,   2,   1,   1,   0,   0,   0,
// Early BlackR
   1,  -4,   0, -12, -15, -14,   6, -11,
  18,   3,   4,   2,   3, -10,  -6,   8,
   9,   2,   7,   3,   3,   1, -11,  -2,
   5,   2,   2,  -2,  -2,   1,  -2,   1,
   2,   0,   1,  -3,  -3,  -5,  -2,  -1,
   0,  -3,  -4,  -2,  -3,  -3,  -2,  -2,
   0,  -3,  -4,  -4,  -2,  -3,  -1,  -2,
   0,   1,   0,   2,   0,   0,   0,  -1,
// Early BlackQ
   8,   5,  -2,  -8,  -5,   8,   0,   1,
   2,  -3, -12,   2,  -9,  -8,  -2,  -1,
   3,  -2,  -2,   8,   0,  -1, -12,  -5,
  -4,  -3,   4,   6,  -6,  -3,  -4,  -8,
   1,  -6,  -2,  -4,  -3, -11,  -3, -20,
   1,   1,  -2,  -4,  -8,  -7,  -5, -14,
   4,   6,   0,  -1,   0,  -6,   1,  -2,
   3,   2,   0,   1,   2,  -1,   0,  -2,
// Early BlackK
   0,  -6,   9,  27, -10,  17,  -6, -12,
  -1,  -5,   2,  11,  -1,   1, -33, -31,
   1,   0,   2,   0,   2,   6,   5,   5,
   1,   0,   0,   0,   0,   0,   2,   3,
   1,   0,   0,  -1,  -1,  -1,   0,   1,
   0,   0,   0,   0,   0,   0,  -1,   0,
   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,
};

const int32_t kLatePieceMap[13*64] = {
  // NO_COLORED_PIECE
   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,
// Late White P
   0,   0,   0,   0,   0,   0,   0,   0,
   5,   3,   3,  -3,  -5,  -3,  -1,   2,
  14,   6,   2, -12, -16,   1,  -4,   3,
  15,   5,   7,  -2,   2,  -2,  -6,   1,
   2,   1,  -5,   4,   7,   1,  -4,  -4,
  -2,  -3,   2,   3,   9,   8,  -7, -10,
   0,  -1,   6,  -5,  13,   2,   0,  -9,
   0,   0,   0,   0,   0,   0,   0,   0,
// Late White N
  -6,  -1,  -2,  -1,   0,  -2,  -1,  -3,
  -3,   1,   1,   3,   2,   3,  -1,  -2,
  -1,   3,  -1,   0,   1,   0,   4,  -2,
  -2,   9,   2,   3,   0,   7,  19,   1,
  -1,   3,   8,   1,  -1,  -2,   5,   2,
  -4,   4, -13,   6,   8, -10,  -8,   2,
  -5,   0,   1,   7,   1,   4,   0,  -6,
  -2,  -4,  -2,  -3,  -3,  -3,  -4,  -3,
// Late White B
   0,  -3,   0,   2,  -1,   0,  -1,   0,
   1,  -2,  -1,  -1,   0,   1,  -1,  -3,
   4,   1,  -1,  -3,   0,   3,   0,   4,
   3,  11,   1,   2,   0,   2,   9,  -1,
   2,   0,  -1,   0,  -2,  -3,   0,  -6,
  -2,  -3,   2,   5,   2,   2,  -6,  -1,
  -3, -14,   0,   1,   3,   2,  -3,  -4,
  -3,   1, -11,   2,   2,   6,  -1,  -3,
// Late White R
   2,   3,   1,   1,  -1,   0,   1,   0,
   3,   6,   5,   3,   1,   2,   2,   1,
   3,   3,   4,  -1,   0,   5,   3,   0,
   2,   1,   3,  -1,   0,   1,   1,  -1,
   2,   3,   5,   3,   2,   2,   3,   2,
   3,   4,   1,   1,  -3,   1,   4,  -2,
   2,   2,   3,  -1,  -2,  -3,  -1,  -1,
   0,   2,   8,  -3,  -6,   3,   7, -19,
// Late White Q
  -2,  -1,   1,  -2,   2,   1,   0,  -1,
   0,  -2,   1,   1,   3,   7,   1,   3,
  -2,   2,   1,   2,   6,   8,   5,   9,
   0,   3,   2,   7,   8,   8,   7,  12,
  -1,   0,   7,  17,  10,   9,   9,   5,
  -2,   4,   8,  14,   8,  14,   5,   1,
  -1,   3,  -8,   4,   0,   2,   0,   1,
   0,   0,   3,  -8,   4,   0,   0,  -1,
// Late White K
   0,  -1,   0,   0,   0,  -1,   0,   0,
  -1,   0,   1,   0,   0,   1,   1,   0,
   0,   1,   1,   1,   1,   4,   2,  -2,
  -2,   1,   4,   0,   2,   5,   2,  -4,
  -2,  -2,   4,   1,  -3,   4,  -6, -11,
  -3,   0,   4,   5,   5,  13,  -1, -10,
  -3,  -3,   4,  -4,  -6,   3,  -8, -21,
  -1,   0,   5,   2,  -2,  21,  -3, -16,
// Late BlackP
   0,   0,   0,   0,   0,   0,   0,   0,
  -3,  -3,  -5,   6,  -3,   0,   1,  13,
   2,   1,   0,  -1,  -4, -10,   7,  11,
  -6,   3,   9,  -6,  -8,  -1,   0,   4,
 -11,   1,  -4,   3,   1,   5,   2,  -1,
 -10,  -4,   1,  10,   6,  -2,  -1,  -3,
  -3,  -1,  -1,   2,   2,   0,   1,   0,
   0,   0,   0,   0,   0,   0,   0,   0,
// Late BlackN
   3,   9,   3,   0,   4,   2,  -6,   2,
   3,   2,  -2,  -3,  -5,  -4,   2,   3,
   2,  -4,   6,  -4,  -4,   4,  -3,  -1,
   4,  -4,  -9,   2,  -2,  -6,  -5,   3,
   4, -13,  -3,  -3,   1,  -4, -15,  -4,
   0,  -2,   3,   0,   0,  -1,  -3,   3,
   3,   3,   0,  -1,  -1,   0,   1,   3,
   5,   1,   1,   0,   1,   0,   1,   3,
// Late BlackB
   5,   1,  11,  -2,   0,  -5,   0,   2,
   3,  17,  -3,  -2,  -6,  -1,   4,   3,
   3,  -4,  -3,  -8, -12,   0,   4,  -3,
  -1,  -1,  -5,  -2,  -6,   1,   0,   2,
  -1, -12,  -3,  -3,  -2,  -3,  -1,   0,
  -1,   0,   1,   1,   0,  -3,  -2,  -2,
   0,  -3,  -1,   1,   1,   0,   4,   0,
   0,   1,   1,   2,   0,   0,  -1,   0,
// Late BlackR
  -4,  -3,  -9,  -1,   2,  -4, -11,  18,
  -7,   0,  -1,   1,   2,   4,   0,   3,
  -2,  -3,  -2,  -3,   3,   2,  -4,   2,
  -5,  -2,  -5,  -5,  -3,  -4,  -2,   0,
  -2,  -3,  -1,  -3,   1,  -2,  -3,   0,
  -2,  -1,  -3,   0,   2,  -2,  -1,   0,
  -1,  -8,  -6,  -5,   0,  -2,  -1,   1,
   0,   2,   1,   0,   0,   0,  -2,   0,
// Late BlackQ
   2,   1,  -3,   4,  -1,   2,   0,   0,
   3,  -2,  10, -12,  -7,  -1,   0,   0,
   3,   0, -10, -16, -10, -15,  -6,  -3,
  13,  -4,  -6, -15,  -9,  -6,  -5,  -4,
   1,  -7,  -2,  -5,  -6,  -8,  -4, -10,
  -2,  -1,  -2,  -3,  -7,  -7,  -2,  -6,
   2,   3,  -2,  -3,  -1,  -5,   0,  -1,
   3,   2,   0,   0,   1,  -1,   0,  -1,
// Late BlackK
   3,   1,  -7,  -5,   5, -23,   1,  16,
   5,   4,  -4,   3,   4,  -5,   6,  18,
   4,  -1,  -7,  -5,  -8, -13,  -1,   6,
   4,   2,  -1,   0,   1,  -6,   5,   8,
   0,   0,  -3,  -3,  -3,  -5,  -1,   5,
   0,  -1,  -2,   0,  -1,  -3,  -2,   1,
   0,  -1,   0,   1,   0,  -1,  -1,   0,
   0,   1,   1,   1,   0,   0,   0,   0,
};

}  // namespace

int32_t early_piece_map(ColoredPiece cp, Square sq) {
	assert(cp >= 0);
	assert(cp < 13);
	assert(sq >= 0);
	assert(sq < 64);
    return kEarlyPieceMap[cp * 64 + sq];
}

int32_t late_piece_map(ColoredPiece cp, Square sq) {
	assert(cp >= 0);
	assert(cp < 13);
	assert(sq >= 0);
	assert(sq < 64);
    return kLatePieceMap[cp * 64 + sq];
}

}  // namespace ChessEngine