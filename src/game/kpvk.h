#ifndef KPVK_H
#define KPVK_H

#include "geometry.h"

namespace ChessEngine {

// Assumes white has the pawn.
// Returns 2 if white wins
// Returns 0 if white draws
// Returns 1 if unknown
int known_kpvk_result(Square yourKing, Square theirKing, Square yourPawn, bool yourMove);

}  // namespace ChessEngine

#endif  // KPVK_H