#ifndef KPVK_H
#define KPVK_H

#include "geometry.h"

namespace ChessEngine {

// Assumes white has the pawn.
// Returns 2 if white wins
// Returns 0 if white draws
// Returns 1 if unknown
int known_kpvk_result(SafeSquare yourKing, SafeSquare theirKing, SafeSquare yourPawn, bool yourMove);

bool is_kpvk_win(SafeSquare yourKing, SafeSquare theirKing, SafeSquare yourPawn, bool yourMove);

bool is_kpvk_draw(SafeSquare yourKing, SafeSquare theirKing, SafeSquare yourPawn, bool yourMove);

}  // namespace ChessEngine

#endif  // KPVK_H