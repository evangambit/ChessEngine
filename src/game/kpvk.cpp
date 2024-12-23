#include "kpvk.h"

#include "geometry.h"

namespace ChessEngine {

// Assumes white has the pawn.
// Returns 2 if white wins
// Returns 0 if white draws
// Returns 1 if unknown
int known_kpvk_result(SafeSquare yourKing, SafeSquare theirKing, SafeSquare yourPawn, bool yourMove) {
  if (is_kpvk_win(yourKing, theirKing, yourPawn, yourMove)) {
    return 2;
  } else if (is_kpvk_draw(yourKing, theirKing, yourPawn, yourMove)) {
    return 0;
  } else {
    return 1;
  }
}

bool is_kpvk_win(SafeSquare yourKing, SafeSquare theirKing, SafeSquare yourPawn, bool yourMove) {
  const int wx = yourKing % 8;
  const int wy = yourKing / 8;
  const int bx = theirKing % 8;
  const int by = theirKing / 8;
  const int px = yourPawn % 8;
  const int py = yourPawn / 8;

  const int wdist = std::max(std::abs(wx - px), std::abs(wy - py));
  const int bdist = std::max(std::abs(bx - px), std::abs(by - py));
  const bool notRookPawn = (px != 0 && px != 7);

  bool isWinning = false;
  {  // Square rule
    const int theirDistFromPromoSquare = std::max(std::abs(bx - px), by) - !yourMove;
    isWinning |= theirDistFromPromoSquare > py;
  }

  // Key squares
  isWinning |= (wy == py - 2 && std::abs(wx - px) <= 1 && bdist + yourMove > 1) && notRookPawn;
  isWinning |= (wy == py - 1 && std::abs(wx - px) <= 1 && py <= 3 && bdist + yourMove > 1) && notRookPawn;

  // Horizontally symmetric is a win for white.
  isWinning |= (std::abs(wx - px) <= std::abs(px - bx) && wy == by) && notRookPawn;

  return isWinning;
}

bool is_kpvk_draw(SafeSquare yourKing, SafeSquare theirKing, SafeSquare yourPawn, bool yourMove) {
  const int wx = yourKing % 8;
  const int wy = yourKing / 8;
  const int bx = theirKing % 8;
  const int by = theirKing / 8;
  const int px = yourPawn % 8;
  const int py = yourPawn / 8;

  const int wdist = std::max(std::abs(wx - px), std::abs(wy - py));
  const int bdist = std::max(std::abs(bx - px), std::abs(by - py));

  // if (wx == bx && wy >= py - 1 && by == wy - 2 && by != 0 && yourMove) {
  //   return 0;
  // }

  bool isDrawn = false;

  // Black king in front of pawn.
  isDrawn |= (bx == px && by == py - 1);

  // Black king two in front of pawn and not on back rank.
  isDrawn |= (by == py - 2 && bx == px && by != 0);

  // Distance Rule:
  //   1) Compute the distance between your king and your pawn
  //   2) Compute the distance between the enemy king and your pawn
  //   3) Subtract 1 from your distance if it's your turn
  //   4) Add 1 to your enemy's distance if they're in front of your pawn and on a diagonal with it.
  // If your distance is greater than your opponent's, then it's a draw.
  isDrawn |= (wdist - yourMove > bdist + ((bx + by == wx + wy) || (bx - by == wx - wy)));

  // No-zones when you're behind your pawn.
  // TODO: this is unreliable in positions like 4k3/8/8/8/8/8/4P3/4K3 w - - 0 1
  isDrawn |= (wy > py && py > by) && (std::abs(px - bx) - !yourMove <= wy - py);

  return isDrawn;
}

}  // namespace ChessEngine
