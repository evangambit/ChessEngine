#ifndef MOVEGEN_PAWNS_H
#define MOVEGEN_PAWNS_H

#include "../Position.h"
#include "../utils.h"

namespace ChessEngine {

// Returns which enemy pawns can attack the target.
template<Color US>
Bitboard compute_enemy_pawn_attackers(const Position& pos, const Bitboard target) {
  constexpr Color enemyColor = opposite_color<US>();
  constexpr Direction forward = (US == Color::WHITE ? Direction::NORTH : Direction::SOUTH);
  constexpr Direction pawnCaptureDir1 = Direction(forward - 1);
  constexpr Direction pawnCaptureDir2 = Direction(forward + 1);
  const Bitboard enemyPawns = pos.pieceBitboards_[coloredPiece<enemyColor, Piece::PAWN>()];
  return (shift<pawnCaptureDir1>(target) & enemyPawns) | (shift<pawnCaptureDir2>(target) & enemyPawns);
}

template<Color US>
Bitboard compute_pawn_targets(const Position& pos) {
  constexpr ColoredPiece cp = coloredPiece<US, Piece::PAWN>();
  constexpr Direction FORWARD = (US == Color::WHITE ? Direction::NORTH : Direction::SOUTH);
  constexpr Direction CAPTURE_DIR1 = Direction(FORWARD - 1);
  constexpr Direction CAPTURE_DIR2 = Direction(FORWARD + 1);
  const Bitboard pawns = pos.pieceBitboards_[cp];
  return shift<CAPTURE_DIR1>(pawns) | shift<CAPTURE_DIR2>(pawns);
}

template<Color US, MoveGenType MGT>
ExtMove *compute_pawn_moves(const Position& pos, ExtMove *moves, Bitboard target) {
  constexpr Direction FORWARD = (US == Color::WHITE ? Direction::NORTH : Direction::SOUTH);
  constexpr Direction CAPTURE_DIR1 = Direction(FORWARD - 1);
  constexpr Direction CAPTURE_DIR2 = Direction(FORWARD + 1);

  constexpr ColoredPiece cp = coloredPiece<US, Piece::PAWN>();
  constexpr Bitboard rowInFrontOfHome = (US == Color::WHITE ? kRanks[5] : kRanks[2]);
  constexpr Bitboard promotionRow = (US == Color::WHITE ? kRanks[0] : kRanks[7]);

  Bitboard enemies = pos.colorBitboards_[opposite_color<US>()];
  const Bitboard emptySquares = ~(pos.colorBitboards_[Color::BLACK] | pos.colorBitboards_[Color::WHITE]);
  const Location epLoc = square2location(pos.currentState_.epSquare);

  const Bitboard pawns = pos.pieceBitboards_[cp];

  Bitboard b1, b2, promoting;

  if (MGT & MoveGenType::QUIET_MOVES) {
    b1 = shift<FORWARD>(pawns) & emptySquares;
    b2 = shift<FORWARD>(b1 & rowInFrontOfHome) & emptySquares;

    b1 &= target;
    b2 &= target;

    promoting = b1 & promotionRow;
    b1 &= ~promoting;
    while (promoting) {
      Square to = pop_lsb(promoting);
      *moves++ = ExtMove(Piece::PAWN, Move{to - FORWARD, to, 0, MoveType::PROMOTION});
      *moves++ = ExtMove(Piece::PAWN, Move{to - FORWARD, to, 1, MoveType::PROMOTION});
      *moves++ = ExtMove(Piece::PAWN, Move{to - FORWARD, to, 2, MoveType::PROMOTION});
      *moves++ = ExtMove(Piece::PAWN, Move{to - FORWARD, to, 3, MoveType::PROMOTION});
    }

    while (b1) {
      Square to = pop_lsb(b1);
      *moves++ = ExtMove(Piece::PAWN, Move{to - FORWARD, to, 0, MoveType::NORMAL});
    }
    while (b2) {
      Square to = pop_lsb(b2);
      *moves++ = ExtMove(Piece::PAWN, Move{to - FORWARD - FORWARD, to, 0, MoveType::NORMAL});
    }
  }

  if (MGT & MoveGenType::CAPTURES) {
    b1 = shift<CAPTURE_DIR1>(pawns) & (enemies | epLoc);
    b1 &= target;
    promoting = b1 & promotionRow;
    b1 &= ~promoting;
    while (promoting) {
      Square to = pop_lsb(promoting);
      Piece capture = cp2p(pos.tiles_[to]);
      *moves++ = ExtMove(Piece::PAWN, capture, Move{to - CAPTURE_DIR1, to, 0, MoveType::PROMOTION});
      *moves++ = ExtMove(Piece::PAWN, capture, Move{to - CAPTURE_DIR1, to, 1, MoveType::PROMOTION});
      *moves++ = ExtMove(Piece::PAWN, capture, Move{to - CAPTURE_DIR1, to, 2, MoveType::PROMOTION});
      *moves++ = ExtMove(Piece::PAWN, capture, Move{to - CAPTURE_DIR1, to, 3, MoveType::PROMOTION});
    }
    while (b1) {
      Square to = pop_lsb(b1);
      Piece capture = cp2p(pos.tiles_[to]);
      *moves++ = ExtMove(Piece::PAWN, capture, Move{to - CAPTURE_DIR1, to, 0, MoveType::NORMAL});
    }

    b1 = shift<CAPTURE_DIR2>(pawns) & (enemies | epLoc);
    b1 &= target;
    promoting = b1 & promotionRow;
    b1 &= ~promoting;
    while (promoting) {
      Square to = pop_lsb(promoting);
      Piece capture = cp2p(pos.tiles_[to]);
      *moves++ = ExtMove(Piece::PAWN, capture, Move{to - CAPTURE_DIR2, to, 0, MoveType::PROMOTION});
      *moves++ = ExtMove(Piece::PAWN, capture, Move{to - CAPTURE_DIR2, to, 1, MoveType::PROMOTION});
      *moves++ = ExtMove(Piece::PAWN, capture, Move{to - CAPTURE_DIR2, to, 2, MoveType::PROMOTION});
      *moves++ = ExtMove(Piece::PAWN, capture, Move{to - CAPTURE_DIR2, to, 3, MoveType::PROMOTION});
    }
    while (b1) {
      Square to = pop_lsb(b1);
      Piece capture = cp2p(pos.tiles_[to]);
      *moves++ = ExtMove(Piece::PAWN, capture, Move{to - CAPTURE_DIR2, to, 0, MoveType::NORMAL});
    }
  }

  return moves;
}

}  // namespace ChessEngine

#endif  // MOVEGEN_PAWNS_H