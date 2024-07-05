#ifndef MOVEGEN_ROOKS_H
#define MOVEGEN_ROOKS_H

#include "../Position.h"
#include "../utils.h"
#include "sliding.h"

namespace ChessEngine {

// Rotates east-most file to south-most rank.
constexpr Bitboard kRookMagic = bb(49) | bb(42) | bb(35) | bb(28) | bb(21) | bb(14) | bb(7) | bb(0);

Bitboard compute_rooklike_targets(Bitboard rookLikePieces, const Bitboard occupied) {
  Bitboard r = kEmptyBitboard;

  while (rookLikePieces) {
    const Square from = pop_lsb(rookLikePieces);
    const Location fromLoc = square2location(from);
    const unsigned y = from / 8;
    const unsigned x = from % 8;

    {  // Compute east/west moves.
      const unsigned rankShift = y * 8;
      uint8_t fromByte = fromLoc >> rankShift;
      uint8_t enemiesByte = occupied >> rankShift;
      r |= Bitboard(sliding_moves(fromByte, enemiesByte)) << rankShift;
    }

    {  // Compute north/south moves.
      const unsigned columnShift = 7 - x;
      uint8_t fromByte = (((fromLoc << columnShift) & kFiles[7]) * kRookMagic) >> 56;
      uint8_t enemiesByte = (((occupied << columnShift) & kFiles[7]) * kRookMagic) >> 56;
      uint8_t toByte = sliding_moves(fromByte, enemiesByte);
      r |= (((Bitboard(toByte & 254) * kRookMagic) & kFiles[0]) | (toByte & 1)) << x;
    }

  }

  return r;
}

template<Color US>
Bitboard compute_rooklike_targets(const Position& pos, Bitboard rookLikePieces) {
  const Bitboard occupied = (pos.colorBitboards_[US] | pos.colorBitboards_[opposite_color<US>()]) & ~rookLikePieces;
  return compute_rooklike_targets(rookLikePieces, occupied);
}

Bitboard compute_rook_check_mask(const Location king, const Bitboard everyone) {
  Bitboard checkMask = kEmptyBitboard;
  const Square kingSq = lsb(king);
  {  // East/west.
    unsigned y = kingSq / 8;
    const unsigned rankShift = y * 8;
    uint8_t fromByte = king >> rankShift;
    uint8_t occupied = (everyone & ~king) >> rankShift;
    checkMask |= Bitboard(sliding_moves(fromByte, occupied)) << rankShift;
  }
  {  // North/south
    const unsigned x = kingSq % 8;
    const unsigned columnShift = 7 - x;
    uint8_t fromByte = (((king << columnShift) & kFiles[7]) * kRookMagic) >> 56;
    uint8_t occupied = ((((everyone & ~king) << columnShift) & kFiles[7]) * kRookMagic) >> 56;
    uint8_t toByte = sliding_moves(fromByte, occupied);
    checkMask |= (((Bitboard(toByte & 254) * kRookMagic) & kFiles[0]) | (toByte & 1)) << x;
  }
  return checkMask;
}

// Computes moves for rook and rook-like moves for queen.
template<Color US, MoveGenType MGT>
ExtMove *compute_rook_like_moves(const Position& pos, ExtMove *moves, Bitboard target, const PinMasks& pm, Bitboard rookCheckMask, Bitboard bishopCheckMask) {
  constexpr ColoredPiece myRookPiece = (US == Color::WHITE ? ColoredPiece::WHITE_ROOK : ColoredPiece::BLACK_ROOK);
  constexpr ColoredPiece myQueenPiece = (US == Color::WHITE ? ColoredPiece::WHITE_QUEEN : ColoredPiece::BLACK_QUEEN);
  const Bitboard friends = pos.colorBitboards_[US];
  const Bitboard enemies = pos.colorBitboards_[opposite_color<US>()];
  const Bitboard myQueens = pos.pieceBitboards_[myQueenPiece];
  Bitboard rookLikePieces = pos.pieceBitboards_[myRookPiece] | pos.pieceBitboards_[myQueenPiece];

  // TODO: horizontal/vertical pins
  rookLikePieces &= ~(pm.northeast | pm.northwest);

  while (rookLikePieces) {
    const Square from = pop_lsb(rookLikePieces);
    const Piece piece = cp2p(pos.tiles_[from]);
    Location fromLoc = square2location(from);
    const unsigned y = from / 8;
    const unsigned x = from % 8;

    Bitboard tos = kEmptyBitboard;

    const Bitboard required = target
      // Target (above) handles checks. The lines below handle pins.
      & select((pm.horizontal & fromLoc) > 0, pm.horizontal, kUniverse)
      & select((pm.vertical & fromLoc) > 0, pm.vertical, kUniverse);

    {  // Compute east/west moves.
      const unsigned rankShift = y * 8;
      uint8_t fromByte = fromLoc >> rankShift;
      uint8_t occByte = (enemies | (friends & ~fromLoc)) >> rankShift;
      tos |= (Bitboard(sliding_moves(fromByte, occByte)) << rankShift) & ~friends;
    }

    {  // Compute north/south moves.
      const unsigned columnShift = 7 - x;
      uint8_t fromByte = (((fromLoc << columnShift) & kFiles[7]) * kRookMagic) >> 56;
      uint8_t occByte = ((((enemies | (friends & ~fromLoc)) << columnShift) & kFiles[7]) * kRookMagic) >> 56;
      uint8_t toByte = sliding_moves(fromByte, occByte);
      tos |= ((((Bitboard(toByte & 254) * kRookMagic) & kFiles[0]) | (toByte & 1)) << x) & ~friends;
    }

    if (MGT == MoveGenType::CAPTURES) {
      tos &= enemies;
    } else if (MGT == MoveGenType::CHECKS_AND_CAPTURES) {
      tos &= enemies | rookCheckMask | value_or_zero((fromLoc & myQueens) > 0, bishopCheckMask);
    }

    tos &= required;

    while (tos) {
      Square to = pop_lsb(tos);
      *moves++ = ExtMove(piece, pos.tiles_[to], Move{from, to, 0, MoveType::NORMAL});
    }

  }
  return moves;
}

}  // namespace ChessEngine

#endif  // MOVEGEN_ROOKS_H