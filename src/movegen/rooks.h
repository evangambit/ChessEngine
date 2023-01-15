#ifndef MOVEGEN_ROOKS_H
#define MOVEGEN_ROOKS_H

#include "../Position.h"
#include "../utils.h"
#include "sliding.h"

namespace ChessEngine {

// Rotates east-most file to south-most rank.
constexpr Bitboard kRookMagic = bb(49) | bb(42) | bb(35) | bb(28) | bb(21) | bb(14) | bb(7) | bb(0);

template<Color US>
Bitboard compute_rooklike_targets(const Position& pos, Bitboard rookLikePieces, const Bitboard occupied) {
  Bitboard r = kEmptyBitboard;

  while (rookLikePieces) {
    const Square from = pop_lsb(rookLikePieces);
    const Location fromLoc = square2location(from);
    const unsigned y = from / 8;
    const unsigned x = from % 8;
    const Bitboard rank = kRanks[y];
    const Bitboard file = kFiles[x];

    Bitboard tos = kEmptyBitboard;

    {  // Compute east/west moves.
      const unsigned rankShift = y * 8;
      uint8_t fromByte = fromLoc >> rankShift;
      uint8_t enemiesByte = (occupied & rank) >> rankShift;
      r |= Bitboard(sliding_moves(fromByte, 0, enemiesByte)) << rankShift;
    }

    {  // Compute north/south moves.
      const unsigned columnShift = 7 - x;
      uint8_t fromByte = (((fromLoc << columnShift) & kFiles[7]) * kRookMagic) >> 56;
      uint8_t enemiesByte = (((occupied << columnShift) & kFiles[7]) * kRookMagic) >> 56;
      uint8_t toByte = sliding_moves(fromByte, 0, enemiesByte);
      r |= (((Bitboard(toByte & 254) * kRookMagic) & kFiles[0]) | (toByte & 1)) << x;
    }

  }

  return r;
}

template<Color US>
Bitboard compute_rooklike_targets(const Position& pos, Bitboard rookLikePieces) {
  const Bitboard occupied = (pos.colorBitboards_[US] | pos.colorBitboards_[opposite_color<US>()]) & ~rookLikePieces;
  return compute_rooklike_targets<US>(pos, rookLikePieces, occupied);
}


template<Color US>
Bitboard compute_rook_targets(const Position& pos, const Bitboard bishopLikePieces) {
  return compute_rooklike_targets<US>(pos, pos.pieceBitboards_[coloredPiece<US, Piece::ROOK>()]);
}

// Computes moves for rook and rook-like moves for queen.
template<Color US, MoveGenType MGT>
ExtMove *compute_rook_like_moves(const Position& pos, ExtMove *moves, Bitboard target) {
  constexpr ColoredPiece myRookPiece = (US == Color::WHITE ? ColoredPiece::WHITE_ROOK : ColoredPiece::BLACK_ROOK);
  constexpr ColoredPiece myQueenPiece = (US == Color::WHITE ? ColoredPiece::WHITE_QUEEN : ColoredPiece::BLACK_QUEEN);
  const Bitboard friends = pos.colorBitboards_[US];
  const Bitboard enemies = pos.colorBitboards_[opposite_color<US>()];
  Bitboard rookLikePieces = pos.pieceBitboards_[myRookPiece] | pos.pieceBitboards_[myQueenPiece];
  while (rookLikePieces) {
    const Square from = pop_lsb(rookLikePieces);
    const Piece piece = cp2p(pos.tiles_[from]);
    Location fromLoc = square2location(from);
    const unsigned y = from / 8;
    const unsigned x = from % 8;
    const Bitboard rank = kRanks[y];
    const Bitboard file = kFiles[x];

    Bitboard tos = kEmptyBitboard;

    {  // Compute east/west moves.
      const unsigned rankShift = y * 8;
      uint8_t fromByte = fromLoc >> rankShift;
      uint8_t friendsByte = (friends & ~fromLoc & rank) >> rankShift;
      uint8_t enemiesByte = (enemies & rank) >> rankShift;

      tos |= Bitboard(sliding_moves(fromByte, friendsByte, enemiesByte)) << rankShift;
    }

    {  // Compute north/south moves.
      const unsigned columnShift = 7 - x;
      uint8_t fromByte = (((fromLoc << columnShift) & kFiles[7]) * kRookMagic) >> 56;
      uint8_t friendsByte = ((((friends & ~fromLoc) << columnShift) & kFiles[7]) * kRookMagic) >> 56;
      uint8_t enemiesByte = (((enemies << columnShift) & kFiles[7]) * kRookMagic) >> 56;
      uint8_t toByte = sliding_moves(fromByte, friendsByte, enemiesByte);
      tos |= (((Bitboard(toByte & 254) * kRookMagic) & kFiles[0]) | (toByte & 1)) << x;
    }

    if (MGT == MoveGenType::CAPTURES) {
      tos &= enemies;
    }

    tos &= target;

    while (tos) {
      Square to = pop_lsb(tos);
      *moves++ = ExtMove(piece, cp2p(pos.tiles_[to]), Move{from, to, 0, MoveType::NORMAL});
    }

  }
  return moves;
}

}  // namespace ChessEngine

#endif  // MOVEGEN_ROOKS_H