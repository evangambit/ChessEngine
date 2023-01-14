#ifndef MOVEGEN_KNIGHTS_H
#define MOVEGEN_KNIGHTS_H

#include "../Position.h"
#include "../utils.h"

namespace ChessEngine {

namespace {

constexpr Bitboard kKnightMoves[64] = {
  0x0000000000020400,
  0x0000000000050800,
  0x00000000000a1100,
  0x0000000000142200,
  0x0000000000284400,
  0x0000000000508800,
  0x0000000000a01000,
  0x0000000000402000,
  0x0000000002040004,
  0x0000000005080008,
  0x000000000a110011,
  0x0000000014220022,
  0x0000000028440044,
  0x0000000050880088,
  0x00000000a0100010,
  0x0000000040200020,
  0x0000000204000402,
  0x0000000508000805,
  0x0000000a1100110a,
  0x0000001422002214,
  0x0000002844004428,
  0x0000005088008850,
  0x000000a0100010a0,
  0x0000004020002040,
  0x0000020400040200,
  0x0000050800080500,
  0x00000a1100110a00,
  0x0000142200221400,
  0x0000284400442800,
  0x0000508800885000,
  0x0000a0100010a000,
  0x0000402000204000,
  0x0002040004020000,
  0x0005080008050000,
  0x000a1100110a0000,
  0x0014220022140000,
  0x0028440044280000,
  0x0050880088500000,
  0x00a0100010a00000,
  0x0040200020400000,
  0x0204000402000000,
  0x0508000805000000,
  0x0a1100110a000000,
  0x1422002214000000,
  0x2844004428000000,
  0x5088008850000000,
  0xa0100010a0000000,
  0x4020002040000000,
  0x0400040200000000,
  0x0800080500000000,
  0x1100110a00000000,
  0x2200221400000000,
  0x4400442800000000,
  0x8800885000000000,
  0x100010a000000000,
  0x2000204000000000,
  0x0004020000000000,
  0x0008050000000000,
  0x00110a0000000000,
  0x0022140000000000,
  0x0044280000000000,
  0x0088500000000000,
  0x0010a00000000000,
  0x0020400000000000,
};

}  // namespace

template<Color US>
Bitboard compute_knight_targets(const Position& pos) {
  constexpr ColoredPiece cp = coloredPiece<US, Piece::KNIGHT>();
  const Bitboard knights = pos.pieceBitboards_[cp];
  Bitboard r = kEmptyBitboard;
  r |= (knights & ~(kFiles[0] | kRanks[0] | kRanks[1])) >> 17;
  r |= (knights & ~(kFiles[7] | kRanks[0] | kRanks[1])) >> 15;
  r |= (knights & ~(kFiles[0] | kFiles[1] | kRanks[0])) >> 10;
  r |= (knights & ~(kFiles[6] | kFiles[7] | kRanks[0])) >> 6;
  r |= (knights & ~(kFiles[0] | kFiles[1] | kRanks[7])) << 6;
  r |= (knights & ~(kFiles[6] | kFiles[7] | kRanks[7])) << 10;
  r |= (knights & ~(kFiles[0] | kRanks[6] | kRanks[7])) << 15;
  r |= (knights & ~(kFiles[7] | kRanks[6] | kRanks[7])) << 17;
  return r;
}

template<Color US, MoveGenType MGT>
ExtMove *compute_knight_moves(const Position& pos, ExtMove *moves, Bitboard target) {
  constexpr ColoredPiece cp = (US == Color::WHITE ? ColoredPiece::WHITE_KNIGHT : ColoredPiece::BLACK_KNIGHT);
  const Bitboard enemies = pos.colorBitboards_[opposite_color<US>()];
  const Bitboard notfriends = ~pos.colorBitboards_[US];
  Bitboard knights = pos.pieceBitboards_[cp];
  while (knights) {
    const Square from = pop_lsb(knights);
    Bitboard tos;
    if (MGT == MoveGenType::ALL_MOVES) {
      tos = kKnightMoves[from] & notfriends & target;
    } else if (MGT == MoveGenType::CAPTURES) {
      tos = kKnightMoves[from] & enemies & target;
    } else if (MGT == MoveGenType::QUIET_MOVES) {
      tos = kKnightMoves[from] & notfriends & ~enemies & target;
    }
    while (tos) {
      Square to = pop_lsb(tos);
      *moves++ = ExtMove(Piece::KNIGHT, cp2p(pos.tiles_[to]), Move{from, to, 0, MoveType::NORMAL});
    }
  }
  return moves;
}

}  // namespace ChessEngine

#endif  // MOVEGEN_KNIGHTS_H
