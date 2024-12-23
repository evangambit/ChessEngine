#ifndef MOVEGEN_KINGS_H
#define MOVEGEN_KINGS_H

#include "../Position.h"
#include "../utils.h"

namespace ChessEngine {

namespace {

constexpr Bitboard kKingMoves[65] = {
  0x0000000000000302,
  0x0000000000000705,
  0x0000000000000e0a,
  0x0000000000001c14,
  0x0000000000003828,
  0x0000000000007050,
  0x000000000000e0a0,
  0x000000000000c040,
  0x0000000000030203,
  0x0000000000070507,
  0x00000000000e0a0e,
  0x00000000001c141c,
  0x0000000000382838,
  0x0000000000705070,
  0x0000000000e0a0e0,
  0x0000000000c040c0,
  0x0000000003020300,
  0x0000000007050700,
  0x000000000e0a0e00,
  0x000000001c141c00,
  0x0000000038283800,
  0x0000000070507000,
  0x00000000e0a0e000,
  0x00000000c040c000,
  0x0000000302030000,
  0x0000000705070000,
  0x0000000e0a0e0000,
  0x0000001c141c0000,
  0x0000003828380000,
  0x0000007050700000,
  0x000000e0a0e00000,
  0x000000c040c00000,
  0x0000030203000000,
  0x0000070507000000,
  0x00000e0a0e000000,
  0x00001c141c000000,
  0x0000382838000000,
  0x0000705070000000,
  0x0000e0a0e0000000,
  0x0000c040c0000000,
  0x0003020300000000,
  0x0007050700000000,
  0x000e0a0e00000000,
  0x001c141c00000000,
  0x0038283800000000,
  0x0070507000000000,
  0x00e0a0e000000000,
  0x00c040c000000000,
  0x0302030000000000,
  0x0705070000000000,
  0x0e0a0e0000000000,
  0x1c141c0000000000,
  0x3828380000000000,
  0x7050700000000000,
  0xe0a0e00000000000,
  0xc040c00000000000,
  0x0203000000000000,
  0x0507000000000000,
  0x0a0e000000000000,
  0x141c000000000000,
  0x2838000000000000,
  0x5070000000000000,
  0xa0e0000000000000,
  0x40c0000000000000,
  0x0,  // NO_SQUARE
};

}  // namespace

template<Color US>
Bitboard compute_king_targets(const Position& pos, UnsafeSquare sq) {
  return kKingMoves[sq];
}

template<Color US>
Bitboard compute_king_targets(const Position& pos, SafeSquare sq) {
  return kKingMoves[sq];
}

template<Color US, MoveGenType MGT, bool inCheck>
ExtMove *compute_king_moves(const Position& pos, ExtMove *moves, Bitboard target) {
  constexpr ColoredPiece cp = (US == Color::WHITE ? ColoredPiece::WHITE_KING : ColoredPiece::BLACK_KING);
  const Bitboard notfriends = ~pos.colorBitboards_[US];
  const Bitboard enemies = pos.colorBitboards_[opposite_color<US>()];
  const Bitboard allPieces = pos.colorBitboards_[Color::WHITE] | pos.colorBitboards_[Color::BLACK];
  Bitboard kings = pos.pieceBitboards_[cp];
  while (kings) {
    const SafeSquare from = (SafeSquare)pop_lsb(kings);
    Bitboard tos = kKingMoves[from] & target;
    if (MGT == MoveGenType::ALL_MOVES) {
      tos &= notfriends;
    } else if (MGT == MoveGenType::CAPTURES || MGT == MoveGenType::CHECKS_AND_CAPTURES) {
      tos &= enemies;
    }
    while (tos) {
      SafeSquare to = (SafeSquare)pop_lsb(tos);
      *moves++ = ExtMove(Piece::KING, pos.tiles_[to], Move{from, to, 0, MoveType::NORMAL});
    }
  }

  // TODO: castling can be allowed when we're looking for checks.
  if (MGT == MoveGenType::ALL_MOVES) {
    CastlingRights cr = pos.currentState_.castlingRights;
    if (!inCheck) {
      if (US == Color::WHITE) {
        if (!inCheck
          && ((allPieces & (bb(SafeSquare(62)) | bb(SafeSquare(61)))) == 0)
          && (cr & kCastlingRights_WhiteKing)
          && !can_enemy_attack<US>(pos, SafeSquare::SF1)
          && !can_enemy_attack<US>(pos, SafeSquare::SG1)
          && (target & bb(SafeSquare::SG1))) {
          *moves++ = ExtMove(Piece::KING, ColoredPiece::NO_COLORED_PIECE, Move{SafeSquare::SE1, SafeSquare::SG1, 0, MoveType::CASTLE});
        }
        if (!inCheck
          && ((allPieces & (bb(SafeSquare(59)) | bb(SafeSquare(58)) | bb(SafeSquare(57)))) == 0)
          && (cr & kCastlingRights_WhiteQueen)
          && !can_enemy_attack<US>(pos, SafeSquare::SD1)
          && !can_enemy_attack<US>(pos, SafeSquare::SC1)
          && (target & bb(SafeSquare::SC1))) {
          *moves++ = ExtMove(Piece::KING, ColoredPiece::NO_COLORED_PIECE, Move{SafeSquare::SE1, SafeSquare::SC1, 0, MoveType::CASTLE});
        }
      } else {
        if (((allPieces & (bb(SafeSquare(5)) | bb(SafeSquare(6)))) == 0)
          && (cr & kCastlingRights_BlackKing)
          && !can_enemy_attack<US>(pos, SafeSquare(5))
          && !can_enemy_attack<US>(pos, SafeSquare(6))
          && (target & bb(SafeSquare(6)))) {
          *moves++ = ExtMove(Piece::KING, ColoredPiece::NO_COLORED_PIECE, Move{SafeSquare::SE8, SafeSquare(6), 0, MoveType::CASTLE});
        }
        if (((allPieces & (bb(SafeSquare(1)) | bb(SafeSquare(2)) | bb(SafeSquare(3)))) == 0)
          && (cr & kCastlingRights_BlackQueen)
          && !can_enemy_attack<US>(pos, SafeSquare(2))
          && !can_enemy_attack<US>(pos, SafeSquare(3))
          && (target & bb(SafeSquare(2)))) {
          *moves++ = ExtMove(Piece::KING, ColoredPiece::NO_COLORED_PIECE, Move{SafeSquare::SE8, SafeSquare(2), 0, MoveType::CASTLE});
        }
      }
    }
  }

  return moves;
}

}  // namespace ChessEngine

#endif  // MOVEGEN_KINGS_H

