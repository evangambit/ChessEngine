#ifndef MOVEGEN_PAWNS_H
#define MOVEGEN_PAWNS_H

#include "../Position.h"
#include "../utils.h"

namespace ChessEngine {

// Returns which pawns can attack the target.
template<Color US>
Bitboard compute_pawn_attackers(const Position& pos, const Bitboard target) {
  constexpr Direction backward = (US == Color::WHITE ? Direction::SOUTH : Direction::NORTH);
  constexpr Direction pawnCaptureDir1 = Direction(backward - 1);
  constexpr Direction pawnCaptureDir2 = Direction(backward + 1);
  const Bitboard ourPawns = pos.pieceBitboards_[coloredPiece<US, Piece::PAWN>()];
  return (shift<pawnCaptureDir1>(target) & ourPawns) | (shift<pawnCaptureDir2>(target) & ourPawns);
}

template<Color US>
Bitboard compute_pawn_targets(const Position& pos) {
  constexpr ColoredPiece cp = coloredPiece<US, Piece::PAWN>();
  constexpr Direction FORWARD = (US == Color::WHITE ? Direction::NORTH : Direction::SOUTH);
  constexpr Direction CAPTURE_NE = (FORWARD == Direction::NORTH ? Direction::NORTH_EAST : Direction::SOUTH_WEST);
  constexpr Direction CAPTURE_NW = (FORWARD == Direction::NORTH ? Direction::NORTH_WEST : Direction::SOUTH_EAST);
  const Bitboard pawns = pos.pieceBitboards_[cp];
  return shift<CAPTURE_NE>(pawns) | shift<CAPTURE_NW>(pawns);
}

// NOTE: We don't include any promotions when MGT = CHECKS_AND_CAPTURES.  Instead we only
// include queen promotions, since if you're interested in checks and captures, you're
// probably interested in promoting to queen too.
template<Color US, MoveGenType MGT>
ExtMove *compute_pawn_moves(const Position& pos, ExtMove *moves, Bitboard target, const PinMasks& pm) {
  constexpr Direction FORWARD = (US == Color::WHITE ? Direction::NORTH : Direction::SOUTH);
  constexpr Direction BACKWARD = (US == Color::WHITE ? Direction::SOUTH : Direction::NORTH);
  constexpr Direction CAPTURE_NE = FORWARD == Direction::NORTH ? Direction::NORTH_EAST : Direction::SOUTH_WEST;
  constexpr Direction CAPTURE_NW = FORWARD == Direction::NORTH ? Direction::NORTH_WEST : Direction::SOUTH_EAST;

  constexpr ColoredPiece cp = coloredPiece<US, Piece::PAWN>();
  constexpr Bitboard rowInFrontOfHome = (US == Color::WHITE ? kRanks[5] : kRanks[2]);
  constexpr Bitboard promotionRow = (US == Color::WHITE ? kRanks[0] : kRanks[7]);

  Bitboard enemies = pos.colorBitboards_[opposite_color<US>()];
  const Bitboard emptySquares = ~(pos.colorBitboards_[Color::BLACK] | pos.colorBitboards_[Color::WHITE]);
  const Location epLoc = square2location(pos.currentState_.epSquare);

  const Bitboard pawns = pos.pieceBitboards_[cp] & ~pm.horizontal;

  Bitboard checkMask;
  if (MGT == MoveGenType::CHECKS_AND_CAPTURES || MGT == MoveGenType::CAPTURES) {
    const Bitboard enemyKing = pos.pieceBitboards_[coloredPiece<opposite_color<US>(), Piece::KING>()];
    // Include pushing pawns close to promotion. Note: pawns moving to the last two ranks automatically
    // implies that they are passed.
    checkMask = (US == Color::WHITE ? (kRanks[0] | kRanks[1]) : (kRanks[7] | kRanks[6]));
    checkMask |= shift<opposite_dir(CAPTURE_NE)>(enemyKing) | shift<opposite_dir(CAPTURE_NW)>(enemyKing);
  } else {
    checkMask = kUniverse;
  }

  Bitboard b1, b2, promoting;

  if (MGT == MoveGenType::ALL_MOVES || MGT == MoveGenType::CHECKS_AND_CAPTURES) {
    b1 = shift<FORWARD>(pawns & ~(pm.northeast | pm.northwest)) & emptySquares;
    b2 = shift<FORWARD>(b1 & rowInFrontOfHome) & emptySquares;

    b1 &= target & checkMask;
    b2 &= target & checkMask;

    promoting = b1 & promotionRow;
    b1 &= ~promoting;
    while (promoting) {
      SafeSquare to = (SafeSquare)pop_lsb(promoting);
      if (MGT != MoveGenType::CHECKS_AND_CAPTURES) {
        *moves++ = ExtMove(Piece::PAWN, Move{to - FORWARD, to, 0, MoveType::PROMOTION});
        *moves++ = ExtMove(Piece::PAWN, Move{to - FORWARD, to, 1, MoveType::PROMOTION});
        *moves++ = ExtMove(Piece::PAWN, Move{to - FORWARD, to, 2, MoveType::PROMOTION});
      }
      *moves++ = ExtMove(Piece::PAWN, Move{to - FORWARD, to, 3, MoveType::PROMOTION});
    }

    while (b1) {
      SafeSquare to = (SafeSquare)pop_lsb(b1);
      *moves++ = ExtMove(Piece::PAWN, Move{to - FORWARD, to, 0, MoveType::NORMAL});
    }
    while (b2) {
      SafeSquare to = (SafeSquare)pop_lsb(b2);
      *moves++ = ExtMove(Piece::PAWN, Move{to - FORWARD - FORWARD, to, 0, MoveType::NORMAL});
    }
  }

  // If a pawn checks the king, en passant may be a legal move, despite not belonging to the target.
  // For example "8/p3Q3/1q6/1k2p3/2Pp4/3K4/5P2/8 b - c3 0 50" or "8/8/8/5Pp1/7K/1k4BP/4p1q1/8 w - g6 0 5"
  // To fix this we modify the target to include the en passant square if the corresponding pawn is part
  // of the target.
  target |= shift<FORWARD>(shift<BACKWARD>(epLoc) & target);

  if (MGT == MoveGenType::CAPTURES || MGT == MoveGenType::ALL_MOVES || MGT == MoveGenType::CHECKS_AND_CAPTURES) {
    b1 = shift<CAPTURE_NE>(pawns & ~(pm.vertical | pm.northwest)) & (enemies | epLoc);
    b1 &= target;
    promoting = b1 & promotionRow;
    b1 &= ~promoting;
    while (promoting) {
      SafeSquare to = (SafeSquare)pop_lsb(promoting);
      ColoredPiece capture = pos.tiles_[to];
      if (MGT != MoveGenType::CHECKS_AND_CAPTURES) {
        *moves++ = ExtMove(Piece::PAWN, capture, Move{to - CAPTURE_NE, to, 0, MoveType::PROMOTION});
        *moves++ = ExtMove(Piece::PAWN, capture, Move{to - CAPTURE_NE, to, 1, MoveType::PROMOTION});
        *moves++ = ExtMove(Piece::PAWN, capture, Move{to - CAPTURE_NE, to, 2, MoveType::PROMOTION});
      }
      *moves++ = ExtMove(Piece::PAWN, capture, Move{to - CAPTURE_NE, to, 3, MoveType::PROMOTION});
    }
    while (b1) {
      SafeSquare to = (SafeSquare)pop_lsb(b1);
      *moves++ = ExtMove(Piece::PAWN, pos.tiles_[to], Move{to - CAPTURE_NE, to, 0, MoveType::NORMAL});
    }

    b1 = shift<CAPTURE_NW>(pawns & ~(pm.vertical | pm.northeast)) & (enemies | epLoc);
    b1 &= target;
    promoting = b1 & promotionRow;
    b1 &= ~promoting;
    while (promoting) {
      SafeSquare to = (SafeSquare)pop_lsb(promoting);
      ColoredPiece capture = pos.tiles_[to];
      *moves++ = ExtMove(Piece::PAWN, capture, Move{to - CAPTURE_NW, to, 0, MoveType::PROMOTION});
      *moves++ = ExtMove(Piece::PAWN, capture, Move{to - CAPTURE_NW, to, 1, MoveType::PROMOTION});
      *moves++ = ExtMove(Piece::PAWN, capture, Move{to - CAPTURE_NW, to, 2, MoveType::PROMOTION});
      *moves++ = ExtMove(Piece::PAWN, capture, Move{to - CAPTURE_NW, to, 3, MoveType::PROMOTION});
    }
    while (b1) {
      SafeSquare to = (SafeSquare)pop_lsb(b1);
      ColoredPiece capture = pos.tiles_[to];
      *moves++ = ExtMove(Piece::PAWN, capture, Move{to - CAPTURE_NW, to, 0, MoveType::NORMAL});
    }
  }

  return moves;
}

}  // namespace ChessEngine

#endif  // MOVEGEN_PAWNS_H