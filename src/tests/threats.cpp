#include "gtest/gtest.h"

#include <cassert>
#include <cstdint>
#include <ctime>
#include <cstdlib>

#include <vector>

#include "../game/Position.h"
#include "../game/geometry.h"
#include "../game/Threats.h"

using namespace ChessEngine;

namespace {

TEST(Position, InitialPosition) {
  Position position = Position::init();
  Threats<Color::WHITE> threats(position);
  Bitboard gt = (kRanks[0] | kRanks[1] | kRanks[2]) & ~(bb(Square::A8) | bb(Square::H8));
  ASSERT_EQ(threats.badForOur[Piece::KING  ], gt);
  ASSERT_EQ(threats.badForOur[Piece::QUEEN], gt & ~bb(Square::D8));
  ASSERT_EQ(threats.badForOur[Piece::ROOK  ], gt & ~(bb(Square::E8) | bb(Square::D8) | bb(Square::A8) | bb(Square::H8)));
  ASSERT_EQ(threats.badForOur[Piece::BISHOP], gt & ~kRanks[0]);
  ASSERT_EQ(threats.badForOur[Piece::KNIGHT], gt & ~kRanks[0]);
  ASSERT_EQ(threats.badForOur[Piece::PAWN  ], gt& ~(kRanks[0] | kRanks[1]));
}

TEST(Position, Knight) {
  Position position = Position();
  position.place_piece_(ColoredPiece::BLACK_KNIGHT, Square::F6);
  position.place_piece_(ColoredPiece::BLACK_KING, Square::A1);
  position.place_piece_(ColoredPiece::WHITE_KING, Square::H8);

  const Bitboard attackedByBlackKnight = bb(Square::G8) | bb(Square::E8) | bb(Square::H7) | bb(Square::D7) | bb(Square::D5) | bb(Square::H5) | bb(Square::G4) | bb(Square::E4);
  const Bitboard attackedByBlackKing = bb(Square::A2) | bb(Square::B1) | bb(Square::B2);
  const Bitboard defendedByWhiteKing = bb(Square::G8) | bb(Square::H7) | bb(Square::G7);

  Threats<Color::WHITE> threats(position);

  ASSERT_EQ(threats.badForOur[Piece::PAWN], (attackedByBlackKnight | attackedByBlackKing) & ~defendedByWhiteKing);
  ASSERT_EQ(threats.badForOur[Piece::KNIGHT], (attackedByBlackKnight | attackedByBlackKing) & ~defendedByWhiteKing);
  ASSERT_EQ(threats.badForOur[Piece::BISHOP], (attackedByBlackKnight | attackedByBlackKing) & ~defendedByWhiteKing);
  ASSERT_EQ(threats.badForOur[Piece::ROOK], attackedByBlackKnight | attackedByBlackKing);
  ASSERT_EQ(threats.badForOur[Piece::QUEEN], attackedByBlackKnight | attackedByBlackKing);
}

}  // namespace

int main() {
  initialize_geometry();
  initialize_zorbrist();
  testing::InitGoogleTest();
  return RUN_ALL_TESTS();
}
