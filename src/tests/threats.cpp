// clang++ src/tests/position.cpp src/game/Position.cpp src/game/piece_maps.cpp src/game/utils.cpp src/game/geometry.cpp -I/opt/homebrew/Cellar/googletest/1.14.0/include -std=c++20 -L/opt/homebrew/Cellar/googletest/1.14.0/lib -lgtest && ./a.out

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
  const std::string gt = ".xxxxxx.\n"
                         "xxxxxxxx\n"
                         "xxxxxxxx\n"
                         "........\n"
                         "........\n"
                         "........\n"
                         "........\n"
                         "........\n";
  ASSERT_EQ(bstr(threats.badForOur[Piece::KING  ]), gt);
  ASSERT_EQ(bstr(threats.badForOur[Piece::QUEEN ]), gt);
  ASSERT_EQ(bstr(threats.badForOur[Piece::ROOK  ]), gt);
  ASSERT_EQ(bstr(threats.badForOur[Piece::BISHOP]), gt);
  ASSERT_EQ(bstr(threats.badForOur[Piece::KNIGHT]), gt);
  ASSERT_EQ(bstr(threats.badForOur[Piece::PAWN  ]), gt);
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
