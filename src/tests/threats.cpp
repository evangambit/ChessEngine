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

::testing::AssertionResult AssertBB(Bitboard actual, Bitboard expected) {
  if (actual == expected) {
      return ::testing::AssertionSuccess();
  } else {
      return ::testing::AssertionFailure() << std::endl << bstr(actual) << " not equal to " << std::endl << bstr(expected) << "diff:" << std::endl << bstr(actual ^ expected) << std::endl;
  }
}

// TEST(Position, InitialPosition) {
//   Position position = Position::init();
//   Threats<Color::WHITE> threats(position);
//   Bitboard gt = (kRanks[0] | kRanks[1] | kRanks[2]) & ~(bb(SafeSquare::SA8) | bb(SafeSquare::SH8));
//   ASSERT_EQ(threats.badForOur[Piece::KING  ], gt);
//   ASSERT_EQ(threats.badForOur[Piece::QUEEN], gt & ~(bb(SafeSquare::SD8) | bb(SafeSquare::SE8)));
//   ASSERT_EQ(threats.badForOur[Piece::ROOK  ], gt & ~(bb(SafeSquare::SE8) | bb(SafeSquare::SD8) | bb(SafeSquare::SA8) | bb(SafeSquare::SH8)));
//   ASSERT_EQ(threats.badForOur[Piece::BISHOP], gt & ~kRanks[0]);
//   ASSERT_EQ(threats.badForOur[Piece::KNIGHT], gt & ~kRanks[0]);
//   ASSERT_EQ(threats.badForOur[Piece::PAWN  ], gt& ~(kRanks[0] | kRanks[1]));
// }

// TEST(Position, Knight) {
//   Position position = Position();
//   position.place_piece_(ColoredPiece::BLACK_KNIGHT, SafeSquare::SF6);
//   position.place_piece_(ColoredPiece::BLACK_KING, SafeSquare::SA1);
//   position.place_piece_(ColoredPiece::WHITE_KING, SafeSquare::SH8);

//   const Bitboard attackedByBlackKnight = bb(SafeSquare::SG8) | bb(SafeSquare::SE8) | bb(SafeSquare::SH7) | bb(SafeSquare::SD7) | bb(SafeSquare::SD5) | bb(SafeSquare::SH5) | bb(SafeSquare::SG4) | bb(SafeSquare::SE4);
//   const Bitboard attackedByBlackKing = bb(SafeSquare::SA2) | bb(SafeSquare::SB1) | bb(SafeSquare::SB2);
//   const Bitboard defendedByWhiteKing = bb(SafeSquare::SG8) | bb(SafeSquare::SH7) | bb(SafeSquare::SG7);

//   Threats<Color::WHITE> threats(position);

//   ASSERT_EQ(threats.badForOur[Piece::PAWN], (attackedByBlackKnight | attackedByBlackKing) & ~defendedByWhiteKing);
//   ASSERT_EQ(threats.badForOur[Piece::KNIGHT], (attackedByBlackKnight | attackedByBlackKing) & ~defendedByWhiteKing);
//   ASSERT_EQ(threats.badForOur[Piece::BISHOP], (attackedByBlackKnight | attackedByBlackKing) & ~defendedByWhiteKing);
//   ASSERT_EQ(threats.badForOur[Piece::ROOK], attackedByBlackKnight | attackedByBlackKing);
//   ASSERT_EQ(threats.badForOur[Piece::QUEEN], attackedByBlackKnight | attackedByBlackKing);
// }

TEST(Position, Pawns) {
  {
    // Tests:
    // 1) 2 defenders and 1 attacker
    // 2) 1 defender and 1 attacker
    // 3) 1 defender and 2 attackers
    // 4) 2 defenders and 2 attackers
    //
    // ........
    // p.p..p.p
    // .P....P.
    // P.P.P..P
    // p.p..p.p
    // ........
    // P.P.P..P
    // K......k
    //
    Position pos("8/p1p2p1p/1P4P1/P1P1P2P/p1p2p1p/8/P1P1P2P/K6k w - - 0 1");
    Threats<Color::WHITE> threats(pos);
    // ........
    // ........
    // ....x.x.
    // ........
    // ........
    // ....x.x.
    // ......xx
    // ......x.
    EXPECT_TRUE(AssertBB(threats.badForOur[Piece::PAWN], 4665817174891298816));
  }

  {
    Position pos("1k4K1/1b6/7b/7p/3P2P1/4PP2/1r2PP1n/8 w - - 0 1");
    Threats<Color::WHITE> threats(pos);
    // .k....K.
    // .b......
    // .......b
    // .......p
    // ...P..P.
    // ....PP..
    // .r..PP.n
    // ........

    // x.x.....
    // x.x.....
    // xxx.....
    // .x.x..x.
    // .x....x.
    // .x......
    // x.xxx...
    // .x...x..
    EXPECT_TRUE(AssertBB(threats.badForOur[Piece::PAWN], 2458123455347229957));
  }
}

TEST(Position, KNIGHTS) {
    Position pos("k7/p4p2/pn6/1q6/3N4/3N3K/2p1p3/8 w - - 0 1");
    Threats<Color::WHITE> threats(pos);
    // k.......
    // p.....pp
    // pn....p.
    // .q......
    // ...N....
    // ...N....
    // ........
    // ...K....

    // .xx.x...
    // xx.x....
    // x.x.x.x.
    // x.xxxxxx
    // xxx.....
    // .x.x....
    // .x......
    // .x.x.x..
    EXPECT_TRUE(AssertBB(threats.badForOur[Piece::KNIGHT], 3026992928977652502));
}

TEST(Position, BISHOPS) {
    Position pos("k7/p5pp/pn4p1/1q6/3B4/3B4/8/3K4 w - - 0 1");
    Threats<Color::WHITE> threats(pos);
    // k.......
    // p.....pp
    // pn....p.
    // .q......
    // ...B....
    // ...B....
    // ........
    // ...K....

    // .xx.x...
    // xx.x....
    // x.x..xxx
    // x..x.xxx
    // xxx.....
    // .x.x....
    // ........
    // ........
    EXPECT_TRUE(AssertBB(threats.badForOur[Piece::BISHOP], 2450532176683666198));
}

TEST(Position, ROOKS) {
    Position pos("kn6/p4p2/p7/1q6/3R1p2/3R3K/2p5/r2r4 w - - 0 1");
    Threats<Color::WHITE> threats(pos);
    // kn......
    // p....p..
    // p.......
    // .q......
    // ...R.p..
    // ...R...K
    // ..p.....
    // r..r....

    // .x..x...
    // xx.x....
    // xxx.x.x.
    // x.x.xxxx
    // xxx.....
    // xx.xx.x.
    // xx......
    // .xx.xxxx
    EXPECT_TRUE(AssertBB(threats.badForOur[Piece::ROOK], 17727112647999425298LLU));
}

TEST(Position, DOUBLE_DEFENSE) {
  Position pos("k6K/8/8/8/4n3/5n2/2RN4/8 w - - 0 1");
  Threats<Color::WHITE> threats(pos);
  // k......K
  // ........
  // ........
  // ........
  // ....n...
  // .....n..
  // ..RN....
  // ........

  // .x......
  // xx......
  // ...x.x..
  // ....x.x.
  // ...x...x
  // ......x.
  // ...x.x.x
  // ....x.x.
  AssertBB(threats.badForOur[Piece::KNIGHT], 5811966273326154498);
}

}  // namespace

int main() {
  initialize_geometry();
  initialize_zorbrist();
  testing::InitGoogleTest();
  return RUN_ALL_TESTS();
}
