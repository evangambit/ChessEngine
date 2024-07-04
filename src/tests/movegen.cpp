#include "gtest/gtest.h"

#include <cstdint>
#include <ctime>
#include <cstdlib>

#include <iostream>

#include "../game/movegen.h"

using namespace ChessEngine;

TEST(PinMaskTests, DiagonalPin) {
  Position pos("3k4/1p1b1p2/P2bp2p/3p4/6P1/2N2K1q/P1P1BP2/R4R2 w - - 2 24");
  ExtMove moves[kMaxNumMoves];
  ExtMove *end = compute_moves<Color::WHITE, MoveGenType::ALL_MOVES>(pos, moves);
  ASSERT_EQ(end - moves, 0);
}

TEST(PinMaskTests, DiagonalPin2) {
  Position pos("r3k2r/p2pqpb1/bnp1pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R4K1R w kq - 0 2");
  ExtMove moves[kMaxNumMoves];
  ExtMove *end = compute_moves<Color::WHITE, MoveGenType::ALL_MOVES>(pos, moves);
  for (ExtMove *move = moves; move < end; ++move) {
    ASSERT_NE(move->uci(), "e2d1");
  }
}

int main(int argc, char *argv[]) {
  initialize_sliding();
  testing::InitGoogleTest();
  return RUN_ALL_TESTS();
}
