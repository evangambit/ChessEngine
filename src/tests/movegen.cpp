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

int main(int argc, char *argv[]) {
  initialize_sliding();
  testing::InitGoogleTest();
  return RUN_ALL_TESTS();
}
