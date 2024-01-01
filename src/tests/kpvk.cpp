// g++ src/tests/kpvk.cpp src/game/kpvk.cpp src/game/geometry.cpp src/game/utils.cpp -std=c++20 -I/opt/homebrew/Cellar/googletest/1.14.0/include -L/opt/homebrew/Cellar/googletest/1.14.0/lib -lgtest -o test

#include "gtest/gtest.h"

#include <cassert>
#include <cstdint>
#include <ctime>
#include <cstdlib>

#include <iostream>

#include "../game/kpvk.h"

using namespace ChessEngine;

TEST(KPVK, SquareRule) {
  ASSERT_EQ(known_kpvk_result(Square::H1, Square::F6, Square::B5, true), 2);
  ASSERT_NE(known_kpvk_result(Square::H1, Square::F6, Square::B5, false), 2);
}

void test_key_squares() {
  Square yourKing = Square::E4;
  Square yourPawn = Square::E2;
  for (int yourMove = 0; yourMove <= 1; ++yourMove) {
    for (int i = 0; i < 24; ++i) {
      Square theirKing = Square(i);
      ASSERT_EQ(known_kpvk_result(yourKing, theirKing, yourPawn, yourMove), 2);
    }
  }
}

void test_no_zones() {
  ASSERT_EQ(known_kpvk_result(Square::D2, Square::F8, Square::D4, true), 0);
  ASSERT_EQ(known_kpvk_result(Square::D2, Square::G8, Square::D4, false), 0);
  // No-zones can only prove a position is a draw, not that it is winning.
  ASSERT_NE(known_kpvk_result(Square::D2, Square::G8, Square::D4, true), 0);
  ASSERT_EQ(known_kpvk_result(Square::D3, Square::F8, Square::D4, false), 0);
  ASSERT_NE(known_kpvk_result(Square::D3, Square::F8, Square::D4, true), 0);
}

int main(int argc, char *argv[]) {
  testing::InitGoogleTest();
  return RUN_ALL_TESTS();
}
