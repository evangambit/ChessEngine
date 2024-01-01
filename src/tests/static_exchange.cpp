// g++ src/tests/static_exchange.cpp src/game/*.cpp -std=c++20 -I/opt/homebrew/Cellar/googletest/1.14.0/include -L/opt/homebrew/Cellar/googletest/1.14.0/lib -lgtest -o static_exchange_tests

#include "gtest/gtest.h"

#include <cassert>
#include <cstdint>
#include <ctime>
#include <cstdlib>

#include <iostream>

#include "../game/Position.h"
#include "../game/movegen.h"

using namespace ChessEngine;

constexpr int kQueenValue = 9;
constexpr int kRookValue = 5;
constexpr int kKnightValue = 3;

// typedef StaticExchangeAnalysis::static_exchange static_exchange;

TEST(StaticExchange, Test1) {
  Position pos("8/1k6/8/8/5q2/8/1K3R2/8 w - - 0 1");
  ASSERT_EQ(
    StaticExchangeAnalysis::static_exchange<Color::WHITE>(&pos),
    kQueenValue
  );
}

TEST(StaticExchange, Test2) {
  Position pos("8/1k6/8/4p3/5q2/8/1K3R2/8 w - - 0 1");
  ASSERT_EQ(
    StaticExchangeAnalysis::static_exchange<Color::WHITE>(&pos),
    kQueenValue - kRookValue
  );
}

TEST(StaticExchange, Test3) {
  // Nxf4  # rook takes queen
  // Rxf4  # knight takes rook
  Position pos("8/1k6/8/8/5q2/1K1n4/5R2/8 w - - 0 1");
  ASSERT_EQ(
    StaticExchangeAnalysis::static_exchange<Color::WHITE>(&pos),
    kQueenValue - kRookValue
  );
}

int main(int argc, char *argv[]) {
  testing::InitGoogleTest();
  return RUN_ALL_TESTS();
}
