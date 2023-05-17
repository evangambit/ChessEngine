// g++ src/tests/static_exchange.cpp src/game/*.cpp -std=c++20 -o static_exchange_tests


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

template<class T>
void assert_equal(const T& a, const T& b, const std::string failureMessage = "") {
  if (a != b) {
    std::cout << "Assertion Failed: " << a << " != " << b << std::endl;
    if (failureMessage.size() > 0) {
      std::cout << failureMessage << std::endl;
    }
  }
}

void test1() {
  Position pos("8/1k6/8/8/5q2/8/1K3R2/8 w - - 0 1");
  assert_equal(
    StaticExchangeAnalysis::static_exchange<Color::WHITE>(&pos),
    kQueenValue,
    "test1 failed"
  );
}

void test2() {
  Position pos("8/1k6/8/4p3/5q2/8/1K3R2/8 w - - 0 1");
  assert_equal(
    StaticExchangeAnalysis::static_exchange<Color::WHITE>(&pos),
    kQueenValue - kRookValue,
    "test2 failed"
  );
}

void test3() {
  // Nxf4  # rook takes queen
  // Rxf4  # knight takes rook
  Position pos("8/1k6/8/8/5q2/1K1n4/5R2/8 w - - 0 1");
  assert_equal(
    StaticExchangeAnalysis::static_exchange<Color::WHITE>(&pos),
    kQueenValue - kRookValue,
    "test3 failed"
  );
}

int main(int argc, char *argv[]) {
  test1();
  test2();
  test3();
  return 0;
}