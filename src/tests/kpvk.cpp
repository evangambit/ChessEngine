// g++ src/tests/kpvk.cpp src/game/kpvk.cpp src/game/geometry.cpp src/game/utils.cpp -std=c++20 -o test

#include <cassert>
#include <cstdint>
#include <ctime>
#include <cstdlib>

#include <iostream>

#include "../game/kpvk.h"

using namespace ChessEngine;

template<class T>
void assert_equal(const T& a, const T& b, const std::string failureMessage = "") {
  if (a != b) {
    std::cout << "Assertion Failed: " << a << " != " << b << std::endl;
    if (failureMessage.size() > 0) {
      std::cout << failureMessage << std::endl;
    }
  }
}

template<class T>
void assert_not_equal(const T& a, const T& b, const std::string failureMessage = "") {
  if (a == b) {
    std::cout << "Assertion Failed: " << a << " != " << b << std::endl;
    if (failureMessage.size() > 0) {
      std::cout << failureMessage << std::endl;
    }
  }
}

void test_square_rule() {
  assert_equal(
    known_kpvk_result(Square::H1, Square::F6, Square::B5, true),
    2,
    "no-zone test failed: Square::D2, Square::G8, Square::D4, false"
  );
  assert_not_equal(
    known_kpvk_result(Square::H1, Square::F6, Square::B5, false),
    2,
    "no-zone test failed: Square::D2, Square::G8, Square::D4, false"
  );
}

void test_key_squares() {
  Square yourKing = Square::E4;
  Square yourPawn = Square::E2;
  for (int yourMove = 0; yourMove <= 1; ++yourMove) {
    for (int i = 0; i < 24; ++i) {
      Square theirKing = Square(i);
      assert_equal(
        known_kpvk_result(yourKing, theirKing, yourPawn, yourMove),
        2,
        "key square test failed"
      );
    }
  }
}

void test_no_zones() {
  assert_equal(
    known_kpvk_result(Square::D2, Square::F8, Square::D4, true),
    0,
    "no-zone test failed: Square::D2, Square::F8, Square::D4, true"
  );
  assert_equal(
    known_kpvk_result(Square::D2, Square::G8, Square::D4, false),
    0,
    "no-zone test failed: Square::D2, Square::G8, Square::D4, false"
  );
  assert_not_equal(    // No-zones can only prove a position is a draw, not that it is winning.
    known_kpvk_result(Square::D2, Square::G8, Square::D4, true),
    0,
    "no-zone test failed: Square::D2, Square::G8, Square::D4, true"
  );


  assert_equal(
    known_kpvk_result(Square::D3, Square::F8, Square::D4, false),
    0,
    "no-zone test failed: Square::D3, Square::F8, Square::D4, false"
  );
  assert_not_equal(
    known_kpvk_result(Square::D3, Square::F8, Square::D4, true),
    0,
    "no-zone test failed: Square::D3, Square::F8, Square::D4, true"
  );
}

int main(int argc, char *argv[]) {
  test_square_rule();
  test_key_squares();
  test_no_zones();
  return 0;
}