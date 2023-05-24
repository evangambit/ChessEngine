// g++ src/tests/pinmask.cpp src/game/*.cpp -std=c++20 -o test

#include <cassert>
#include <cstdint>
#include <ctime>
#include <cstdlib>

#include <iostream>

#include "../game/movegen.h"

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

void test_diagonal_pin() {
  Position pos("8/8/5b2/8/3N4/2K5/8/8 w - - 0 1");
  PinMasks pm = compute_pin_masks<Color::WHITE>(pos, lsb(pos.pieceBitboards_[ColoredPiece::WHITE_KING]));
  assert_equal(
    (pm.northeast & pos.pieceBitboards_[ColoredPiece::WHITE_KNIGHT]) > 0,
    true,
    "no-zone test failed: Square::D2, Square::G8, Square::D4, false"
  );
}

void test_diagonal_discovered_attack() {
  Position pos("8/8/5b2/4r3/8/2K5/8/8 w - - 0 1");
  PinMasks pm = compute_pin_masks<Color::WHITE>(pos, lsb(pos.pieceBitboards_[ColoredPiece::WHITE_KING]));
  assert_equal(
    (pm.northeast & pos.pieceBitboards_[ColoredPiece::BLACK_ROOK]) > 0,
    true,
    "no-zone test failed: Square::D2, Square::G8, Square::D4, false"
  );
}

void test_vertical_discovered_attack() {
  Position pos("8/2r5/8/2b5/8/2K5/8/8 w - - 0 1");
  PinMasks pm = compute_pin_masks<Color::WHITE>(pos, lsb(pos.pieceBitboards_[ColoredPiece::WHITE_KING]));
  assert_equal(
    (pm.vertical & pos.pieceBitboards_[ColoredPiece::BLACK_BISHOP]) > 0,
    true,
    "no-zone test failed: Square::D2, Square::G8, Square::D4, false"
  );
}

int main(int argc, char *argv[]) {
  initialize_sliding();
  test_diagonal_pin();
  test_diagonal_discovered_attack();
  test_vertical_discovered_attack();
  return 0;
}