#include "gtest/gtest.h"

#include <cstdint>
#include <ctime>
#include <cstdlib>

#include <iostream>

#include "../game/movegen.h"

using namespace ChessEngine;

TEST(PinMaskTests, DiagonalPin) {
  Position pos("8/8/5b2/8/3N4/2K5/8/8 w - - 0 1");
  PinMasks pm = compute_pin_masks<Color::WHITE>(pos, lsb(pos.pieceBitboards_[ColoredPiece::WHITE_KING]));
  ASSERT_EQ((pm.northeast & pos.pieceBitboards_[ColoredPiece::WHITE_KNIGHT]) > 0, true);
}

TEST(PinMaskTests, DiagonalDiscoveredAttack) {
  Position pos("8/8/5b2/4r3/8/2K5/8/8 w - - 0 1");
  PinMasks pm = compute_pin_masks<Color::WHITE>(pos, lsb(pos.pieceBitboards_[ColoredPiece::WHITE_KING]));
  ASSERT_EQ((pm.northeast & pos.pieceBitboards_[ColoredPiece::BLACK_ROOK]) > 0, true);
}

TEST(PinMaskTests, VerticalDiscoveredAttack) {
  Position pos("8/2r5/8/2b5/8/2K5/8/8 w - - 0 1");
  PinMasks pm = compute_pin_masks<Color::WHITE>(pos, lsb(pos.pieceBitboards_[ColoredPiece::WHITE_KING]));
  ASSERT_EQ((pm.vertical & pos.pieceBitboards_[ColoredPiece::BLACK_BISHOP]) > 0, true);
}

int main(int argc, char *argv[]) {
  initialize_sliding();
  testing::InitGoogleTest();
  return RUN_ALL_TESTS();
}
