#include "gtest/gtest.h"

#include <cassert>
#include <cstdint>
#include <ctime>
#include <cstdlib>

#include <vector>

#include "../game/Position.h"

using namespace ChessEngine;

namespace {

const Move Nf3 = Move{Square::G1, Square::F3, 0, MoveType::NORMAL};
const Move Ng1 = Move{Square::F3, Square::G1, 0, MoveType::NORMAL};
const Move Nf6 = Move{Square::G8, Square::F6, 0, MoveType::NORMAL};
const Move Ng8 = Move{Square::F6, Square::G8, 0, MoveType::NORMAL};

TEST(Position, Repetition3) {
  Position position = Position::init();

  bool isDraw = false;
  for (int loop = 1; loop <= 3; ++loop) {
    ez_make_move(&position, Nf3);
    ASSERT_EQ(position.is_draw_assuming_no_checkmate(), isDraw);
    ez_make_move(&position, Nf6);
    ASSERT_EQ(position.is_draw_assuming_no_checkmate(), isDraw);
    ez_make_move(&position, Ng1);
    ASSERT_EQ(position.is_draw_assuming_no_checkmate(), isDraw);
    ez_make_move(&position, Ng8);
    isDraw |= (loop == 2);
    ASSERT_EQ(position.is_draw_assuming_no_checkmate(), isDraw);
  }
}

// While searching, a single repetition since the root node is enough to
// make something a draw.
TEST(Position, Repetition2) {
  Position position = Position::init();

  ez_make_move(&position, Nf3);
  ez_make_move(&position, Nf6);
  ez_make_move(&position, Ng1);
  ez_make_move(&position, Ng8);

  ASSERT_FALSE(position.is_draw_assuming_no_checkmate(3));
  ASSERT_TRUE(position.is_draw_assuming_no_checkmate(4));

  ez_make_move(&position, Nf3);
  ez_make_move(&position, Nf6);
  ez_make_move(&position, Ng1);
  ez_make_move(&position, Ng8);

  // This is true now, since there was a repetition before the root.
  ASSERT_TRUE(position.is_draw_assuming_no_checkmate(3));
}

}  // namespace

int main() {
  initialize_geometry();
  initialize_zorbrist();
  testing::InitGoogleTest();
  return RUN_ALL_TESTS();
}
