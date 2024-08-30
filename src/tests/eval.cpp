#include "gtest/gtest.h"

#include <cassert>
#include <cstdint>
#include <ctime>
#include <cstdlib>

#include <fstream>
#include <iostream>

#include "../game/Evaluator.h"
#include "../game/movegen.h"
#include "../game/Position.h"

using namespace ChessEngine;

Evaluator make_evaluator() {
  Evaluator e;
  std::ifstream file;
  file.open("./src/tests/test-weights.txt");
  e.load_weights_from_file(file);
  std::fill_n(e.features, EF::NUM_EVAL_FEATURES, 999);
  return e;
}

int16_t f(const std::string fen, EF feature) {
  Position pos(fen);
  Evaluator evaluator = make_evaluator();
  if (pos.turn_ == Color::WHITE) {
    evaluator.score<Color::WHITE>(pos);
  } else {
    evaluator.score<Color::BLACK>(pos);
  }
  return evaluator.features[feature];
}

TEST(Eval, IN_CHECK) {
  // No check.
  ASSERT_EQ(f("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", EF::IN_CHECK), 0);

  // Pawn check.
  ASSERT_EQ(f("8/1k6/2P5/3K4/8/8/8/8 b - - 0 1", EF::IN_CHECK), 1);

  // Knight check.
  ASSERT_EQ(f("8/1k6/8/2NK4/8/8/7P/8 b - - 0 1", EF::IN_CHECK), 1);

  // Bishop check.
  ASSERT_EQ(f("8/1k6/8/2KB4/8/8/7P/8 b - - 0 1", EF::IN_CHECK), 1);

  // Rook check.
  ASSERT_EQ(f("8/1k6/8/1RK5/8/8/7P/8 b - - 0 1", EF::IN_CHECK), 1);

  // Queen check.
  ASSERT_EQ(f("8/1k6/8/1QK5/8/8/7P/8 b - - 0 1", EF::IN_CHECK), 1);
}

TEST(Eval, KING_ON_BACK_RANK) {
  // Starting position
  ASSERT_EQ(f("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", EF::KING_ON_BACK_RANK), 0);

  // Bong cloud
  ASSERT_EQ(f("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPPKPPP/RNBQ1BNR b kq - 0 1", EF::KING_ON_BACK_RANK), 1);
}

TEST(Eval, KING_ON_CENTER_FILE) {
  // Starting position
  ASSERT_EQ(f("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", EF::KING_ON_CENTER_FILE), 0);

  ASSERT_EQ(f("8/k7/8/8/8/4K3/7P/8 w - - 0 1", EF::KING_ON_CENTER_FILE), 1);

  ASSERT_EQ(f("r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1 b kq - 5 4", EF::KING_ON_CENTER_FILE), 1);
}

TEST(Eval, THREATS_NEAR_KING) {
  // Starting position
  ASSERT_EQ(f("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", EF::THREATS_NEAR_KING_2), 0);
  ASSERT_EQ(f("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", EF::THREATS_NEAR_KING_3), 0);
  ASSERT_EQ(f("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", EF::THREATS_NEAR_THEIR_KING), 0);
  ASSERT_EQ(f("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", EF::THREATS_NEAR_OUR_KING), 0);
  ASSERT_EQ(f("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", EF::QUEEN_THREATS_NEAR_KING), 0);

  ASSERT_EQ(f("6k1/8/8/8/3B4/3B4/1K6/8 w - - 0 1", EF::THREATS_NEAR_KING_2), -2);
  ASSERT_EQ(f("6k1/8/8/8/3B4/3B4/1K6/8 w - - 0 1", EF::THREATS_NEAR_KING_3), -4);
  ASSERT_EQ(f("6k1/8/8/8/3B4/3B4/1K6/8 w - - 0 1", EF::THREATS_NEAR_THEIR_KING), 0);
  ASSERT_EQ(f("6k1/8/8/8/3B4/3B4/1K6/8 w - - 0 1", EF::THREATS_NEAR_OUR_KING), 0);
  ASSERT_EQ(f("6k1/8/8/8/3B4/3B4/1K6/8 w - - 0 1", EF::QUEEN_THREATS_NEAR_KING), 0);

  ASSERT_EQ(f("6k1/8/8/8/3B4/3B4/1K6/8 w - - 0 1", EF::THREATS_NEAR_KING_2), -2);
  ASSERT_EQ(f("6k1/8/8/8/3B4/3B4/1K6/8 w - - 0 1", EF::THREATS_NEAR_KING_3), -4);
  ASSERT_EQ(f("6k1/8/8/5N2/3B4/3B4/1K6/8 w - - 0 1", EF::THREATS_NEAR_THEIR_KING), 1);
  ASSERT_EQ(f("6k1/8/8/5N2/3B4/3B4/1K6/8 w - - 0 1", EF::THREATS_NEAR_OUR_KING), 0);
  ASSERT_EQ(f("6k1/8/8/5N2/3B4/3B4/1K6/8 w - - 0 1", EF::QUEEN_THREATS_NEAR_KING), 0);

  ASSERT_EQ(f("6k1/8/8/5N2/3B4/3B4/1K6/8 w - - 0 1", EF::THREATS_NEAR_THEIR_KING), 1);
  ASSERT_EQ(f("6k1/Q7/8/7N/3B4/3B4/1K6/8 w - - 0 1", EF::QUEEN_THREATS_NEAR_KING), -2);

//   QUEEN_THREATS_NEAR_KING
// THREATS_NEAR_OUR_KING
}

TEST(Eval, PINNED_PIECES) {
  // No pins.
  ASSERT_EQ(f("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", EF::PINNED_PIECES), 0);

  // Both side has a pin (cancels each other out).
  ASSERT_EQ(f("4k3/8/2r5/bB6/8/2N5/8/4K3 w - - 0 1", EF::PINNED_PIECES), 0);

  // White has a pin.
  ASSERT_EQ(f("4k3/8/8/b7/8/2N5/8/4K3 w - - 0 1", EF::PINNED_PIECES), 1);

  // White has two pins.
  ASSERT_EQ(f("4k3/4r3/8/b7/4B3/2N5/8/4K3 w - - 0 1", EF::PINNED_PIECES), 2);

  // A bishop cannot pin a bishop.
  ASSERT_EQ(f("4k3/8/8/b7/8/2B5/8/4K3 w - - 0 1", EF::PINNED_PIECES), 0);

  // Pins against the queen count too.
  ASSERT_EQ(f("4k3/8/8/b7/8/2R5/8/4QK2 w - - 0 1", EF::PINNED_PIECES), 1);
}

TEST(Eval, QUEEN_MOVES) {
  // No pins.
  ASSERT_EQ(f("rnbqkbnr/ppp1pppp/8/3p4/3P4/2N5/PPP1PPPP/R1BQKBNR b KQkq - 1 2", EF::QUEEN_MOVES), 0);
}

TEST(Eval, KPVK) {
  // When you have extra material, you should still get a bonus if you're winning the
  // pawn endgame (so you don't prefer less material).
  ASSERT_EQ(f("8/1k6/8/6P1/8/8/8/3KQ3 w - - 0 1", EF::KNOWN_KPVK_WIN), 1);

  // However if your opponent has any material, you should not get a bonus.
  ASSERT_EQ(f("8/pk6/8/6P1/8/8/8/3KQ3 w - - 0 1", EF::KNOWN_KPVK_WIN), 0);

  // When there are two pawns, look at the pawn closer to the end.
  ASSERT_EQ(f("8/1k6/8/6P1/1P6/8/8/3K4 w - - 0 1", EF::KNOWN_KPVK_WIN), 1);
}

int main(int argc, char *argv[]) {
  initialize_geometry();
  initialize_zorbrist();
  initialize_sliding();
  testing::InitGoogleTest();
  return RUN_ALL_TESTS();
}
