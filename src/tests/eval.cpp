#include "gtest/gtest.h"

#include <cassert>
#include <cstdint>
#include <ctime>
#include <cstdlib>

#include <fstream>
#include <iostream>

#include "../game/Evaluator.h"
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

// TEST(Eval, IN_CHECK) {
//   // No check.
//   ASSERT_EQ(f("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", EF::IN_CHECK), 0);

//   // Pawn check.
//   ASSERT_EQ(f("8/1k6/2P5/3K4/8/8/8/8 b - - 0 1", EF::IN_CHECK), 1);

//   // Knight check.
//   ASSERT_EQ(f("8/1k6/8/2NK4/8/8/7P/8 b - - 0 1", EF::IN_CHECK), 1);

//   // Bishop check.
//   ASSERT_EQ(f("8/1k6/8/2KB4/8/8/7P/8 b - - 0 1", EF::IN_CHECK), 1);

//   // Rook check.
//   ASSERT_EQ(f("8/1k6/8/1RK5/8/8/7P/8 b - - 0 1", EF::IN_CHECK), 1);

//   // Queen check.
//   ASSERT_EQ(f("8/1k6/8/1QK5/8/8/7P/8 b - - 0 1", EF::IN_CHECK), 1);
// }

// TEST(Eval, KING_ON_BACK_RANK) {
//   // Starting position
//   ASSERT_EQ(f("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", EF::KING_ON_BACK_RANK), 0);

//   // Bong cloud
//   ASSERT_EQ(f("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPPKPPP/RNBQ1BNR b kq - 0 1", EF::KING_ON_BACK_RANK), 1);
// }

// TEST(Eval, KING_ON_CENTER_FILE) {
//   // Starting position
//   ASSERT_EQ(f("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", EF::KING_ON_CENTER_FILE), 0);

//   ASSERT_EQ(f("8/k7/8/8/8/4K3/7P/8 w - - 0 1", EF::KING_ON_CENTER_FILE), 1);

//   ASSERT_EQ(f("r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1 b kq - 5 4", EF::KING_ON_CENTER_FILE), 1);
// }

TEST(Eval, THREATS_NEAR_KING) {
  // Starting position
  // ASSERT_EQ(f("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", EF::THREATS_NEAR_KING_2), 0);
  // ASSERT_EQ(f("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", EF::THREATS_NEAR_KING_3), 0);
  // ASSERT_EQ(f("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", EF::THREATS_NEAR_THEIR_KING), 0);
  // ASSERT_EQ(f("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", EF::THREATS_NEAR_OUR_KING), 0);
  // ASSERT_EQ(f("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", EF::QUEEN_THREATS_NEAR_KING), 0);

  // ASSERT_EQ(f("6k1/8/8/8/3B4/3B4/1K6/8 w - - 0 1", EF::THREATS_NEAR_KING_2), -2);
  // ASSERT_EQ(f("6k1/8/8/8/3B4/3B4/1K6/8 w - - 0 1", EF::THREATS_NEAR_KING_3), -4);
  // ASSERT_EQ(f("6k1/8/8/8/3B4/3B4/1K6/8 w - - 0 1", EF::THREATS_NEAR_THEIR_KING), 0);
  // ASSERT_EQ(f("6k1/8/8/8/3B4/3B4/1K6/8 w - - 0 1", EF::THREATS_NEAR_OUR_KING), 0);
  // ASSERT_EQ(f("6k1/8/8/8/3B4/3B4/1K6/8 w - - 0 1", EF::QUEEN_THREATS_NEAR_KING), 0);

  // ASSERT_EQ(f("6k1/8/8/8/3B4/3B4/1K6/8 w - - 0 1", EF::THREATS_NEAR_KING_2), -2);
  // ASSERT_EQ(f("6k1/8/8/8/3B4/3B4/1K6/8 w - - 0 1", EF::THREATS_NEAR_KING_3), -4);
  // ASSERT_EQ(f("6k1/8/8/5N2/3B4/3B4/1K6/8 w - - 0 1", EF::THREATS_NEAR_THEIR_KING), 1);
  // ASSERT_EQ(f("6k1/8/8/5N2/3B4/3B4/1K6/8 w - - 0 1", EF::THREATS_NEAR_OUR_KING), 0);
  // ASSERT_EQ(f("6k1/8/8/5N2/3B4/3B4/1K6/8 w - - 0 1", EF::QUEEN_THREATS_NEAR_KING), 0);

  ASSERT_EQ(f("6k1/8/8/5N2/3B4/3B4/1K6/8 w - - 0 1", EF::THREATS_NEAR_THEIR_KING), 2);
  // ASSERT_EQ(f("6k1/Q7/8/7N/3B4/3B4/1K6/8 w - - 0 1", EF::QUEEN_THREATS_NEAR_KING), -2);

//   QUEEN_THREATS_NEAR_KING
// THREATS_NEAR_OUR_KING
}



int main(int argc, char *argv[]) {
  initialize_geometry();
  initialize_zorbrist();
  testing::InitGoogleTest();
  return RUN_ALL_TESTS();
}
