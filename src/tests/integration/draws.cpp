// g++ src/tests/draws.cpp src/game/geometry.cpp src/game/utils.cpp src/game/Position.cpp src/game/piece_maps.cpp src/game/kpvk.cpp -std=c++20 -I/opt/homebrew/Cellar/googletest/1.14.0/include -L/opt/homebrew/Cellar/googletest/1.14.0/lib -lgtest -o test

#include "gtest/gtest.h"

#include <cassert>
#include <cstdint>
#include <ctime>
#include <cstdlib>

#include <iostream>

#include "../../game/search.h"

using namespace ChessEngine;

SearchResult<Color::WHITE> go(Thinker* thinker, std::string fen, size_t depth) {
  Position position(fen);
  thinker->load_weights_from_file("./src/tests/test-weights.txt");
  GoCommand cmd;
  cmd.pos = position;
  cmd.depthLimit = depth;
  cmd.moves = compute_legal_moves_set(&position);
  return search(thinker, cmd, nullptr);
}

TEST(Draws, ForcedRepetition) {
  Thinker thinker;
  const std::string fen = "k7/2Q5/p7/8/8/8/5rrr/K7 w - - 0 1";
  // k.......
  // ..Q.....
  // p.......
  // ........
  // ........
  // ........
  // .....rrr
  // K.......
  EXPECT_EQ(go(&thinker, fen, 8).score, 0);
}

TEST(Draws, FiftyMoveRule) {
  Thinker thinker;
  // Simple checkmate in 2 moves
  // .k......
  // ........
  // ..K.....
  // ..R.....
  // ........
  // ........
  // ........
  // ........
  EXPECT_EQ(go(&thinker, "1k6/8/2K5/2R5/8/8/8/8 w - - 98 1", 3).score, 0);
  EXPECT_EQ(go(&thinker, "1k6/8/2K5/2R5/8/8/8/8 w - - 97 1", 4).score, -(kCheckmate + 3));
}

TEST(Draws, SimplifyToDrawnKPKV) {
  Thinker thinker;
  SearchResult<Color::WHITE> result = go(&thinker, "6k1/8/8/6p1/5P2/6P1/1K6/8 b - - 0 1", 2);
  EXPECT_EQ(result.score, 0);
  EXPECT_EQ(result.move.uci(), "g5f4");
}



int main(int argc, char *argv[]) {
  testing::InitGoogleTest();

  initialize_geometry();
  initialize_zorbrist();
  initialize_movegen();

  return RUN_ALL_TESTS();
}
