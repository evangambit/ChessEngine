#include "gtest/gtest.h"

#include <cassert>
#include <cstdint>
#include <ctime>
#include <cstdlib>

#include <iostream>

#include "../../game/search.h"
#include "../../game/Thinker.h"

using namespace ChessEngine;

SearchResult<Color::WHITE> go(Thinker* thinker, std::string fen, size_t depth) {
  Position position(fen);
  {
    std::ifstream file;
    file.open("./src/tests/test-weights.txt");
    EXPECT_TRUE(file.is_open());
    thinker->load_weights(file);
  }
  GoCommand cmd;
  cmd.pos = position;
  cmd.depthLimit = depth;
  cmd.moves = compute_legal_moves_set(&position);
  return Search::search(thinker, cmd, nullptr);
}

TEST(Mates, CheckmateScoreCorrect) {
  Thinker thinker;
  std::cout << thinker.get_variations() << std::endl;
  SearchResult<Color::WHITE> result = go(&thinker, "1k6/3R4/2K5/8/8/8/8/8 w - - 12 7", 6);
  std::cout << result << std::endl;
  EXPECT_EQ(result.score, -kCheckmate - 5);
}

int main(int argc, char *argv[]) {
  testing::InitGoogleTest();

  initialize_geometry();
  initialize_zorbrist();
  initialize_movegen();

  return RUN_ALL_TESTS();
}
