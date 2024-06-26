#include "gtest/gtest.h"

#include <cassert>
#include <cstdint>
#include <ctime>
#include <cstdlib>

#include <thread>
#include <vector>

#include "../game/TranspositionTable.h"

using namespace ChessEngine;

namespace {

const Move kRandomMove = Move{Square::G1, Square::F3, 0, MoveType::NORMAL};
const uint64_t kRootHash = 1;  // Anything but zero

TEST(Position, DeeperResultsOverrideShallowResults) {
  TranspositionTable tt(1);
  tt.starting_search(kRootHash);
  const uint64_t hash = 1;
  tt.insert<false>(tt.create_cache_result(
    hash,
    1,  // depth
    10,  // eval
    kRandomMove,  // best move
    NodeTypePV,  // node type
    0  // distFromPV
  ));

  tt.insert<false>(tt.create_cache_result(
    hash,
    2,  // depth
    20,  // eval
    kRandomMove,  // best move
    NodeTypePV,  // node type
    0  // distFromPV
  ));

  EXPECT_EQ(tt.find<false>(uint64_t(hash)).eval, 20);
}

TEST(Position, ShallowerResultsIgnored) {
  TranspositionTable tt(1);
  tt.set_size_to_one();  // Forcing collisions.

  tt.starting_search(kRootHash);
  const uint64_t hash = 1;
  tt.insert<false>(tt.create_cache_result(
    hash,
    2,  // depth
    10,  // eval
    kRandomMove,  // best move
    NodeTypePV,  // node type
    0  // distFromPV
  ));

  tt.insert<false>(tt.create_cache_result(
    hash,
    1,  // depth
    20,  // eval
    kRandomMove,  // best move
    NodeTypePV,  // node type
    0  // distFromPV
  ));

  EXPECT_EQ(tt.find<false>(uint64_t(hash)).eval, 10);
}

TEST(Position, PreviousSearchResultsReplaced) {
  TranspositionTable tt(1);
  tt.set_size_to_one();  // Forcing collisions.


  tt.starting_search(kRootHash);
  tt.insert<false>(tt.create_cache_result(
    uint64_t(1),
    2,  // depth
    10,  // eval
    kRandomMove,  // best move
    NodeTypePV,  // node type
    0  // distFromPV
  ));

  // Insert result with lower depth, but it's newer.
  tt.starting_search(kRootHash + 1);
  tt.insert<false>(tt.create_cache_result(
    uint64_t(2),
    1,  // depth
    20,  // eval
    kRandomMove,  // best move
    NodeTypePV,  // node type
    0  // distFromPV
  ));

  EXPECT_EQ(tt.find<false>(uint64_t(2)).eval, 20);
}

TEST(Position, ThreadSafe) {
  TranspositionTable tt(1);
  tt.set_size_to_one();  // Forcing collisions.

  constexpr int kNumberOfThreads = 10;
  constexpr int kResultsPerThread = 10;

  tt.starting_search(kRootHash);
  std::vector<CacheResult> results;
  for (int i = 0; i < kNumberOfThreads * kResultsPerThread; ++i) {
    results.push_back(tt.create_cache_result(
      uint64_t(i), Depth(i), Evaluation(i), kRandomMove, NodeTypePV, 0));
  }

  std::vector<std::thread> threads;
  for (size_t i = 0; i < kNumberOfThreads; ++i) {
    threads.push_back(std::thread([](TranspositionTable *tt, const std::vector<CacheResult>& results, int start, int end) {
        for (int j = start; j < end; ++j) {
          tt->insert<true>(results[j]);
        }
      },
      &tt,
      results,
      i * kResultsPerThread,
      (i + 1) * kResultsPerThread
    ));
  }
  for (size_t i = 0; i < threads.size(); ++i) {
    threads[i].join();
  }

  for (int i = 0; i < results.size(); ++i) {
    CacheResult r = tt.find<false>(results[i].positionHash);
    if (!isNullCacheResult(r)) {
      std::cout << r << std::endl;
    }
  }
}

TEST(Position, PreviousSearchResultsNotIgnored) {
  TranspositionTable tt(1);
  tt.set_size_to_one();  // Forcing collisions.

  tt.starting_search(kRootHash);
  auto firstResult = tt.create_cache_result(
    uint64_t(1),
    9,  // depth
    10,  // eval
    kRandomMove,  // best move
    NodeTypePV,  // node type
    0  // distFromPV
  );
  tt.insert<false>(firstResult);

  // Start new search, but old results are still available.
  tt.starting_search(kRootHash + 1);
  EXPECT_EQ(tt.find<false>(uint64_t(firstResult.positionHash)).eval, firstResult.eval);

  // Much shallower than first result.
  auto secondResult = tt.create_cache_result(
    uint64_t(2),
    5,  // depth
    20,  // eval
    kRandomMove,  // best move
    NodeTypePV,  // node type
    0  // distFromPV
  );

  // New result doesn't overwrite old result because it's much shallower.
  tt.insert<false>(secondResult);
  EXPECT_EQ(tt.find<false>(uint64_t(firstResult.positionHash)).eval, firstResult.eval);

  // Need to start two new searches, bc our "find" above marked firstResult as "useful".
  tt.starting_search(kRootHash + 2);
  tt.starting_search(kRootHash + 3);

  // New result *does* overwrite old result because it wasn't used in the last search.
  tt.insert<false>(secondResult);
  EXPECT_EQ(tt.find<false>(uint64_t(secondResult.positionHash)).eval, secondResult.eval);
}

}  // namespace

int main() {
  testing::InitGoogleTest();
  return RUN_ALL_TESTS();
}
