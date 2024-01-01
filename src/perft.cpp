#import "game/search.h"
#import "game/Position.h"
#import "game/movegen.h"
#import "game/utils.h"
#import "game/string_utils.h"

#include <condition_variable>
#include <deque>
#include <iostream>
#include <mutex>
#include <unordered_set>

using namespace ChessEngine;

template<Color TURN>
uint64_t perft(Position *position, uint16_t depthRemaining) {
  if (depthRemaining == 0) {
    return 1;
  }
  ExtMove moves[kMaxNumMoves];
  ExtMove *movesEnd = compute_legal_moves<TURN>(position, moves);
  if (depthRemaining == 1) {
    return movesEnd - moves;
  }
  uint64_t r = 0;
  for (ExtMove *move = moves; move != movesEnd; ++move) {
    make_move<TURN>(position, move->move);
    r += perft<opposite_color<TURN>()>(position, depthRemaining - 1);
    undo<TURN>(position);
  }
  return r;
}

int main(int argc, char *argv[]) {
  initialize_geometry();
  initialize_zorbrist();
  initialize_movegen();

  int depth = std::stoi(argv[1]);
  Position pos(argv[2]);

  auto start = std::chrono::steady_clock::now();
  uint64_t n;
  if (pos.turn_ == Color::WHITE) {
    n = perft<Color::WHITE>(&pos, depth);
  } else {
    n = perft<Color::BLACK>(&pos, depth);
  }
  auto end = std::chrono::steady_clock::now();
  double durationMs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

  std::cout << n << " (" << durationMs << "ms)" << std::endl;
}

