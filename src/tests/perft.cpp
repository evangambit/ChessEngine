#include "gtest/gtest.h"

#include <cassert>
#include <cstdint>
#include <ctime>
#include <cstdlib>

#import "../game/search.h"
#import "../game/Position.h"
#import "../game/movegen.h"
#import "../game/utils.h"
#import "../game/string_utils.h"

#include <condition_variable>
#include <deque>
#include <iostream>
#include <mutex>
#include <unordered_set>

using namespace ChessEngine;

template<Color TURN>
uint64_t _perft(Position *position, uint16_t depthRemaining) {
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
    r += _perft<opposite_color<TURN>()>(position, depthRemaining - 1);
    undo<TURN>(position);
  }
  return r;
}

uint64_t perft(std::string fen, uint16_t depth) {
  Position position(fen);
  if (position.turn_ == Color::WHITE) {
    return _perft<Color::WHITE>(&position, depth);
  } else {
    return _perft<Color::BLACK>(&position, depth);
  }
}


TEST(Perft, InitialPosition) {
  ASSERT_EQ(perft("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 1), 20);
  ASSERT_EQ(perft("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 2), 400);
  ASSERT_EQ(perft("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 3), 8'902);
  ASSERT_EQ(perft("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 4), 197'281);
}

TEST(Perft, Kiwipete) {
  ASSERT_EQ(perft("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 1), 48);
  ASSERT_EQ(perft("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 2), 2'039);
  ASSERT_EQ(perft("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 3), 97'862);
}

TEST(Perft, Position3) {
  ASSERT_EQ(perft("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 1), 14);
  ASSERT_EQ(perft("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 2), 191);
  ASSERT_EQ(perft("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 3), 28'12);
  ASSERT_EQ(perft("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 4), 43'238);
}

TEST(Perft, Position4) {
  ASSERT_EQ(perft("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 1), 6);
  ASSERT_EQ(perft("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 2), 264);
  ASSERT_EQ(perft("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 3), 9'467);
}

TEST(Perft, Position5) {
  ASSERT_EQ(perft("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 1), 44);
  ASSERT_EQ(perft("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 2), 1'486);
  ASSERT_EQ(perft("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 3), 62'379);
}

TEST(Perft, Position6) {
  ASSERT_EQ(perft("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 1), 46);
  ASSERT_EQ(perft("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 2), 2'079);
  ASSERT_EQ(perft("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 3), 89'890);
}


int main(int argc, char *argv[]) {
  initialize_geometry();
  initialize_zorbrist();
  initialize_movegen();

  testing::InitGoogleTest();
  return RUN_ALL_TESTS();
}

