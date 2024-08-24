#include <iostream>
#include <fstream>

#if NNUE_EVAL
#include "game/nnue.h"
#endif

#include "game/movegen.h"
#include "game/Position.h"

using namespace ChessEngine;

void make_uci_move(Position *pos, std::string uciMove) {
  ExtMove moves[256];
  ExtMove *end;
  if (pos->turn_ == Color::WHITE) {
    end = compute_legal_moves<Color::WHITE>(pos, moves);
  } else {
    end = compute_legal_moves<Color::BLACK>(pos, moves);
  }
  ExtMove *move;
  for (move = moves; move < end; ++move) {
    if (move->move.uci() == uciMove) {
      break;
    }
  }
  if (move->move.uci() != uciMove) {
    throw std::runtime_error("Unrecognized uci move \"" + uciMove + "\"");
    exit(1);
  }
  if (pos->turn_ == Color::WHITE) {
    make_move<Color::WHITE>(pos, move->move);
  } else {
    make_move<Color::BLACK>(pos, move->move);
  }
}

float sigmoid(float x) {
  return 1.0 / (1.0 + std::exp(-x));
}

int main(int argc, char* argv[]) {
  std::shared_ptr<NnueNetwork> network = std::make_shared<NnueNetwork>();
  network->load("nnue.bin");

  // "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
  // "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
  // "rnbqkbnr/pppppppp/8/8/3PP3/2N2N2/PPP2PPP/R1BQKB1R w KQkq - 5 5"
  // "rnbqkb1r/ppp1pppp/5n2/3p4/5P2/8/PPPPP1PP/RNBQKBNR b KQkq - 3 3"
  // "r1b1k2r/ppq1bppp/2n2n2/2ppp3/8/8/PPPPPPPP/RNBQKBNR b KQkq - 3 8"
// tensor([[0.5208],
//         [0.5328],
//         [0.9791],
//         [0.3459],
//         [0.0177]], grad_fn=<SigmoidBackward0>)


  for (size_t i = 0; i < 2; ++i) {
    {
      Position pos("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
      pos.set_network(network);
      std::cout << sigmoid(network->slowforward()) << std::endl;
    }

    {
      // Position pos = Position::init();
      Position pos("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1");
      pos.set_network(network);
      // make_move<Color::WHITE>(&pos, Move{Square::E2, Square::E4});
      std::cout << sigmoid(network->slowforward()) << std::endl;
    }

    {
      Position pos("rnbqkbnr/pppppppp/8/8/3PP3/2N2N2/PPP2PPP/R1BQKB1R w KQkq - 5 5");
      pos.set_network(network);
      std::cout << sigmoid(network->slowforward()) << std::endl;
    }

    {
      Position pos("rnbqkb1r/ppp1pppp/5n2/3p4/5P2/8/PPPPP1PP/RNBQKBNR b KQkq - 3 3");
      pos.set_network(network);
      std::cout << sigmoid(network->slowforward()) << std::endl;
    }

    {
      Position pos("r1b1k2r/ppq1bppp/2n2n2/2ppp3/8/8/PPPPPPPP/RNBQKBNR b KQkq - 3 8");
      pos.set_network(network);
      std::cout << sigmoid(network->slowforward()) << std::endl;
    }

    std::cout << std::endl;
  }

  return 0;
}
