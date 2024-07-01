#include <iostream>
#include <fstream>

#include "game/nnue.h"
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

int main(int argc, char* argv[]) {
  std::shared_ptr<NnueNetwork> network = std::make_shared<NnueNetwork>();

  Position pos = Position::init();
  pos.set_network(network);

  std::cout << network->fastforward() << std::endl;
  std::cout << network->slowforward() << std::endl;

  make_uci_move(&pos, "e2e4");

  std::cout << network->fastforward() << std::endl;
  std::cout << network->slowforward() << std::endl;

  make_uci_move(&pos, "g8f6");

  std::cout << network->fastforward() << std::endl;
  std::cout << network->slowforward() << std::endl;

  make_uci_move(&pos, "d2d4");

  std::cout << network->fastforward() << std::endl;
  std::cout << network->slowforward() << std::endl;

  make_uci_move(&pos, "f6g8");

  std::cout << network->fastforward() << std::endl;
  std::cout << network->slowforward() << std::endl;



  return 0;
}
