#include "game/geometry.h"
#include "game/utils.h"
#include "game/Position.h"
#include "game/movegen.h"
#include "game/movegen/sliding.h"
#include "game/Evaluator.h"
#include "game/SquareEmbeddings.h"
#include "game/nnue.h"

#include <bitset>

using namespace ChessEngine;

void process(std::string line, std::ostream &outfile) {
  Position pos(line);
  std::shared_ptr<NnueNetwork> network = std::make_shared<NnueNetwork>();
  pos.set_network(network);

  uint8_t pieceMaps[8 * 12 + 1];
  std::fill_n(pieceMaps, 8 * 12 + 1, 0);
  for (size_t i = 0; i < NnueFeatures::NF_NUM_FEATURES; ++i) {
    pieceMaps[i / 8] |= (network->x0(0, i) > 0.5 ? 1 : 0) << (i % 8);
  }

  // std::fill_n(pieceMaps, 8 * 12, 0);
  // for (int cp = ColoredPiece::WHITE_PAWN; cp <= ColoredPiece::BLACK_KING; ++cp) {
  //   for (int y = 0; y < 8; ++y) {
  //     pieceMaps[(cp - 1) * 8 + y] |= pos.pieceBitboards_[cp] >> (y * 8);
  //   }
  // }

  // // Note: reverse order from "NnueFeatures" bc little endian.
  // pieceMaps[8 * 12] = (pos.turn_ == Color::WHITE) << 0;
  // pieceMaps[8 * 12] |= ((pos.currentState_.castlingRights & kCastlingRights_WhiteKing) > 0) << 1;
  // pieceMaps[8 * 12] |= ((pos.currentState_.castlingRights & kCastlingRights_WhiteQueen) > 0) << 2;
  // pieceMaps[8 * 12] |= ((pos.currentState_.castlingRights & kCastlingRights_BlackKing) > 0) << 3;
  // pieceMaps[8 * 12] |= ((pos.currentState_.castlingRights & kCastlingRights_BlackQueen) > 0) << 4;

  // write out piece maps
  outfile.write(reinterpret_cast<char*>(pieceMaps), (8 * 12 + 1) * sizeof(uint8_t));
}

std::string get_shard_name(size_t n) {
  // return string with 0 padding
  return std::string(5 - std::to_string(n).length(), '0') + std::to_string(n);
}

int main(int argc, char *argv[]) {
  initialize_geometry();
  initialize_zorbrist();
  initialize_movegen();

  const std::string fenpath = argv[1];
  const std::string evalpath = argv[2];
  const std::string outpath = argv[3];
  const size_t kPositionsPerShard = 65536;

  std::chrono::time_point<std::chrono::system_clock> startTime = std::chrono::system_clock::now();

  {
    size_t shardCounter = 1;
    size_t counter = 0;
    std::ifstream infile(fenpath);
    std::ofstream outfile(outpath + "-" + get_shard_name(shardCounter), std::ios::binary);
    std::string line;
    if (!infile.is_open()) {
      std::cerr << "Could not open file: " << fenpath << std::endl;
      return 1;
    }
    while (std::getline(infile, line)) {
      if (line == "") {
        continue;
      }
      process(line, outfile);
      if ((++counter) % kPositionsPerShard == 0) {
        // open new shard
        outfile.close();

        double ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - startTime).count();
        startTime = std::chrono::system_clock::now();

        std::cout << "Finished shard " << shardCounter << " in " << ms / 1000 << " seconds" << std::endl;
        outfile.open(outpath + "-" + get_shard_name(++shardCounter), std::ios::binary);
      }
    }
  }

  {
    size_t shardCounter = 1;
    size_t counter = 0;
    std::ifstream infile(evalpath);
    std::ofstream outfile(outpath + "-scores-" + get_shard_name(shardCounter), std::ios::binary);
    std::string line;
    if (!infile.is_open()) {
      std::cerr << "Could not open file: " << evalpath << std::endl;
      return 1;
    }
    while (std::getline(infile, line)) {
      if (line == "") {
        continue;
      }

      std::vector<std::string> parts = split(line, '|');
      if (parts.size() != 3) {
        std::cerr << "Invalid line \"" << line << "\"" << std::endl;
        return 1;
      }
      int16_t a = std::stoi(parts[0]) + std::stoi(parts[1]) / 2;
      outfile.write(reinterpret_cast<char*>(&a), sizeof(int16_t));

      if ((++counter) % kPositionsPerShard == 0) {
        // open new shard
        outfile.close();
        std::cout << "Done with shard " << shardCounter << std::endl;
        outfile.open(outpath + "-scores-" + get_shard_name(++shardCounter), std::ios::binary);
      }
    }
  }


  return 0;
}
