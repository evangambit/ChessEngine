#include "game/geometry.h"
#include "game/utils.h"
#include "game/Position.h"
#include "game/movegen.h"
#include "game/movegen/sliding.h"
#include "game/Evaluator.h"
#include "game/Thinker.h"
#include "game/nnue.h"

using namespace ChessEngine;

Thinker gThinker;

void write_feature(uint8_t *pieceMaps, NnueFeatures feature, bool value) {
  pieceMaps[feature / 8] |= (value ? 1 : 0) << (feature % 8);
}

void process(const std::vector<std::string>& line, std::ostream &tableFile, std::ostream &evalFile) {
  
  Position pos(line[0]);
  std::shared_ptr<NnueNetwork> network = std::make_shared<NnueNetwork>();
  pos.set_network(network);

  uint8_t pieceMaps[8 * 12 + 1];
  std::fill_n(pieceMaps, 8 * 12 + 1, 0);
  for (size_t i = 0; i < NnueFeatures::NF_NUM_FEATURES; ++i) {
    pieceMaps[i / 8] |= (network->x0(0, i) > 0.5 ? 1 : 0) << (i % 8);
  }

  write_feature(pieceMaps, NnueFeatures::NF_IS_WHITE_TURN, pos.turn_ == Color::WHITE);
  write_feature(pieceMaps, NnueFeatures::NF_WHITE_KING_CASTLING, (pos.currentState_.castlingRights & kCastlingRights_WhiteKing) > 0);
  write_feature(pieceMaps, NnueFeatures::NF_WHITE_QUEEN_CASTLING, (pos.currentState_.castlingRights & kCastlingRights_WhiteQueen) > 0);
  write_feature(pieceMaps, NnueFeatures::NF_BLACK_KING_CASTLING, (pos.currentState_.castlingRights & kCastlingRights_BlackKing) > 0);
  write_feature(pieceMaps, NnueFeatures::NF_BLACK_QUEEN_CASTLING, (pos.currentState_.castlingRights & kCastlingRights_BlackQueen) > 0);

  tableFile.write(reinterpret_cast<char*>(pieceMaps), (8 * 12 + 1) * sizeof(uint8_t));

  int16_t a = std::stoi(line[1]) + std::stoi(line[2]) / 2;
  evalFile.write(reinterpret_cast<char*>(&a), sizeof(int16_t));
}

std::string get_shard_name(size_t n) {
  // return string with 0 padding
  return std::string(5 - std::to_string(n).length(), '0') + std::to_string(n);
}

int main(int argc, char *argv[]) {
  initialize_geometry();
  initialize_zorbrist();
  initialize_movegen();

  gThinker.load_weights_from_file("weights.txt");

  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " <input> <output>" << std::endl;
    return 1;
  }

  const std::string inpath = argv[1];
  const std::string outpath = argv[2];


  const size_t kPositionsPerShard = 65536;

  std::chrono::time_point<std::chrono::system_clock> startTime = std::chrono::system_clock::now();

  {
    size_t shardCounter = 1;
    size_t counter = 0;
    std::ifstream infile(inpath);
    std::ofstream tableFile(outpath + "-tables-" + get_shard_name(shardCounter), std::ios::binary);
    std::ofstream evalFile(outpath + "-eval-" + get_shard_name(shardCounter), std::ios::binary);
    std::string line;
    if (!infile.is_open()) {
      std::cerr << "Could not open file: " << inpath << std::endl;
      return 1;
    }
    while (std::getline(infile, line)) {
      if (line == "") {
        continue;
      }
      std::vector<std::string> parts = split(line, '|');
      if (parts.size() != 4) {
        continue;
      }

      process(parts, tableFile, evalFile);

      if ((++counter) % kPositionsPerShard == 0) {
        // open new shard
        tableFile.close();
        evalFile.close();

        double ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - startTime).count();
        startTime = std::chrono::system_clock::now();

        std::cout << "Finished shard " << shardCounter << " in " << ms / 1000 << " seconds" << std::endl;

        tableFile.open(outpath + "-tables-" + get_shard_name(++shardCounter), std::ios::binary);
        evalFile.open(outpath + "-eval-" + get_shard_name(shardCounter), std::ios::binary);
      }
    }
  }

  return 0;
}
