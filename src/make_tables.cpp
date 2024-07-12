#include "game/geometry.h"
#include "game/utils.h"
#include "game/Position.h"
#include "game/movegen.h"
#include "game/movegen/sliding.h"
#include "game/Evaluator.h"
#include "game/Thinker.h"
#include "game/nnue.h"
#include "sharded_matrix.h"

using namespace ChessEngine;

Thinker gThinker;

void write_feature(uint8_t *pieceMaps, NnueFeatures feature, bool value) {
  pieceMaps[feature / 8] |= (value ? 1 : 0) << (feature % 8);
}

void process(const std::vector<std::string>& line, ShardedWriter<bool>& tableWriter, ShardedWriter<int16_t>& evalWriter) {
  Position pos(line[0]);
  std::shared_ptr<DummyNetwork> network = std::make_shared<DummyNetwork>();
  pos.set_network(network);

  bool pieceMaps[NnueFeatures::NF_NUM_FEATURES];
  std::fill_n(pieceMaps, 8 * 12 + 1, 0);
  for (size_t i = 0; i < NnueFeatures::NF_NUM_FEATURES; ++i) {
    pieceMaps[i] = network->x[i] > 0;
  }
  tableWriter.write_row(pieceMaps);

  int16_t a = std::stoi(line[1]) + std::stoi(line[2]) / 2;
  evalWriter.write_row(&a);
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

  std::ifstream infile(inpath);
  ShardedWriter<bool> tableWriter(outpath + "-table", { NnueFeatures::NF_NUM_FEATURES });
  ShardedWriter<int16_t> evalWriter(outpath + "-eval", { 1 });

  std::chrono::time_point<std::chrono::system_clock> startTime = std::chrono::system_clock::now();

  size_t counter = 0;
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

    process(parts, tableWriter, evalWriter);

    if ((++counter) % 100'000 == 0) {
      double ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - startTime).count();
      std::cout << "Finished " << counter / 1000 << "k in " << ms / 1000 << " seconds" << std::endl;
    }
  }

  return 0;
}
