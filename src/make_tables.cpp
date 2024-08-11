#include "game/geometry.h"
#include "game/utils.h"
#include "game/Position.h"
#include "game/movegen.h"
#include "game/movegen/sliding.h"
#include "game/Evaluator.h"
#include "game/Thinker.h"
#include "game/nnue.h"
#include "sharded_matrix.h"

#include <thread>

using namespace ChessEngine;
using WriterF32 = ShardedMatrix::Writer<float>;
using WriterB = ShardedMatrix::Writer<bool>;
using WriterI8 = ShardedMatrix::Writer<int8_t>;
using WriterI16 = ShardedMatrix::Writer<int16_t>;

Thinker gThinker;

void write_feature(uint8_t *pieceMaps, NnueFeatures feature, bool value) {
  pieceMaps[feature / 8] |= (value ? 1 : 0) << (feature % 8);
}

void process(const std::vector<std::string>& line, WriterB& tableWriter, WriterI8& featureWriter, WriterI16& evalWriter, WriterI8& turnWriter) {
  Position pos(line[0]);
  std::shared_ptr<DummyNetwork> network = std::make_shared<DummyNetwork>();
  pos.set_network(network);

  bool pieceMaps[NnueFeatures::NF_NUM_FEATURES];
  std::fill_n(pieceMaps, 8 * 12 + 1, 0);
  for (size_t i = 0; i < NnueFeatures::NF_NUM_FEATURES; ++i) {
    pieceMaps[i] = network->x[i] > 0;
  }
  tableWriter.write_row(pieceMaps);

  if (pos.turn_ == Color::WHITE) {
    gThinker.evaluator.score<Color::WHITE>(pos);
  } else {
    gThinker.evaluator.score<Color::BLACK>(pos);
  }
  int8_t features[EF::NUM_EVAL_FEATURES];
  for (size_t i = 0; i < EF::NUM_EVAL_FEATURES; ++i) {
    features[i] = gThinker.evaluator.features[i];
  }
  featureWriter.write_row(features);

  int16_t a = std::stoi(line[1]) + std::stoi(line[2]) / 2;
  evalWriter.write_row(&a);

  int8_t turn = pos.turn_ == Color::WHITE ? 1 : -1;
  turnWriter.write_row(&turn);
}

std::string get_shard_name(size_t n) {
  // return string with 0 padding
  return std::string(5 - std::to_string(n).length(), '0') + std::to_string(n);
}

int main(int argc, char *argv[]) {
  initialize_geometry();
  initialize_zorbrist();
  initialize_movegen();

  gThinker.load_weights_from_file("w2.txt");

  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " <input> <output>" << std::endl;
    return 1;
  }

  const std::string inpath = argv[1];
  const std::string outpath = argv[2];

  std::ifstream infile(inpath);
  WriterB tableWriter(outpath + "-table", { NnueFeatures::NF_NUM_FEATURES });
  WriterI8 featureWriter(outpath + "-features", { EF::NUM_EVAL_FEATURES });
  WriterI16 evalWriter(outpath + "-eval", { 1 });
  WriterI8 turnWriter(outpath + "-turn", { 1 });

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

    process(parts, tableWriter, featureWriter, evalWriter, turnWriter);

    if ((++counter) % 100'000 == 0) {
      double ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - startTime).count();
      std::cout << "Finished " << counter / 1000 << "k in " << ms / 1000 << " seconds" << std::endl;
    }
  }

  return 0;
}
