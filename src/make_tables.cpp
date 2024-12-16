#include "game/geometry.h"
#include "game/utils.h"
#include "game/Position.h"
#include "game/movegen.h"
#include "game/movegen/sliding.h"
#include "game/Evaluator.h"
#include "game/Thinker.h"
#include "sharded_matrix.h"

#if NNUE_EVAL
#include "game/nnue.h"
#endif

#define PST 1
#define NNUE 1

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

void process(
  const std::vector<std::string>& line
  #if NNUE
  , WriterB& tableWriter
  #endif
  , WriterI8& featureWriter
  , WriterI16& evalWriter
  , WriterI8& turnWriter
  , WriterF32& timeWriter
  , WriterI8& pieceCountWriter
  #if PST
  , WriterI8& pstWriter
  #endif
  ) {
  if (line.size() != 4) {
    std::cout << "error: line has " << line.size() << " elements" << std::endl;
    return;
  }

  Position pos(line[0]);

  Evaluator& evaluator = ((ThinkerInterface *)(&gThinker))->get_evaluator();

  if (pos.turn_ == Color::WHITE) {
    evaluator.score<Color::WHITE>(pos);
  } else {
    evaluator.score<Color::BLACK>(pos);
  }
  if (evaluator.features[EF::KNOWN_DRAW] || evaluator.features[EF::KNOWN_KPVK_DRAW]) {
    return;
  }

  #if NNUE
  std::shared_ptr<DummyNetwork> network = std::make_shared<DummyNetwork>();
  pos.set_network(network);

  bool pieceMaps[NnueFeatures::NF_NUM_FEATURES];
  std::fill_n(pieceMaps, 8 * 12 + 1, 0);
  for (size_t i = 0; i < NnueFeatures::NF_NUM_FEATURES; ++i) {
    pieceMaps[i] = network->x[i] > 0;
  }
  tableWriter.write_row(pieceMaps);
  #endif

  #if PST
  int8_t pst[6 * kNumSquares];
  for (Square sq = Square(0); sq < Square::NO_SQUARE; sq = Square(sq + 1)) {
    int8_t x = sq % 8;
    int8_t y = sq / 8;
    int8_t op = (7 - y) * 8 + x;
    pst[sq * 6 + 0] = (pos.tiles_[sq] == ColoredPiece::WHITE_PAWN) - (pos.tiles_[op] == ColoredPiece::BLACK_PAWN);
    pst[sq * 6 + 1] = (pos.tiles_[sq] == ColoredPiece::WHITE_KNIGHT) - (pos.tiles_[op] == ColoredPiece::BLACK_KNIGHT);
    pst[sq * 6 + 2] = (pos.tiles_[sq] == ColoredPiece::WHITE_BISHOP) - (pos.tiles_[op] == ColoredPiece::BLACK_BISHOP);
    pst[sq * 6 + 3] = (pos.tiles_[sq] == ColoredPiece::WHITE_ROOK) - (pos.tiles_[op] == ColoredPiece::BLACK_ROOK);
    pst[sq * 6 + 4] = (pos.tiles_[sq] == ColoredPiece::WHITE_QUEEN) - (pos.tiles_[op] == ColoredPiece::BLACK_QUEEN);
    pst[sq * 6 + 5] = (pos.tiles_[sq] == ColoredPiece::WHITE_KING) - (pos.tiles_[op] == ColoredPiece::BLACK_KING);
  }
  pstWriter.write_row(pst);
  #endif

  int8_t features[EF::NUM_EVAL_FEATURES];
  for (size_t i = 0; i < EF::NUM_EVAL_FEATURES; ++i) {
    features[i] = evaluator.features[i];
  }
  featureWriter.write_row(features);

  // int16_t a[3] = {
  //   (int16_t) (std::stof(line[2]) * 1000),
  //   (int16_t) (std::stof(line[4]) * 1000),
  //   (int16_t) (std::stof(line[6]) * 1000)
  // };
  // evalWriter.write_row(a);
  int16_t a = std::stof(line[1]) + std::stof(line[2]) * 0.5;
  evalWriter.write_row(&a);

  int8_t turn = pos.turn_ == Color::WHITE ? 1 : -1;
  turnWriter.write_row(&turn);

  int8_t pieceCounts[10];
  pieceCounts[0] = std::popcount(pos.pieceBitboards_[ColoredPiece::WHITE_PAWN]);
  pieceCounts[1] = std::popcount(pos.pieceBitboards_[ColoredPiece::WHITE_KNIGHT]);
  pieceCounts[2] = std::popcount(pos.pieceBitboards_[ColoredPiece::WHITE_BISHOP]);
  pieceCounts[3] = std::popcount(pos.pieceBitboards_[ColoredPiece::WHITE_ROOK]);
  pieceCounts[4] = std::popcount(pos.pieceBitboards_[ColoredPiece::WHITE_QUEEN]);
  pieceCounts[5] = std::popcount(pos.pieceBitboards_[ColoredPiece::BLACK_PAWN]);
  pieceCounts[6] = std::popcount(pos.pieceBitboards_[ColoredPiece::BLACK_KNIGHT]);
  pieceCounts[7] = std::popcount(pos.pieceBitboards_[ColoredPiece::BLACK_BISHOP]);
  pieceCounts[8] = std::popcount(pos.pieceBitboards_[ColoredPiece::BLACK_ROOK]);
  pieceCounts[9] = std::popcount(pos.pieceBitboards_[ColoredPiece::BLACK_QUEEN]);
  pieceCountWriter.write_row(pieceCounts);

  float time = float(features[EF::TIME]) / 18.0;
  timeWriter.write_row(&time);
}

std::string get_shard_name(size_t n) {
  // return string with 0 padding
  return std::string(5 - std::to_string(n).length(), '0') + std::to_string(n);
}

int main(int argc, char *argv[]) {
  initialize_geometry();
  initialize_zorbrist();
  initialize_movegen();

  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " <input> <output>" << std::endl;
    return 1;
  }

  const std::string inpath = argv[1];
  const std::string outpath = argv[2];

  std::ifstream infile(inpath);
  #if NNUE
  WriterB tableWriter(outpath + "-table", { NnueFeatures::NF_NUM_FEATURES });
  #endif
  WriterI8 featureWriter(outpath + "-features", { EF::NUM_EVAL_FEATURES });
  WriterI16 evalWriter(outpath + "-eval", { 1 });
  WriterI8 turnWriter(outpath + "-turn", { 1 });
  WriterF32 timeWriter(outpath + "-time", { 1 });
  WriterI8 pieceCountWriter(outpath + "-piece-counts", { 10 });
  #if PST
  WriterI8 pstWriter(outpath + "-pst", { 6 * kNumSquares });
  #endif

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

    process(parts
    #if NNUE
    , tableWriter
    #endif
    , featureWriter
    , evalWriter
    , turnWriter
    , timeWriter
    , pieceCountWriter
    #if PST
    , pstWriter
    #endif
    );

    if ((++counter) % 100'000 == 0) {
      double ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - startTime).count();
      std::cout << "Finished " << counter / 1000 << "k in " << ms / 1000 << " seconds" << std::endl;
    }
  }

  return 0;
}
