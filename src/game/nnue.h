#ifndef CHESS_ENGINE_NNUE_H
#define CHESS_ENGINE_NNUE_H

#include <eigen3/Eigen/Dense>

#include "utils.h"
#include "geometry.h"

using Eigen::MatrixXd;
using Eigen::Matrix;

namespace ChessEngine {

typedef float MatType;

static MatType kZero = MatType(0.0);

enum NnueFeatures {
  NF_WHITE_PAWN_A8 = 0,
  NF_WHITE_PAWN_B8 = 1,
  // ...
  NF_BLACK_KING_G1 = 766,
  NF_BLACK_KING_H1 = 767,

  NF_IS_WHITE_TURN = 768,
  NF_WHITE_KING_CASTLING = 769,
  NF_WHITE_QUEEN_CASTLING = 770,
  NF_BLACK_KING_CASTLING = 771,
  NF_BLACK_QUEEN_CASTLING = 772,

  NF_EMPTY_1 = 773,
  NF_EMPTY_2 = 774,
  NF_EMPTY_3 = 775,

  NF_NUM_FEATURES = 776,
};

struct NnueNetworkInterface {
  virtual void empty() {}
  virtual float slowforward() { return 0.0; }
  virtual float fastforward() { return 0.0; }
  virtual void load(std::string filename) {}
  virtual void set_piece(ColoredPiece piece, Square square, float newValue) {}
  virtual void set_index(size_t index, float newValue) {}
};

struct DummyNetwork : public NnueNetworkInterface {
  int16_t x[NnueFeatures::NF_NUM_FEATURES];

  DummyNetwork() {}

  void empty() {
    std::fill_n(x, NnueFeatures::NF_NUM_FEATURES, 0);
  }

  float slowforward() {
    return 0.0;
  }
  float fastforward() {
    return 0.0;
  }

  void load(std::string filename) {}

  void set_piece(ColoredPiece piece, Square square, float newValue) {
    int y = square / 8;
    int x = square % 8;
    size_t index = (piece - 1) * 64 + y * 8 + x;
    this->set_index(index, newValue);
  }

  void set_index(size_t index, float newValue) {
    x[index] = newValue;
  }
};

struct NnueNetwork : public NnueNetworkInterface {
  static constexpr int kInputDim = 12 * 8 * 8 + 8;
  static constexpr int kWidth1 = 32;
  static constexpr int kWidth2 = 8;
  static constexpr int kWidth3 = 1;

  Matrix<MatType, 1, Eigen::Dynamic> x0;  // 1 x 64*12  (768)
  Matrix<MatType, Eigen::Dynamic, kWidth1> w0;  // kInputDim x kWidth1
  Matrix<MatType, 1, kWidth1> b0;

  Matrix<MatType, 1, kWidth1> x1;
  Matrix<MatType, 1, kWidth1> x1_relu;
  Matrix<MatType, Eigen::Dynamic, kWidth2> w1;  // kWidth1 x kWidth2
  Matrix<MatType, 1, kWidth2> b1;

  Matrix<MatType, 1, kWidth2> x2;
  Matrix<MatType, Eigen::Dynamic, kWidth3> w2;  // kWidth2 x kWidth3
  Matrix<MatType, 1, kWidth3> b2;

  Matrix<MatType, 1, kWidth3> x3;

  // info depth 1 multipv 1 score cp 1419 nodes 0 nps 0 time 4 pv d2d4
  // info depth 2 multipv 1 score cp 1289 nodes 253 nps 14882 time 17 pv e2e4 e7e5
  // info depth 3 multipv 1 score cp 920 nodes 610 nps 14878 time 41 pv e2e4 e7e5 d2d4 e5d4
  // info depth 4 multipv 1 score cp 1587 nodes 2429 nps 19277 time 126 pv e2e4 c7c5 g1f3 d7d6
  // info depth 5 multipv 1 score cp 1351 nodes 6552 nps 20284 time 323 pv e2e4 e7e5 g1f3 b8c6 f1b5
  // info depth 6 multipv 1 score cp 1418 nodes 35967 nps 21244 time 1693 pv e2e4 e7e6 g1f3 d7d5 b1c3 d5e4 c3e4
  // bestmove e2e4 ponder e7e6

  NnueNetwork() {
    w0 = Matrix<MatType, Eigen::Dynamic, kWidth1>::Zero(NnueFeatures::NF_NUM_FEATURES, kWidth1);
    w1 = Matrix<MatType, Eigen::Dynamic, kWidth2>::Zero(kWidth1, kWidth2);
    w2 = Matrix<MatType, Eigen::Dynamic, kWidth3>::Zero(kWidth2, kWidth3);

    x0 = Matrix<MatType, 1, Eigen::Dynamic>::Zero(1, NnueFeatures::NF_NUM_FEATURES);
    x1 = Matrix<MatType, 1, kWidth1>::Zero(1, kWidth1);
    x1_relu = Matrix<MatType, 1, kWidth1>::Zero(1, kWidth1);
    x2 = Matrix<MatType, 1, kWidth2>::Zero(1, kWidth2);
    x3 = Matrix<MatType, 1, kWidth3>::Zero(1, kWidth3);

    b0 = Matrix<MatType, 1, kWidth1>::Zero(1, kWidth1);
    b1 = Matrix<MatType, 1, kWidth2>::Zero(1, kWidth2);
    b2 = Matrix<MatType, 1, kWidth3>::Zero(1, kWidth3);

    std::ifstream myfile;
    myfile.open("nnue-776-1024-128.bin", std::ios::in | std::ios::binary);
    if (!myfile.is_open()) {
      std::cout << "Error opening file \"nnue-776-512-64.bin\"" << std::endl;
      exit(0);
    }
    this->load(myfile);
    myfile.close();
  }

  void empty() {
    x0.setZero();
    x1.setZero();
    x1.noalias() += b0;
  }

  float slowforward() {
    x1.noalias() = x0 * w0;
    x1.noalias() += b0;
    return this->fastforward();
  }
  float fastforward() {
    x1_relu.noalias() = x1.unaryExpr([](float x) -> float {
      return x > 0 ? x : x * 0.01;
    });

    x2.noalias() = x1_relu * w1;
    x2.noalias() += b1;
    x2.noalias() = x2.unaryExpr([](float x) -> float {
      return x > 0 ? x : x * 0.01;
    });
    x3.noalias() = x2 * w2;
    x3.noalias() += b2;
    return x3(0, 0);
  }

  template<int H, int W>
  void _load(std::istream& file, Eigen::Matrix<MatType, H, W>& out) {
    const size_t N = out.size();
    for (size_t i = 0; i < N; ++i) {
      float value;
      file.read(reinterpret_cast<char*>(&value), sizeof(float));
      reinterpret_cast<MatType*>(out.data())[i] = MatType(value);
    }
  }

  void load(std::istream& myfile) {
    // myfile.read(reinterpret_cast<char*>(w0.data()), w0.size() * sizeof(float));
    // myfile.read(reinterpret_cast<char*>(w1.data()), w1.size() * sizeof(float));
    // myfile.read(reinterpret_cast<char*>(w2.data()), w2.size() * sizeof(float));
    // myfile.read(reinterpret_cast<char*>(b0.data()), b0.size() * sizeof(float));
    // myfile.read(reinterpret_cast<char*>(b1.data()), b1.size() * sizeof(float));
    _load(myfile, w0);
    _load(myfile, w1);
    _load(myfile, w2);
    _load(myfile, b0);
    _load(myfile, b1);
  }

  void set_piece(ColoredPiece piece, Square square, float newValue) {
    assert(piece != ColoredPiece::NO_COLORED_PIECE);
    assert(square >= 0);
    assert(square < Square::NO_SQUARE);
    int y = square / 8;
    int x = square % 8;
    size_t index = (piece - 1) * 64 + y * 8 + x;
    this->set_index(index, newValue);
  }

  void set_index(size_t index, float newValue) {
    MatType delta = MatType(newValue) - x0(0, index);
    if (delta == 0.0) {
      return;
    }
    x0.data()[index] += delta;
    if (delta == 1.0) {
      x1.row(0).noalias() += w0.row(index);
    } else if (delta == -1.0) {
      x1.row(0).noalias() -= w0.row(index);
    } else {
      x1.row(0).noalias() += w0.row(index) * MatType(delta);
    }
  }
};

}  // namespace ChessEngine

#endif // CHESS_ENGINE_NNUE_H