#ifndef CHESS_ENGINE_NNUE_H
#define CHESS_ENGINE_NNUE_H

#include "utils.h"
#include "geometry.h"

using Eigen::MatrixXd;
using Eigen::Matrix;

namespace ChessEngine {

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
  static constexpr int kWidth1 = 512;
  static constexpr int kWidth2 = 64;
  static constexpr int kWidth3 = 1;

  Matrix<float, 1, Eigen::Dynamic> x0;  // 1 x 64*12  (768)
  Matrix<float, Eigen::Dynamic, kWidth1> w0;  // kInputDim x kWidth1
  Matrix<float, 1, kWidth1> b0;

  Matrix<float, 1, kWidth1> x1;
  Matrix<float, 1, kWidth1> x1_relu;
  Matrix<float, Eigen::Dynamic, kWidth2> w1;  // kWidth1 x kWidth2
  Matrix<float, 1, kWidth2> b1;

  Matrix<float, 1, kWidth2> x2;
  Matrix<float, Eigen::Dynamic, kWidth3> w2;  // kWidth2 x kWidth3
  Matrix<float, 1, kWidth3> b2;

  Matrix<float, 1, kWidth3> x3;

  NnueNetwork() {
    w0 = Matrix<float, Eigen::Dynamic, kWidth1>::Zero(NnueFeatures::NF_NUM_FEATURES, kWidth1);
    w1 = Matrix<float, Eigen::Dynamic, kWidth2>::Zero(kWidth1, kWidth2);
    w2 = Matrix<float, Eigen::Dynamic, kWidth3>::Zero(kWidth2, kWidth3);

    x0 = Matrix<float, 1, Eigen::Dynamic>::Zero(1, NnueFeatures::NF_NUM_FEATURES);
    x1 = Matrix<float, 1, kWidth1>::Zero(1, kWidth1);
    x1_relu = Matrix<float, 1, kWidth1>::Zero(1, kWidth1);
    x2 = Matrix<float, 1, kWidth2>::Zero(1, kWidth2);
    x3 = Matrix<float, 1, kWidth3>::Zero(1, kWidth3);

    b0 = Matrix<float, 1, kWidth1>::Zero(1, kWidth1);
    b1 = Matrix<float, 1, kWidth2>::Zero(1, kWidth2);
    b2 = Matrix<float, 1, kWidth3>::Zero(1, kWidth3);

    this->load("nnue.bin");
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
      return x > 0 ? x : 0;
    });

    x2.noalias() = x1_relu * w1;
    x2.noalias() += b1;
    x2.noalias() = x2.unaryExpr([](float x) -> float {
      return x > 0 ? x : 0;
    });
    x3.noalias() = x2 * w2;
    x3.noalias() += b2;
    return x3(0, 0);
  }

  void load(std::string filename) {
    FILE *f = fopen(filename.c_str(), "rb");
    fread(w0.data(), sizeof(float), w0.size(), f);
    fread(w1.data(), sizeof(float), w1.size(), f);
    fread(w2.data(), sizeof(float), w2.size(), f);
    fread(b0.data(), sizeof(float), b0.size(), f);
    fread(b1.data(), sizeof(float), b1.size(), f);
    fclose(f);
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
    float delta = newValue - x0(0, index);
    if (delta == 0.0) {
      return;
    }
    x0.data()[index] += delta;
    if (delta == 1.0) {
      x1.row(0).noalias() += w0.row(index);
    } else if (delta == -1.0) {
      x1.row(0).noalias() -= w0.row(index);
    } else {
      x1.row(0).noalias() += w0.row(index) * delta;
    }
  }
};

}  // namespace ChessEngine

#endif // CHESS_ENGINE_NNUE_H