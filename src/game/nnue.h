#ifndef CHESS_ENGINE_NNUE_H
#define CHESS_ENGINE_NNUE_H

#include <eigen3/Eigen/Dense>

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
  NF_IS_BLACK_TURN = 769,

  NF_WHITE_KING_CASTLING = 770,
  NF_WHITE_QUEEN_CASTLING = 771,
  NF_BLACK_KING_CASTLING = 772,
  NF_BLACK_QUEEN_CASTLING = 773,

  NF_NUM_WHITE_PAWNS = 774,
  NF_NUM_WHITE_KNIGHTS = 775,
  NF_NUM_WHITE_BISHOPS = 776,
  NF_NUM_WHITE_ROOKS = 777,
  NF_NUM_WHITE_QUEENS = 778,

  NF_NUM_BLACK_PAWNS = 779,
  NF_NUM_BLACK_KNIGHTS = 780,
  NF_NUM_BLACK_BISHOPS = 781,
  NF_NUM_BLACK_ROOKS = 782,
  NF_NUM_BLACK_QUEENS = 783,

  NF_NUM_FEATURES = 784,
};

struct NnueNetwork {
  static constexpr float kFirstLayerDivide = 1.0;
  static constexpr float kSecondLayerDivide = 1.0;
  static constexpr int kWidth1 = 64;

  Matrix<float, 1, Eigen::Dynamic> x0;  // 1 x 64*12  (768)
  Matrix<float, Eigen::Dynamic, kWidth1> w0;  // 768 x kWidth1
  Matrix<float, 1, kWidth1> b0;

  Matrix<float, 1, kWidth1> x1;
  Matrix<float, 1, kWidth1> x1_relu;
  Matrix<float, kWidth1, 64> w1;
  Matrix<float, 1, 64> b1;

  Matrix<float, 1, 64> x2;
  Matrix<float, 64, 1> w2;
  Matrix<float, 1, 64> b2;

  Matrix<float, 1, 1> x3;

  NnueNetwork() {
    x0 = Matrix<float, 1, Eigen::Dynamic>::Zero(1, NnueFeatures::NF_NUM_FEATURES);
    w0 = Matrix<float, Eigen::Dynamic, kWidth1>::Zero(NnueFeatures::NF_NUM_FEATURES, kWidth1);
    this->load("nnue.bin");
  }

  float slowforward() {
    x1.noalias() = x0 * w0;
    x1.noalias() += b0;

    x1_relu = x1.unaryExpr([](float x) -> float {
      return x > 0 ? x / kFirstLayerDivide : 0;
    });
    return this->fastforward();
  }
  float fastforward() {
    x2.noalias() = x1_relu * w1;
    x2.noalias() += b1;

    x2.noalias() = x2.unaryExpr([](float x) -> float {
      return x > 0 ? x / kSecondLayerDivide : 0;
    });

    x3.noalias() = x2 * w2;

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

  void update_x1() {
    this->x1.noalias() = x0 * w0;
    this->x1.noalias() += b0;
    this->x1_relu = x1.unaryExpr([](float x) -> float {
      return x > 0 ? x / kFirstLayerDivide : 0;
    });

  }

  void set_piece(ColoredPiece piece, Square square, float newValue) {
    assert(piece != ColoredPiece::NO_COLORED_PIECE);
    assert(square >= 0);
    assert(square < Square::NO_SQUARE);
    size_t index = (piece - 1) * 64 + square;
    this->set_index(index, newValue);
  }

  void set_index(size_t index, float newValue) {
    assert(newValue == 1 || newValue == 0);
    float delta = newValue - x0(0, index);
    if (delta == 0) {
      return;
    }
    x0(0, index) += delta;
    if (delta == 1) {
      x1.row(0).noalias() += w0.row(index);
    } else if (delta == -1) {
      x1.row(0).noalias() -= w0.row(index);
    } else {
      x1.row(0).noalias() += w0.row(index) * delta;
    }
    x1_relu.row(0) = x1.row(0).unaryExpr([](float x) -> float {
      return x > 0 ? x / kFirstLayerDivide : 0;
    });
  }
};

}  // namespace ChessEngine

#endif // CHESS_ENGINE_NNUE_H