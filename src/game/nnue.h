#ifndef CHESS_ENGINE_NNUE_H
#define CHESS_ENGINE_NNUE_H

// #include <eigen3/Eigen/Dense>

#include "utils.h"
#include "geometry.h"

#include <algorithm>

namespace {
  float leaky_relu(float x) {
    return x > 0.0 ? x : 0.01 * x;
  }
  template<size_t ROWS, size_t COLS>
  struct Matrix {
    Matrix() : _data(new float[ROWS * COLS]) {
      std::fill_n(_data, ROWS * COLS, 0.0);
    }
    ~Matrix() {
      delete[] _data;
    }
    Matrix(const Matrix& other) : _data(new float[ROWS * COLS]) {
      std::copy_n(other._data, ROWS * COLS, _data);
    }
    Matrix& operator=(const Matrix& other) {
      std::copy_n(other._data, ROWS * COLS, _data);
      return *this;
    }
    void zero_() {
      std::fill_n(_data, ROWS * COLS, 0.0);
    }
    Matrix& operator+=(const Matrix& other) {
      for (size_t i = 0; i < ROWS * COLS; ++i) {
        _data[i] += other._data[i];
      }
      return *this;
    }

    void affine(const Matrix<COLS, 1>& in, const Matrix<ROWS, 1>& bias, Matrix<ROWS, 1>& out) {
      for (size_t i = 0; i < ROWS; ++i) {
        out(i, 0) = bias(i, 0);
        for (size_t j = 0; j < COLS; ++j) {
          out(i, 0) += (*this)(i, j) * in(j, 0);
        }
      }
    }

    void leaky_relu_then_affine(const Matrix<COLS, 1>& in, const Matrix<ROWS, 1>& bias, Matrix<ROWS, 1>& out) {
      for (size_t i = 0; i < ROWS; ++i) {
        out(i, 0) = bias(i, 0);
        for (size_t j = 0; j < COLS; ++j) {
          out(i, 0) += (*this)(i, j) * leaky_relu(in(j, 0));
        }
      }
    }

    inline float operator()(size_t i, size_t j) const {
      return _data[i * COLS + j];
    }
    inline float& operator()(size_t i, size_t j) {
      return _data[i * COLS + j];
    }

    float *data() {
      return _data;
    }

    size_t size() const {
      return ROWS * COLS;
    }

    float *_data;
  };

  template<size_t ROWS, size_t COLS>
  void incremental_update(Matrix<ROWS, 1>& mutable_matrix, const Matrix<ROWS, COLS>& weights_matrix, size_t col, float scale) {
    for (size_t i = 0; i < ROWS; ++i) {
      mutable_matrix(i, 0) += weights_matrix(i, col) * scale;
    }
  }
}

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
  virtual void set_piece(ColoredPiece piece, SafeSquare square, float newValue) {}
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

  void set_piece(ColoredPiece piece, SafeSquare square, float newValue) {
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
  static constexpr int kWidth1 = 48;
  static constexpr int kWidth2 = 16;
  static constexpr int kWidth3 = 1;

  Matrix<kInputDim, 1> x0;  // 1 x 64*12  (768)
  Matrix<kWidth1, kInputDim> w0;  // kInputDim x kWidth1
  Matrix<kWidth1, 1> b0;

  Matrix<kWidth1, 1> x1;
  Matrix<kWidth1, 1> x1_relu;
  Matrix<kWidth2, kWidth1> w1;  // kWidth1 x kWidth2
  Matrix<kWidth2, 1> b1;

  Matrix<kWidth2, 1> x2;
  Matrix<kWidth3, kWidth2> w2;  // kWidth2 x kWidth3
  Matrix<kWidth3, 1> b2;

  Matrix<kWidth3, 1> x3;

  NnueNetwork() {
    // this->load("nnue-776-32-8.bin");
  }

  void empty() {
    x0.zero_();
    x1.zero_();
    x1 += b0;
  }

  float slowforward() {
    w0.affine(x0, b0, x1);
    // x1.noalias() = x0 * w0;
    // x1.noalias() += b0;
    return this->fastforward();
  }
  float fastforward() {
    w1.leaky_relu_then_affine(x1, b1, x2);
    w2.leaky_relu_then_affine(x2, b2, x3);
    return x3(0, 0);
  }

  void load(std::string filename) {
    std::ifstream myfile;
    myfile.open(filename, std::ios::in | std::ios::binary);
    if (!myfile.is_open()) {
      std::cout << "Error opening file \"" << filename << "\"" << std::endl;
      exit(0);
    }
    this->load(myfile);
    myfile.close();
  }

  void load(std::istream& myfile) {
    myfile.read(reinterpret_cast<char*>(w0.data()), w0.size() * sizeof(float));
    myfile.read(reinterpret_cast<char*>(w1.data()), w1.size() * sizeof(float));
    myfile.read(reinterpret_cast<char*>(w2.data()), w2.size() * sizeof(float));
    myfile.read(reinterpret_cast<char*>(b0.data()), b0.size() * sizeof(float));
    myfile.read(reinterpret_cast<char*>(b1.data()), b1.size() * sizeof(float));
  }

  void set_piece(ColoredPiece piece, SafeSquare square, float newValue) {
    assert(piece != ColoredPiece::NO_COLORED_PIECE);
    assert_valid_square(square);
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
    incremental_update(x1, w0, index, delta);
    // if (delta == 1.0) {
    //   // x1.row(0).noalias() += w0.row(index);
    //   x1.increment_from_col(w0, index, delta);
    // } else if (delta == -1.0) {
    //   x1.row(0).noalias() -= w0.row(index);
    // } else {
    //   x1.row(0).noalias() += w0.row(index) * MatType(delta);
    // }
  }
};

}  // namespace ChessEngine

#endif // CHESS_ENGINE_NNUE_H