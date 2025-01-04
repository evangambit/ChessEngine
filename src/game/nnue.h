#ifndef CHESS_ENGINE_NNUE_H
#define CHESS_ENGINE_NNUE_H

#include "utils.h"
#include "geometry.h"

#include <algorithm>

namespace {
  typedef int16_t VecType;
  typedef int16_t MatType;

  constexpr int32_t kScale = 256;

  // Intepreted as <N, 1> matrix.
  template<size_t N>
  struct Vector {
    Vector() : _data(new VecType[N]) {
      std::fill_n(_data, N, VecType(0));
    }
    ~Vector() {
      delete[] _data;
    }
    Vector(const Vector& other) : _data(new VecType[N]) {
      std::copy_n(other._data, N, _data);
    }
    Vector& operator=(const Vector& other) {
      std::copy_n(other._data, N, _data);
      return *this;
    }
    void zero_() {
      std::fill_n(_data, N, 0.0);
    }
    Vector& operator+=(const Vector& other) {
      for (size_t i = 0; i < N; ++i) {
        _data[i] += other._data[i];
      }
      return *this;
    }

    inline VecType operator()(size_t i) const {
      return _data[i];
    }
    inline VecType& operator()(size_t i) {
      return _data[i];
    }
    size_t size() const {
      return N;
    }

    void read(std::istream& myfile) {
      float *data = new float[N];
      myfile.read(reinterpret_cast<char*>(data), this->size() * sizeof(float));
      for (size_t i = 0; i < N; ++i) {
        _data[i] = data[i] * kScale;
      }
    }

    void clipped_relu(Vector<N>& out) {
      for (size_t i = 0; i < N; ++i) {
        VecType x = (*this)(i);
        out(i) = std::max(VecType(0), std::min(VecType(kScale), x));
      }
    }

    VecType *_data;
  };

  template<size_t ROWS, size_t COLS>
  struct Matrix {
    Matrix() : _data(new MatType[ROWS * COLS]) {
      std::fill_n(_data, ROWS * COLS, MatType(0));
    }
    ~Matrix() {
      delete[] _data;
    }
    Matrix(const Matrix& other) : _data(new MatType[ROWS * COLS]) {
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

    void affine(const Vector<COLS>& in, const Vector<ROWS>& bias, Vector<ROWS>& out) {
      for (size_t i = 0; i < ROWS; ++i) {
        int32_t sum = 0;
        for (size_t j = 0; j < COLS; ++j) {
          sum += (*this)(i, j) * in(j);
        }
        out(i) = sum / kScale + bias(i);
      }
    }

    inline MatType operator()(size_t i, size_t j) const {
      return _data[i * COLS + j];
    }
    inline MatType& operator()(size_t i, size_t j) {
      return _data[i * COLS + j];
    }

    size_t size() const {
      return ROWS * COLS;
    }

    void read(std::istream& myfile) {
      float *data = new float[ROWS * COLS];
      myfile.read(reinterpret_cast<char*>(data), this->size() * sizeof(float));
      for (size_t i = 0; i < ROWS * COLS; ++i) {
        _data[i] = data[i] * kScale;
      }
    }

    MatType *_data;
  };

  template<size_t ROWS, size_t COLS>
  void incremental_update(Vector<ROWS>& mutable_vector, const Matrix<ROWS, COLS>& weights_matrix, size_t col, float scale) {
    for (size_t i = 0; i < ROWS; ++i) {
      mutable_vector(i) += weights_matrix(i, col) * scale;
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
  int32_t x[NnueFeatures::NF_NUM_FEATURES];

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

  Vector<kInputDim> x0;
  Matrix<kWidth1, kInputDim> w0;
  Vector<kWidth1> b0;

  Vector<kWidth1> x1;
  Vector<kWidth1> x1_relu;
  Matrix<kWidth2, kWidth1> w1;
  Vector<kWidth2> b1;

  Vector<kWidth2> x2;
  Vector<kWidth2> x2_relu;
  Matrix<kWidth3, kWidth2> w2;
  Vector<kWidth3> b2;

  Vector<kWidth3> x3;

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
    return this->fastforward();
  }
  float fastforward() {
    x1.clipped_relu(x1_relu);
    w1.affine(x1_relu, b1, x2);
    x2.clipped_relu(x2_relu);
    w2.affine(x2_relu, b2, x3);
    return x3(0);
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
    w0.read(myfile);
    w1.read(myfile);
    w2.read(myfile);
    b0.read(myfile);
    b1.read(myfile);
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
    MatType delta = MatType(newValue) - x0(index);
    if (delta == 0.0) {
      return;
    }
    x0(index) += delta;
    incremental_update(x1, w0, index, delta);
  }
};

}  // namespace ChessEngine

#endif // CHESS_ENGINE_NNUE_H