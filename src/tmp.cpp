#include <eigen3/Eigen/Core>
// #include <eigen3/EigenRand/EigenRand>

#include <iostream>
#include <memory>
#include <random>

#include "game/geometry.h"
#include "game/utils.h"
#include "game/Position.h"
#include "game/movegen.h"
#include "game/movegen/sliding.h"
#include "game/Evaluator.h"
#include "game/search.h"

using namespace Eigen;

struct Node {
  Node(MatrixXd value) : value(value) {
    grad = MatrixXd::Zero(value.rows(), value.cols());
  }
  MatrixXd value;
  MatrixXd grad;

  virtual void backward() = 0;
};

struct Multiply : public Node {
  Multiply(std::shared_ptr<Node> a, std::shared_ptr<Node> b) : Node(a->value * b->value), a(a), b(b) {}
  std::shared_ptr<Node> a;
  std::shared_ptr<Node> b;

  void backward() {
    a->grad += grad * b->value.transpose();
    b->grad += a->value.transpose() * grad;
  }
};

struct Add : public Node {
  Add(std::shared_ptr<Node> a, std::shared_ptr<Node> b) : Node(a->value + b->value), a(a), b(b) {}
  std::shared_ptr<Node> a;
  std::shared_ptr<Node> b;

  void backward() {
    a->grad += grad;
    b->grad += grad;
  }
};

struct Variable : public Node {
  Variable(MatrixXd value) : Node(value) {}

  void backward() {}
};

struct ReLU : public Node {
  ReLU(std::shared_ptr<Node> a) : Node(a->value.cwiseMax(0)), a(a) {}
  std::shared_ptr<Node> a;

  void backward() {
    a->grad += grad.cwiseMax(0);
  }
};

struct Sigmoid : public Node {
  Sigmoid(std::shared_ptr<Node> a) : Node(1 / (1 + (-a->value).array().exp())), a(a) {}
  std::shared_ptr<Node> a;

  void backward() {
    a->grad += grad * (value * (MatrixXd::Ones(value.rows(), value.cols()) - value));
  }
};

MatrixXd concat_rows(const std::vector<std::shared_ptr<Node>>& inputs) {
  int rows = 0;
  for (auto& input : inputs) {
    rows += input->value.rows();
  }
  MatrixXd res(rows, inputs[0]->value.cols());
  int offset = 0;
  for (auto& input : inputs) {
    res.block(offset, 0, input->value.rows(), input->value.cols()) = input->value;
    offset += input->value.rows();
  }
  return res;
}

struct ConcatRows : public Node {
  ConcatRows(std::vector<std::shared_ptr<Node>> inputs) : Node(concat_rows(inputs)), inputs(inputs) {}
  std::vector<std::shared_ptr<Node>> inputs;

  void backward() {
    int offset = 0;
    for (auto& input : inputs) {
      input->grad += grad.block(offset, 0, input->value.rows(), input->value.cols());
      offset += input->value.rows();
    }
  }
};

MatrixXd position_to_matrix(const ChessEngine::Position& pos) {
  MatrixXd res(1, 8 * 8 * ChessEngine::ColoredPiece::NUM_COLORED_PIECES);
  for (int i = 0; i < ChessEngine::kNumSquares; ++i) {
    for (int j = 0; j < ChessEngine::ColoredPiece::NUM_COLORED_PIECES; j++) {
      res(0, i * ChessEngine::ColoredPiece::NUM_COLORED_PIECES + j) = (pos.tiles_[i] == j);
    }
  }
  return res;
}

MatrixXd randn(int rows, int cols) {
  std::default_random_engine generator;
  std::normal_distribution<double> distribution(0.0, 1.0);
  MatrixXd res(rows, cols);
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      res(i, j) = distribution(generator);
    }
  }
  return res;
}

struct Model {
  Model() {
    w0 = std::make_shared<Variable>(randn(8 * 8 * ChessEngine::ColoredPiece::NUM_COLORED_PIECES, 32));
    b0 = std::make_shared<Variable>(randn(32, 1));
  }
  std::shared_ptr<Node> operator()(std::shared_ptr<Node> x) {
    return std::make_shared<ReLU>(std::make_shared<Add>(std::make_shared<Multiply>(w0, x), b0));
  }
  std::shared_ptr<Node> w0;
  std::shared_ptr<Node> b0;
};

int main() {
  // std::shared_ptr<Node> a = std::make_shared<Variable>(MatrixXd::Identity(3, 3));
  // std::shared_ptr<Node> b = std::make_shared<Variable>(MatrixXd::Ones(3, 3));

  // std::shared_ptr<Node> c = std::make_shared<Multiply>(a, b);

  // std::cout << c->value << std::endl;

  // c->grad = MatrixXd::Ones(3, 3);
  // c->backward();

  // std::cout << a->grad << std::endl;
  // std::cout << b->grad << std::endl;

  ChessEngine::Position pos = ChessEngine::Position::init();

  std::shared_ptr<Variable> input = std::make_shared<Variable>(position_to_matrix(pos));

  std::shared_ptr<Node> x = std::make_shared<ConcatRows>(std::vector<std::shared_ptr<Node>>({
    input,
    input,
    input,
  }));


  Model model;

  auto output = model(x);

  return 0;
}

