// sqlite3 db.sqlite3 "select * from make_train2_d6_n0 limit 5;" > training_data.txt

#include <cassert>
#include <cstdint>
#include <ctime>
#include <cstdlib>

#include <string>
#include <vector>
#include <iostream>
#include <algorithm>
#include <random>
#include <fstream>
#include <deque>

#include "game/geometry.h"
#include "game/utils.h"
#include "game/Position.h"
#include "game/movegen.h"
#include "game/movegen/sliding.h"
#include "game/Evaluator.h"
#include "game/search.h"

using namespace ChessEngine;

std::chrono::time_point<std::chrono::steady_clock> current_time() {
  return std::chrono::steady_clock::now();
}

int64_t elapsed_ms(std::chrono::time_point<std::chrono::steady_clock> start, std::chrono::time_point<std::chrono::steady_clock> end) {
  std::chrono::duration<double> delta = end - start;
  return std::chrono::duration_cast<std::chrono::milliseconds>(delta).count();
}

struct Result {
  std::vector<std::pair<Evaluation, std::string>> pvs;
};

template<Color TURN>
Evaluation simple_qsearch(
  Position *position,
  Evaluator *evaluator,
  const Depth depth,
  Evaluation alpha, 
  Evaluation beta) {
  
  if (std::popcount(position->pieceBitboards_[coloredPiece<TURN, Piece::KING>()]) == 0) {
    return alpha;
  }

  const bool lookAtChecksToo = depth < 2;

  constexpr Color opposingColor = opposite_color<TURN>();
  constexpr ColoredPiece moverKing = coloredPiece<TURN, Piece::KING>();
  const bool inCheck = can_enemy_attack<TURN>(*position, lsb(position->pieceBitboards_[moverKing]));

  ExtMove moves[kMaxNumMoves];
  ExtMove *end;
  if (lookAtChecksToo) {
    end = compute_moves<TURN, MoveGenType::CHECKS_AND_CAPTURES>(*position, moves);
  } else {
    end = compute_moves<TURN, MoveGenType::CAPTURES>(*position, moves);
  }

  if (moves == end && inCheck) {
    return std::max<Evaluation>(kQCheckmate + depth, alpha);
  }

  for (ExtMove *move = moves; move < end; ++move) {
    move->score = kMoveOrderPieceValues[move->capture];
    move->score -= kMoveOrderPieceValues[move->piece];
    move->score += (move->capture != Piece::NO_PIECE) * 1000;
  }
  std::sort(moves, end, [](ExtMove a, ExtMove b) {
    return a.score > b.score;
  });

  Evaluation e = evaluator->score<TURN>(*position);
  if (e >= beta) {
    return beta;
  }
  alpha = std::max(alpha, e);

  for (ExtMove *move = moves; move < end; ++move) {
    make_move<TURN>(position, move->move);
    Evaluation child = -simple_qsearch<opposingColor>(position, evaluator, depth + 1, -beta, -alpha);
    undo<TURN>(position);

    child -= (child > -kQLongestForcedMate);
    child += (child <  kQLongestForcedMate);

    if (child > alpha) {
      alpha = child;
      if (alpha >= beta) {
        return beta;
      }
    }
  }


  return alpha;
}

template<Color TURN>
static void simple_search(
  Position *position,
  Evaluator *evaluator,
  Result *result,
  int multiPV) {

  constexpr Color opposingColor = opposite_color<TURN>();
  constexpr ColoredPiece moverKing = coloredPiece<TURN, Piece::KING>();

  ExtMove moves[kMaxNumMoves];
  ExtMove *movesEnd = compute_legal_moves<TURN>(position, moves);

  for (ExtMove *move = moves; move < movesEnd; ++move) {
    move->score = kMoveOrderPieceValues[move->capture];
    move->score -= kMoveOrderPieceValues[move->piece];
    move->score += (move->capture != Piece::NO_PIECE) * 1000;
  }
  std::sort(moves, movesEnd, [](ExtMove a, ExtMove b) {
    return a.score > b.score;
  });

  const Evaluation beta = kMaxEval;
  Evaluation alpha = kMinEval;

  result->pvs.clear();

  for (ExtMove *move = moves; move < movesEnd; ++move) {
    make_move<TURN>(position, move->move);
    Evaluation a = -simple_qsearch<opposingColor>(position, evaluator, 0, -beta, -alpha);
    undo<TURN>(position);

    result->pvs.push_back(std::make_pair(a, move->move.uci()));
    if (result->pvs.size() > multiPV) {
      std::sort(result->pvs.begin(), result->pvs.end(), [](std::pair<Evaluation, std::string> a, std::pair<Evaluation, std::string> b) {
        return a.first > b.first;
      });
      result->pvs.pop_back();
    }
    if (result->pvs.size() >= multiPV) {
      alpha = result->pvs.back().first;
    }
  }
}

struct Datapoint {
  std::string fen;
  std::vector<std::string> ucis;
  std::vector<Evaluation> evals;
};

int main(int argc, char *argv[]) {
  initialize_geometry();
  initialize_zorbrist();
  initialize_movegen();

  std::deque<std::string> args;
  for (size_t i = 1; i < argc; ++i) {
    args.push_back(argv[i]);
  }

  if (args.size() != 3) {
    throw std::runtime_error("Usage: opt <train.txt> <weights.txt> <SampleSize>");
  }
  const int sampleSize = std::stoi(args[2]);

  std::ifstream myfile;
  myfile.open(args[0]);
  if (!myfile.is_open()) {
    std::cout << "Failed to open file" << std::endl;
    return 1;
  }

  std::vector<Datapoint> datapoints;

  std::string line;
  getline(myfile, line);
  while (line.size() > 0) {
    std::vector<std::string> parts = split(line, '|');

    datapoints.push_back(Datapoint{});
    Datapoint& datapoint = datapoints.back();
    datapoint.fen = parts[0];

    for (size_t i = 1; i < parts.size(); i += 3) {
      datapoint.ucis.push_back(parts[i + 0]);
      datapoint.evals.push_back(std::stoi(parts[i + 1]));
      // parts[i + 2] is "isCapture"
    }

    if (datapoints.size() >= sampleSize) {
      break;
    }

    getline(myfile, line);
  }

  Evaluator evaluator;
  PieceMaps pieceMaps;
  {
    std::ifstream f;
    f.open(args[1]);
    if (!f.is_open()) {
      std::cout << "Failed to open file" << std::endl;
      return 1;
    }
    evaluator.load_weights_from_file(f);
    pieceMaps.load_weights_from_file(f);
  }

  const int multiPV = 1;

  double loss = 0.0;
  double loss2 = 0.0;
  for (const Datapoint& datapoint : datapoints) {
    Position pos(datapoint.fen);
    pos.set_piece_maps(pieceMaps);
    
    Result result;
    if (pos.turn_ == Color::WHITE) {
      simple_search<WHITE>(&pos, &evaluator, &result, multiPV);
    } else {
      simple_search<BLACK>(&pos, &evaluator, &result, multiPV);
    }
    std::vector<std::pair<Evaluation, std::string>> pvs = result.pvs;

    double score = datapoint.evals.back();  // Default
    for (size_t i = 0; i < datapoint.evals.size(); ++i) {
      if (datapoint.ucis[i] == pvs[0].second) {
        score = datapoint.evals[i];
        break;
      }
    }

    double delta = std::min(datapoint.evals[0] - score, 50.0);
    if (delta < 0.0) {
      throw std::runtime_error("lalala");
    }

    loss += delta;
    loss2 += delta * delta;
  }

  double mean = loss / datapoints.size();
  double variance = (loss2 / datapoints.size() - mean * mean) / (datapoints.size() - 1);

  std::cout << mean << " ± " << std::sqrt(variance) << std::endl;

  return 0;
}
