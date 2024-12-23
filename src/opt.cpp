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
std::pair<Evaluation, Move> simple_qsearch(
  Position *position,
  Evaluator *evaluator,
  const Depth depth,
  Evaluation alpha, 
  Evaluation beta) {
  
  if (std::popcount(position->pieceBitboards_[coloredPiece<TURN, Piece::KING>()]) == 0) {
    return std::make_pair(alpha, kNullMove);
  }

  if (depth > 5) {
    return std::make_pair(std::min(beta, std::max(alpha, evaluator->score<TURN>(*position))), kNullMove);
  }

  const bool lookAtAllMoves = (depth == 0);
  const bool lookAtChecks = (depth <= 2) && !lookAtAllMoves;
  constexpr Color opposingColor = opposite_color<TURN>();
  constexpr ColoredPiece moverKing = coloredPiece<TURN, Piece::KING>();
  const bool inCheck = can_enemy_attack<TURN>(*position, safe_lsb(position->pieceBitboards_[moverKing]));

  ExtMove moves[kMaxNumMoves];
  ExtMove *end;
  if (lookAtAllMoves) {
    end = compute_moves<TURN, MoveGenType::ALL_MOVES>(*position, moves);
  } else if (lookAtChecks) {
    end = compute_moves<TURN, MoveGenType::CHECKS_AND_CAPTURES>(*position, moves);
  } else {
    end = compute_moves<TURN, MoveGenType::CAPTURES>(*position, moves);
  }

  if (moves == end && inCheck) {
    return std::make_pair(std::max<Evaluation>(kQCheckmate + depth, alpha), kNullMove);
  }

  for (ExtMove *move = moves; move < end; ++move) {
    move->score = kMoveOrderPieceValues[cp2p(move->capture)];
    move->score -= kMoveOrderPieceValues[move->piece];
    move->score += (move->capture != ColoredPiece::NO_COLORED_PIECE) * 1000;
  }
  std::sort(moves, end, [](ExtMove a, ExtMove b) {
    return a.score > b.score;
  });

  if (!lookAtAllMoves) {
    alpha = std::max(alpha, evaluator->score<TURN>(*position));
    if (alpha >= beta) {
      return std::make_pair(beta, Move{Square::A1, Square::A2});
    }
  }

  Move bestmove = kNullMove;
  for (ExtMove *move = moves; move < end; ++move) {
    make_move<TURN>(position, move->move);
    std::pair<Evaluation, Move> a = simple_qsearch<opposingColor>(position, evaluator, depth + 1, -beta, -alpha);
    Evaluation child = -a.first;
    undo<TURN>(position);

    if (child > alpha) {
      alpha = std::min(beta, child);
      bestmove = move->move;
      if (alpha >= beta) {
        return std::make_pair(beta, bestmove);
      }
    }
  }


  return std::make_pair(alpha, bestmove);
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
    move->score = kMoveOrderPieceValues[cp2p(move->capture)];
    move->score -= kMoveOrderPieceValues[move->piece];
    move->score += (move->capture != ColoredPiece::NO_COLORED_PIECE) * 1000;
  }
  std::sort(moves, movesEnd, [](ExtMove a, ExtMove b) {
    return a.score > b.score;
  });

  const Evaluation beta = kMaxEval;
  Evaluation alpha = kMinEval;

  result->pvs.clear();

  for (ExtMove *move = moves; move < movesEnd; ++move) {
    make_move<TURN>(position, move->move);
    std::pair<Evaluation, Move> a = simple_qsearch<opposingColor>(position, evaluator, 0, -beta, -alpha);
    Evaluation childScore = -a.first;
    undo<TURN>(position);

    result->pvs.push_back(std::make_pair(childScore, move->move.uci()));
    if (result->pvs.size() > multiPV) {
      std::stable_sort(result->pvs.begin(), result->pvs.end(), [](std::pair<Evaluation, std::string> a, std::pair<Evaluation, std::string> b) {
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

std::pair<double, double> process_datapoints(int threadId, const Datapoint *start, const Datapoint *end, std::string weightsFile) {
  Evaluator evaluator;
  PieceMaps pieceMaps;
  {
    std::ifstream f;
    f.open(weightsFile);
    if (!f.is_open()) {
      std::cout << "Failed to open file" << std::endl;
      return std::make_pair(0.0, 0.0);
    }
    evaluator.load_weights_from_file(f);
    pieceMaps.load_weights_from_file(f);
  }

  const int multiPV = 1;

  double loss = 0.0;
  double loss2 = 0.0;
  for (const Datapoint *datapoint = start; datapoint != end; ++datapoint) {
    Position pos(datapoint->fen);
    pos.set_piece_maps(pieceMaps);
    
    Result result;
    if (pos.turn_ == Color::WHITE) {
      simple_search<WHITE>(&pos, &evaluator, &result, multiPV);
    } else {
      simple_search<BLACK>(&pos, &evaluator, &result, multiPV);
    }
    std::vector<std::pair<Evaluation, std::string>> pvs = result.pvs;

    double score = datapoint->evals.back();  // Default
    for (size_t i = 0; i < datapoint->evals.size(); ++i) {
      if (datapoint->ucis[i] == pvs[0].second) {
        score = datapoint->evals[i];
        break;
      }
    }

    // double delta = std::min(datapoint->evals[0] - score, 200.0);
    double delta = std::abs(datapoint->evals[0] - score);

    loss += delta;
    loss2 += delta * delta;
  }

  return std::make_pair(loss, loss2);
}

int main(int argc, char *argv[]) {
  initialize_geometry();
  initialize_zorbrist();
  initialize_movegen();

  // Evaluator evaluator;
  // PieceMaps pieceMaps;
  // {
  //   std::ifstream f;
  //   f.open("weights.txt");
  //   if (!f.is_open()) {
  //     std::cout << "Failed to open file" << std::endl;
  //     return 1;
  //   }
  //   evaluator.load_weights_from_file(f);
  //   pieceMaps.load_weights_from_file(f);
  // }

  // Position pos("1kr2b1r/pp3pp1/1nb1p2p/q2pP3/1N1P4/P1P5/1QBB1PPP/RR4K1 w - - 10 21");
  // Result result;
  // if (pos.turn_ == Color::WHITE) {
  //   simple_search<WHITE>(&pos, &evaluator, &result, 1);
  // } else {
  //   simple_search<BLACK>(&pos, &evaluator, &result, 1);
  // }
  // std::cout << result.pvs[0] << std::endl;

  std::deque<std::string> args;
  for (size_t i = 1; i < argc; ++i) {
    args.push_back(argv[i]);
  }

  if (args.size() != 4) {
    throw std::runtime_error("Usage: opt <train.txt> <weights.txt> <SampleSize> <NumThreads>");
  }
  const int sampleSize = std::stoi(args[2]);
  const int numThreads = std::stoi(args[3]);

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

    for (size_t i = 1; i < parts.size(); i += 2) {
      datapoint.ucis.push_back(parts[i + 0]);
      // string to float
      datapoint.evals.push_back(std::stof(parts[i + 1]) * 1000);
      // parts[i + 2] is "isCapture"
    }

    if (std::abs(datapoint.evals.front() - datapoint.evals.back()) < 2) {
      datapoints.pop_back();
    }

    if (datapoints.size() >= sampleSize) {
      break;
    }

    getline(myfile, line);
  }

  std::vector<std::pair<double, double>> results(numThreads);
  std::vector<std::thread> threads;
  for (int i = 0; i < numThreads; ++i) {
    threads.push_back(std::thread([i, numThreads, &results, &datapoints, &args] {
      const uint64_t n = datapoints.size();
      const uint64_t start = i * n / numThreads;
      const uint64_t end = (i == numThreads - 1 ? n : (i + 1) * n / numThreads);
      results[i] = process_datapoints(i, &datapoints[0] + start, &datapoints[0] + end, args[1]);
    }));
  }

  double loss1 = 0.0;
  double loss2 = 0.0;
  for (int i = 0; i < numThreads; ++i) {
    threads[i].join();
    loss1 += results[i].first;
    loss2 += results[i].second;
  }

  const double mean = loss1 / datapoints.size();
  const double variance = (loss2 / datapoints.size() - mean * mean) / (datapoints.size() - 1);

  std::cout << mean << " ± " << std::sqrt(variance) << std::endl;

  return 0;
}
