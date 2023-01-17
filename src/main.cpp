// Production:
// g++ src/*.cpp -std=c++20 -O3 -DNDEBUG
// 
// Debug:
// g++ src/*.cpp -std=c++20 -std=c++20 -rdynamic -g1
//
// To Generate train.txt
// ./a.out mode printvec fens eval.txt > ./train.txt
//
// To evaluate changes
// ./a.out mode evaluate fens eval.txt depth 2

#include <cassert>
#include <cstdint>
#include <ctime>
#include <cstdlib>

#include <string>
#include <vector>
#include <iostream>
#include <algorithm>
#include <random>
#include <chrono>
#include <fstream>
#include <unordered_map>

#include <stdio.h>
#include <execinfo.h>
#include <signal.h>
#include <stdlib.h>
#include <unistd.h>
#include <signal.h>    
#include <stdio.h>
#include <stdlib.h>
#include <execinfo.h>
#include <unistd.h>


#include "geometry.h"
#include "utils.h"
#include "Position.h"
#include "movegen.h"
#include "Evaluator.h"

using namespace ChessEngine;

void test_moves() {
  ExtMove moves[kMaxNumMoves];
  ExtMove *end;

  std::ifstream infile("move_test_data.txt");
  std::string line;
  size_t counter = 0;
  while (std::getline(infile, line)) {
    ++counter;
    // std::cout << counter << std::endl;
    // if (counter != 19) {
    //   continue;
    // }
    std::vector<std::string> parts = split(line, ':');
    assert(parts.size() == 2);
    Position pos(parts[0]);
    const std::vector<std::string> expected = split(parts[1], ' ');

    const size_t h0 = pos.hash_;

    if (pos.turn_ == Color::WHITE) {
      end = compute_legal_moves<Color::WHITE>(&pos, moves);
    } else {
      end = compute_legal_moves<Color::BLACK>(&pos, moves);
    }

    const size_t h1 = pos.hash_;

    if (h0 != h1) {
      throw std::runtime_error("h0 != h1");
    }

    const size_t n = end - moves;
    std::sort(moves, end, [](ExtMove a, ExtMove b) {
      return a.uci() < b.uci();
    });

    if (n != expected.size()) {
      std::cout << "counter: " << counter << std::endl;
      std::cout << pos << std::endl;
      std::cout << parts[0] << std::endl;
      for (size_t i = 0; i < std::max(expected.size(), n); ++i) {
        if (i < expected.size()) {
          std::cout << expected[i];
        } else {
          std::cout << "    ";
        }
        std::cout << " : ";
        if (i < n) {
          std::cout << moves[i].uci();
        } else {
          std::cout << "    ";
        }
        if (i < expected.size() && i < n && moves[i].uci() != expected[i]) {
          std::cout << " *";
        }
        std::cout << std::endl;
      }
      throw std::runtime_error("test_moves error");
    }
  }

  std::cout << "tested " << counter << " positions' move generations" << std::endl;
}

void test1() {
  assert(compute_colored_piece(Piece::PAWN, Color::WHITE) == ColoredPiece::WHITE_PAWN);
  assert(compute_colored_piece(Piece::KING, Color::BLACK) == ColoredPiece::BLACK_KING);

  assert(cp2color(ColoredPiece::WHITE_PAWN) == Color::WHITE);
  assert(cp2color(ColoredPiece::WHITE_KING) == Color::WHITE);
  assert(cp2color(ColoredPiece::BLACK_PAWN) == Color::BLACK);
  assert(cp2color(ColoredPiece::BLACK_KING) == Color::BLACK);
  assert(cp2color(ColoredPiece::NO_COLORED_PIECE) == Color::NO_COLOR);

  assert(colored_piece_to_char(ColoredPiece::WHITE_PAWN) == 'P');
  assert(colored_piece_to_char(ColoredPiece::WHITE_KING) == 'K');
  assert(colored_piece_to_char(ColoredPiece::BLACK_PAWN) == 'p');
  assert(colored_piece_to_char(ColoredPiece::BLACK_KING) == 'k');

  assert(std::vector<std::string>({"a", "b"}) == split("a b", ' '));
  assert(std::vector<std::string>({"", "a", "b"}) == split(" a b", ' '));
  assert(std::vector<std::string>({"", "a", "b", ""}) == split(" a b ", ' '));

  for (int i = 0; i < 64; ++i) {
    assert(string_to_square(square_to_string(Square(i))) == i);
  }

  std::random_device rd;
  std::mt19937_64 e2(rd());
  std::uniform_int_distribution<uint64_t> dist(0, uint64_t(-1));
  for (int i = 0; i < 64; ++i) {
    for (int j = 0; j < 64; ++j) {
      Square sq = Square(j);
      Bitboard randboard = dist(e2);
      uint64_t a = std::popcount(randboard & diag::kSouthWestDiagonalMask[sq]);
      uint64_t b = std::popcount(diag::southwest_diag_to_byte(sq, randboard));
      assert(a == b);
    }
  }

  {
    Position pos = Position::init();
    const std::vector<std::string> expected = {
      "Pa2a3",
      "Pb2b3",
      "Pc2c3",
      "Pd2d3",
      "Pe2e3",
      "Pf2f3",
      "Pg2g3",
      "Ph2h3",
      "Pa2a4",
      "Pb2b4",
      "Pc2c4",
      "Pd2d4",
      "Pe2e4",
      "Pf2f4",
      "Pg2g4",
      "Ph2h4",
      "Nb1a3",
      "Nb1c3",
      "Ng1f3",
      "Ng1h3",
    };
    ExtMove moves[kMaxNumMoves];
    ExtMove *end = compute_moves<Color::WHITE, MoveGenType::ALL_MOVES>(pos, moves);
    assert(expected.size() == (end - moves));
    for (size_t i = 0; i < expected.size(); ++i) {
      assert(expected[i] == moves[i].str());
    }
  }
}

/*
  0  1  2  3  4  5  6  7
  8  9 10 11 12 13 14 15
 16 17 18 19 20 21 22 23
 24 25 26 27 28 29 30 31
 32 33 34 35 36 37 38 39
 40 41 42 43 44 45 46 47
 48 49 50 51 52 53 54 55
 56 57 58 59 60 61 62 63
*/

size_t leafCounter = 0;
size_t nodeCounter = 0;

Evaluator gEvaluator;

typedef int8_t Depth;
constexpr Depth kDepthScale = 4;

#ifndef NDEBUG
std::vector<std::string> gStackDebug;
#endif

struct CacheResult {
  Depth depth;
  Evaluation eval;
  Move bestMove;
  #ifndef NDEBUG
  std::string fen;
  #endif
};

std::unordered_map<uint64_t, CacheResult> gCache;

constexpr int kSimplePieceValues[7] = {
  0, 100, 450, 500, 1000, 2000
};

constexpr int kQSimplePieceValues[7] = {
  // Note "NO_PIECE" has a score of 200 since this
  // encourages qsearch to value checks.
  200, 100, 450, 500, 1000, 2000
};


struct RecommendedMoves {
  Move moves[2];
  RecommendedMoves() {
    std::fill_n(moves, 2, kNullMove);
  }
  inline void add(Move move) {
    if (move != moves[0]) {
      moves[1] = moves[0];
      moves[0] = move;
    }
  }
  inline int score(Move move) {
    int r = 0;
    r += (move == moves[0]) * 2;
    r += (move == moves[1]) * 1;
    return r;
  }
};

// TODO: qsearch can leave you in check
template<Color TURN>
std::pair<Evaluation, Move> qsearch(Position *pos, int32_t depth, Evaluation alpha, Evaluation beta) {
  constexpr Color opposingColor = opposite_color<TURN>();
  constexpr ColoredPiece moverKing = coloredPiece<TURN, Piece::KING>();

  ExtMove moves[kMaxNumMoves];
  ExtMove *end = compute_moves<TURN, MoveGenType::CHECKS_AND_CAPTURES>(*pos, moves);

  if (moves == end) {
    Evaluation e = gEvaluator.score<TURN>(*pos);
    return std::make_pair(e, kNullMove);
  }

  const Bitboard theirTargets = compute_my_targets<opposingColor>(*pos);
  const Bitboard theirHanging = ~theirTargets & pos->colorBitboards_[opposingColor];

  for (ExtMove *move = moves; move < end; ++move) {
    move->score = kQSimplePieceValues[move->capture] - kQSimplePieceValues[move->piece];
    move->score += kQSimplePieceValues[move->piece] * ((theirHanging & bb(move->move.to)) > 0);
  }

  std::sort(moves, end, [](ExtMove a, ExtMove b) {
    return a.score > b.score;
  });

  std::pair<Evaluation, Move> r(gEvaluator.score<TURN>(*pos), kNullMove);
  if (r.first >= beta) {
    return r;
  }

  std::pair<Evaluation, Move> bestChild(Evaluation(kMinEval), kNullMove);
  for (ExtMove *move = moves; move < end; ++move) {
    if (move->score < 0) {
      break;
    }

    make_move<TURN>(pos, moves[0].move);

    std::pair<Evaluation, Move> child = qsearch<opposingColor>(pos, depth + 1, -beta, -alpha);
    child.first *= -1;

    if (child.first > r.first) {
      r.first = child.first;
      r.second = move->move;
    }

    undo<TURN>(pos);

    if (r.first >= beta) {
      return r;
    }

    alpha = std::max(alpha, child.first);
  }

  return r;
}

template<Color TURN>
std::pair<Evaluation, Move> search(Position* pos, Depth depth, Evaluation alpha, const Evaluation beta, RecommendedMoves recommendedMoves) {
  constexpr Color opposingColor = opposite_color<TURN>();
  constexpr ColoredPiece moverKing = coloredPiece<TURN, Piece::KING>();

  if (std::popcount(pos->pieceBitboards_[coloredPiece<TURN, Piece::KING>()]) == 0) {
    return std::make_pair(kMinEval, kNullMove);
  }

  const bool inCheck = can_enemy_attack<TURN>(*pos, lsb(pos->pieceBitboards_[moverKing]));

  if (depth <= 0) {
    ++leafCounter;
    std::pair<Evaluation, Move> r = qsearch<TURN>(pos, 0, alpha, beta);
    return r;
  }
  ++nodeCounter;

  if (pos->currentState_.halfMoveCounter >= 50) {
    return std::make_pair(0, kNullMove);
  }

  size_t repeatCounter = 0;
  for (size_t i = pos->hashes_.size() - 2; i < pos->hashes_.size(); i -= 2) {
    // TODO: stop looking when we hit a pawn move or capture.
    if (pos->hashes_[i] == pos->hash_) {
      repeatCounter += 1;
    }
  }
  if (repeatCounter == 3) {
    return std::make_pair(0, kNullMove);
  }

  auto it = gCache.find(pos->hash_);
  {
    if (it != gCache.end()) {
      if (it->second.depth >= depth) {
        return std::make_pair(
          it->second.eval,
          it->second.bestMove
        );
      }
    }
  }

  Move lastFoundBestMove = (it != gCache.end() ? it->second.bestMove : kNullMove);

  #ifndef NDEBUG
  std::string fen0 = pos->fen();
  #endif

  std::pair<Evaluation, Move> r(
    kMinEval + 1,
    kNullMove
  );

  ExtMove moves[kMaxNumMoves];
  ExtMove *end = compute_moves<TURN, MoveGenType::ALL_MOVES>(*pos, moves);

  if (end - moves == 0) {
    if (inCheck) {
      return r;
    } else {
      return std::make_pair(0, kNullMove);
    }
  }

  const Bitboard theirHanging = ~compute_my_targets<opposingColor>(*pos) & pos->colorBitboards_[opposingColor];
  const Move lastLastMove = pos->history_.size() > 2 ? pos->history_[pos->history_.size() - 3].move : kNullMove;
  const Move lastMove = pos->history_.size() > 0 ? pos->history_.back().move : kNullMove;

  for (ExtMove *move = moves; move < end; ++move) {
    move->score = 0;

    // When a lower piece captures a higher one, add the difference in value.
    move->score += kSimplePieceValues[move->capture] - kSimplePieceValues[move->piece];

    // Refund the moving piece's value if the captured piece is hanging.
    move->score += ((bb(move->move.to) & theirHanging) > 0) * kSimplePieceValues[move->piece];

    move->score += (move->move == lastFoundBestMove) * 10000;
    move->score += recommendedMoves.score(move->move) * 2000;

    move->score += (move->move.to == lastMove.to) * 250;
  }

  std::sort(moves, end, [](ExtMove a, ExtMove b) {
    return a.score > b.score;
  });

  RecommendedMoves recommendationsForChildren;

  size_t numValidMoves = 0;
  for (ExtMove *extMove = moves; extMove < end; ++extMove) {

    #ifndef NDEBUG
    gStackDebug.push_back(extMove->uci());
    pos->assert_valid_state("a " + extMove->uci());
    #endif

    #ifndef NDEBUG
    const size_t h0 = pos->hash_;
    #endif

    make_move<TURN>(pos, extMove->move);
    Depth depthReduction = kDepthScale;

    // Don't move into check.
    if (can_enemy_attack<TURN>(*pos, lsb(pos->pieceBitboards_[moverKing]))) {
      undo<TURN>(pos);
      continue;
    }

    ++numValidMoves;

    std::pair<Evaluation, Move> a = search<opposingColor>(pos, depth - depthReduction, -beta, -alpha, recommendationsForChildren);
    a.first *= -1;
    if (a.first > kMaxEval - 100) {
      a.first -= 1;
    }
    if (a.first < kMinEval + 100) {
      a.first += 1;
    }

    if (a.first > r.first) {
      r.first = a.first;
      r.second = extMove->move;
      recommendationsForChildren.add(a.second);
    }
    undo<TURN>(pos);

    #ifndef NDEBUG
    const size_t h1 = pos->hash_;
    if (h0 != h1) {
      throw std::runtime_error("h0 != h1");
    }
    #endif

    #ifndef NDEBUG
    gStackDebug.pop_back();
    if (pos->fen() != fen0) {
      std::cout << fen0 << std::endl;
      std::cout << pos->fen() << std::endl;
      throw std::runtime_error("fen != fen0");
    }
    pos->assert_valid_state("b " + extMove->uci());
    #endif

    if (r.first >= beta) {
      return r;
    }
    alpha = std::max(alpha, r.first);
  }

  if (numValidMoves == 0) {
    r.first = inCheck ? kMinEval + 1 : 0;
    r.second = kNullMove;
  }


  it = gCache.find(pos->hash_);  // Need to re-search since the iterator may have changed when searching my children.
  if (it == gCache.end() || depth > it->second.depth) {
    const CacheResult cr = CacheResult{
      depth,
      r.first,
      r.second,
      #ifndef NDEBUG
      pos->fen(),
      #endif
    };
    if (it != gCache.end()) {
      it->second = cr;
    } else {
      gCache.insert(std::make_pair(pos->hash_, cr));
    }
  }

  if (numValidMoves == 0) {
    if (inCheck) {
      return std::make_pair(kMinEval + 1, kNullMove);
    } else {
      return std::make_pair(0, kNullMove);
    }
  }

  return r;
}

// Gives scores from white's perspective
std::pair<Evaluation, Move> search(Position* pos, Depth depth) {
  if (pos->turn_ == Color::WHITE) {
    return search<Color::WHITE>(pos, depth, kMinEval, kMaxEval, RecommendedMoves());
  } else {
    std::pair<Evaluation, Move> r = search<Color::BLACK>(pos, depth, kMinEval, kMaxEval, RecommendedMoves());
    r.first *= -1;
    return r;
  }
}

void handler(int sig) {
  void *array[40];
  size_t size;

  // get void*'s for all entries on the stack
  size = backtrace(array, 40);

  // print out all the frames to stderr
  fprintf(stderr, "Error: signal %d:\n", sig);
  backtrace_symbols_fd(array, size, STDERR_FILENO);

  #ifndef NDEBUG
  for (size_t i = 0; i < gStackDebug.size(); ++i) {
    std::cout << gStackDebug[i] << std::endl;
  }
  #endif

  #ifndef NDEBUG
  if (gDebugPos != nullptr) {
    std::cout << "<gDebugPos>" << std::endl;
    if (gDebugPos->turn_ == Color::WHITE) {
      std::cout << "turn = white" << std::endl;
    } else {
      std::cout << "turn = black" << std::endl;
    }
    std::cout << "ep = " << unsigned(gDebugPos->currentState_.epSquare) << std::endl;
    for (size_t i = 0; i < gDebugPos->history_.size(); ++i) {
      std::cout << gDebugPos->history_[i].uci() << std::endl;
    }
    std::cout << bstr(gDebugPos->colorBitboards_[Color::BLACK]) << std::endl;
    std::cout << bstr(gDebugPos->colorBitboards_[Color::WHITE]) << std::endl;
    std::cout << "</gDebugPos>" << std::endl;
  }
  #endif

  exit(1);
}

template<Color TURN>
void print_feature_vec(Position *pos, const std::string& originalFen) {
  std::pair<Evaluation, Move> r = qsearch<TURN>(pos, 0, kMinEval, kMaxEval);
  if (r.second != kNullMove) {
    make_move<TURN>(pos, r.second);
    print_feature_vec<opposite_color<TURN>()>(pos, originalFen);
    undo<TURN>(pos);
    return;
  }

  Evaluation e = gEvaluator.score<TURN>(*pos);
  std::cout << "ORIGINAL_FEN " << originalFen << std::endl;
  std::cout << "FEN " << pos->fen() << std::endl;
  std::cout << "SCORE " << e << std::endl;
  for (size_t i = 0; i < EF::NUM_EVAL_FEATURES; ++i) {
    std::cout << gEvaluator.features[i] << " " << EFSTR[i] << std::endl;
  }
}

void mymain(std::vector<std::string>& fens, const std::string& mode, double timeLimitMs, Depth depth) {
  if (mode == "printvec") {
    for (auto fen : fens) {
      Position pos(fen);
      if (pos.turn_ == Color::WHITE) {
        print_feature_vec<Color::WHITE>(&pos, fen);
      } else {
        print_feature_vec<Color::BLACK>(&pos, fen);
      }
    }
    return;
  }
  else if (mode == "evaluate") {
    const time_t t0 = clock();

    size_t numCorrect = 0;
    size_t total = 0;

    for (auto line : fens) {
      gCache.clear();
      std::vector<std::string> parts = split(line, ':');
      if (parts.size() != 3) {
        throw std::runtime_error("Unrecognized line \"" + line + "\"");
      }
      std::string fen = parts[0];
      std::string bestMove = parts[1];
      Position pos(fen);
      std::pair<Evaluation, Move> results;
      for (size_t i = 1; i <= depth; ++i) {
        results = search(&pos, i * kDepthScale);
      }
      numCorrect += (results.second.uci() == bestMove);
      total += 1;
      if (total % 10000 == 0) {
        std::cout << numCorrect << " / " << total << std::endl;
      }
    }

    std::cout << numCorrect << " / " << total << std::endl;

    double duration = double(clock() - t0) / CLOCKS_PER_SEC;
    std::cout << duration * 1000 << "ms" << std::endl;
    std::cout << "nodeCounter = " << nodeCounter / 1000 << "k" << std::endl;
    std::cout << "leafCounter = " << leafCounter / 1000 << "k" << std::endl;
    return;
  } else if (mode == "analyze") {
    for (auto fen : fens) {
      gCache.clear();
      Position pos(fen);
      std::pair<Evaluation, Move> results;
      time_t tstart = clock();
      for (size_t i = 1; i <= depth; ++i) {
        results = search(&pos, i * kDepthScale);
        if (fens.size() == 1) {
          std::cout << i << " : " << results.second << " : " << results.first << " (" << double(clock() - tstart)/CLOCKS_PER_SEC << " secs)" << std::endl;
        }
        if (double(clock() - tstart)/CLOCKS_PER_SEC*1000 >= timeLimitMs) {
          break;
        }
      }

      auto it = gCache.find(pos.hash_);
      size_t i = 0;
      while (it != gCache.end()) {
        if (++i > 10) {
          break;
        }
        if (pos.turn_ == Color::BLACK) {
          it->second.eval *= -1;
        }
        std::cout << it->second.bestMove.uci() << " (" << it->second.eval << ", " << unsigned(it->second.depth) << ")" << std::endl;
        if (it->second.bestMove == kNullMove) {
          break;
        }
        if (pos.turn_ == Color::WHITE) {
          make_move<Color::WHITE>(&pos, it->second.bestMove);
        } else {
          make_move<Color::BLACK>(&pos, it->second.bestMove);
        }
        it = gCache.find(pos.hash_);
      }

    }
  } else if (mode == "play") {
    for (auto fen : fens) {
      Position pos(fen);
      while (true) {
        gCache.clear();
        std::pair<Evaluation, Move> results;
        time_t tstart = clock();
        for (size_t i = 1; i <= depth; ++i) {
          results = search(&pos, i * kDepthScale);
          if (double(clock() - tstart)/CLOCKS_PER_SEC*1000 >= timeLimitMs) {
            break;
          }
        }
        if (results.second == kNullMove) {
          break;
        }
        std::cout << results.second << " " << std::flush;
        if (pos.turn_ == Color::WHITE) {
          make_move<Color::WHITE>(&pos, results.second);
        } else {
          make_move<Color::BLACK>(&pos, results.second);
        }
      }
      std::cout << std::endl;
    }
  }
}

int main(int argc, char *argv[]) {
  signal(SIGSEGV, handler);
  #ifndef NDEBUG
  signal(SIGABRT, handler);
  #endif

  initialize_geometry();
  initialize_zorbrist();

  std::vector<std::string> args;
  for (size_t i = 1; i < argc; ++i) {
    args.push_back(argv[i]);
  }

  // test1();
  // test_moves();

  Position pos = Position::init();
  Depth depth = 10;
  std::string mode = "analyze";
  double timeLimitMs = 60000.0;
  std::string fenFile;
  std::string fen;

  while (args.size() > 0) {
    if (args.size() >= 7 && args[0] == "fen") {
      std::vector<std::string> fenVec(args.begin() + 1, args.begin() + 7);
      args = std::vector<std::string>(args.begin() + 7, args.end());
      fen = join(fenVec, " ");
    } else if (args.size() >= 2 && args[0] == "depth") {
      depth = std::stoi(args[1]);
      if (depth < 0) {
        throw std::invalid_argument("");
        exit(1);
      }
      args = std::vector<std::string>(args.begin() + 2, args.end());
    } else if (args.size() >= 2 && args[0] == "time") {
      timeLimitMs = std::stoi(args[1]);
      if (timeLimitMs < 1) {
        throw std::invalid_argument("");
        exit(1);
      }
      args = std::vector<std::string>(args.begin() + 2, args.end());
    } else if (args.size() >= 2 && args[0] == "mode") {
      mode = args[1];
      args = std::vector<std::string>(args.begin() + 2, args.end());
    } else if (args.size() >= 2 && args[0] == "fens") {
      fenFile = args[1];
      args = std::vector<std::string>(args.begin() + 2, args.end());
    } else {
      std::cout << "Cannot understand arguments" << std::endl;
      return 1;
    }
  }

  if (mode != "evaluate" && mode != "analyze" && mode != "play" && mode != "printvec") {
    throw std::runtime_error("Cannot recognize mode \"" + mode + "\"");
  }
  if (fenFile.size() == 0 && fen.size() == 0) {
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
  }
  if ((fenFile.size() != 0) == (fen.size() != 0)) {
    throw std::runtime_error("Cannot provide fen and fens");
  }
  if (depth <= 0) {
    throw std::runtime_error("invalid depth");
  }

  std::vector<std::string> fens;

  if (fenFile.size() > 0) {
    std::ifstream infile(fenFile);
    std::string line;
    while (std::getline(infile, line)) {
      fens.push_back(line);
    }
  } else {
    fens.push_back(fen);
  }

  mymain(fens, mode, timeLimitMs, depth);

  return 0;
}
