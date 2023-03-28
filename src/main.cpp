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
#include "movegen/sliding.h"
#include "Evaluator.h"

using namespace ChessEngine;

#define GENERATE_MOVE_ORDERING_DATA 0

std::string repeat(const std::string text, size_t n) {
  std::string r = "";
  for (size_t i = 0; i < n; ++i) {
    r += text;
  }
  return r;
}

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
    if (parts.size() != 2) {
      throw std::runtime_error("parts.size() != 2");
    }
    Position pos(parts[0]);
    std::vector<std::string> expected = split(parts[1], ' ');

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
    std::sort(expected.begin(), expected.end());

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

size_t gLeafCounter = 0;
size_t gNodeCounter = 0;

Evaluator gEvaluator;

typedef int8_t Depth;

#ifndef NDEBUG
std::vector<std::string> gStackDebug;
#endif

// PV-nodes ("principal variation" nodes) have a score that lies between alpha and beta; their scores are exact.

// Cut-nodes are nodes which contain a beta cutoff; score is a lower bound

// All-Nodes are nodes where no score exceeded alpha; score is an upper bound
enum NodeType {
  NodeTypeAll_UpperBound,
  NodeTypeCut_LowerBound,
  NodeTypePV,
};

struct CacheResult {
  Depth depth;
  Evaluation eval;
  Move bestMove;
  NodeType nodeType;
  #ifndef NDEBUG
  std::string fen;
  #endif
};

std::unordered_map<uint64_t, CacheResult> gCache;

constexpr int kQSimplePieceValues[7] = {
  // Note "NO_PIECE" has a score of 200 since this
  // encourages qsearch to value checks.
  200, 100, 450, 500, 1000, 2000, 9999
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

template<Color PERSPECTIVE>
struct SearchResult {
  SearchResult(Evaluation score, Move move) : score(score), move(move) {}
  Evaluation score;
  Move move;
};

std::ostream& operator<<(std::ostream& stream, SearchResult<Color::WHITE> sr) {
  stream << "(" << sr.score << ", " << sr.move << ")";
  return stream;
}

template<Color COLOR>
SearchResult<opposite_color<COLOR>()> flip(SearchResult<COLOR> r) {
  return SearchResult<opposite_color<COLOR>()>(r.score * -1, r.move);
}

template<Color COLOR>
SearchResult<Color::WHITE> to_white(SearchResult<COLOR> r);

template<>
SearchResult<Color::WHITE> to_white(SearchResult<Color::WHITE> r) {
  return r;
}

template<>
SearchResult<Color::WHITE> to_white(SearchResult<Color::BLACK> r) {
  return flip(r);
}

// TODO: qsearch can leave you in check
template<Color TURN>
SearchResult<TURN> qsearch(Position *pos, int32_t depth, Evaluation alpha, Evaluation beta) {
  ++gNodeCounter;

  if (pos->is_draw()) {
    return SearchResult<TURN>(0, kNullMove);
  }

  if (std::popcount(pos->pieceBitboards_[coloredPiece<TURN, Piece::KING>()]) == 0) {
    return SearchResult<TURN>(kMissingKing, kNullMove);
  }

  if (depth > 8) {
    Evaluation e = gEvaluator.score<TURN>(*pos);
    return SearchResult<TURN>(e, kNullMove);
  }

  constexpr Color opposingColor = opposite_color<TURN>();
  constexpr ColoredPiece moverKing = coloredPiece<TURN, Piece::KING>();
  const bool inCheck = can_enemy_attack<TURN>(*pos, lsb(pos->pieceBitboards_[moverKing]));

  ExtMove moves[kMaxNumMoves];
  ExtMove *end = compute_moves<TURN, MoveGenType::CHECKS_AND_CAPTURES>(*pos, moves);

  if (moves == end && inCheck) {
    return SearchResult<TURN>(kCheckmate, kNullMove);
  }

  // If we can stand pat for a beta cutoff, or if we have no moves, return.
  SearchResult<TURN> r(gEvaluator.score<TURN>(*pos) - 50, kNullMove);
  if (moves == end || r.score >= beta) {
    return r;
  }

  if (inCheck) {
    // Cannot stand pat if you're in check.
    r.score = kCheckmate;
  }

  const Bitboard theirTargets = compute_my_targets_except_king<opposingColor>(*pos);
  const Bitboard theirUnprotected = ~theirTargets;

  for (ExtMove *move = moves; move < end; ++move) {
    move->score = kQSimplePieceValues[move->capture] - kQSimplePieceValues[move->piece];
    move->score += kQSimplePieceValues[move->piece] * ((theirUnprotected & bb(move->move.to)) > 0);
  }

  std::sort(moves, end, [](ExtMove a, ExtMove b) {
    return a.score > b.score;
  });

  for (ExtMove *move = moves; move < end; ++move) {
    if (move->score < 0 && r.score > kLongestForcedMate) {
      break;
    }

    make_move<TURN>(pos, move->move);

    SearchResult<TURN> child = flip(qsearch<opposingColor>(pos, depth + 1, -beta, -alpha));

    if (child.score > r.score) {
      r.score = child.score;
      r.move = move->move;
    }

    undo<TURN>(pos);

    if (r.score >= beta) {
      break;
    }

    alpha = std::max(alpha, child.score);
  }

  return r;
}

constexpr Evaluation kMovePieceBonus[7] = {
  0,  -107, -128, -272, -186, -267, -124 };
constexpr Evaluation kMovePieceBonus_Capture[7] = {
  0,  547,    6,  120,   30, -184,   -1 };
constexpr Evaluation kMovePieceBonus_WeAreHanging[7] = {
  0, -81,  124,  145,  -68, -194,  334 };
constexpr Evaluation kCapturePieceBonus[7] = {
  0, -423,   -5,   51,  278,  617,  9999 }; // todo: 999 should be zero?
constexpr Evaluation kCapturePieceBonus_Hanging[7] = {
  0, -543,  142,  221,  254,  533,   -1};

uint32_t gHistoryHeuristicTable[Color::NUM_COLORS][64][64];

void reset_stuff() {
  gLeafCounter = 0;
  gNodeCounter = 0;
  gCache.clear();
  std::fill_n(gHistoryHeuristicTable[Color::WHITE][0], 64 * 64, 0);
  std::fill_n(gHistoryHeuristicTable[Color::BLACK][0], 64 * 64, 0);
}

template<Color TURN>
SearchResult<TURN> search(
  Position* pos, const Depth depth,
  Evaluation alpha, const Evaluation beta,
  RecommendedMoves recommendedMoves) {

  // alpha: a score we're guaranteed to get
  //  beta: a score our opponent is guaranteed to get
  //
  // if r.score >= beta
  //   we know our opponent will never let this position occur
  //
  // if r.score >= alpha
  //   we have just found a way to do better

  ++gNodeCounter;

  constexpr Color opposingColor = opposite_color<TURN>();
  constexpr ColoredPiece moverKing = coloredPiece<TURN, Piece::KING>();

  if (std::popcount(pos->pieceBitboards_[coloredPiece<TURN, Piece::KING>()]) == 0) {
    return SearchResult<TURN>(kMissingKing, kNullMove);
  }

  if (depth <= 0) {
    ++gLeafCounter;
    return qsearch<TURN>(pos, 0, alpha, beta);
  }

  if (pos->is_draw()) {
    SearchResult<TURN>(Evaluation(0), kNullMove);
  }

  auto it = gCache.find(pos->hash_);
  if (it != gCache.end() && it->second.depth >= depth) {
    const CacheResult& cr = it->second;
    if (cr.nodeType == NodeTypePV) {
      return SearchResult<TURN>(cr.eval, cr.bestMove);
    }
    if (cr.nodeType == NodeTypeCut_LowerBound && cr.eval >= beta) {
      return SearchResult<TURN>(cr.eval, cr.bestMove);
    }
  }
  
  if (depth == 1) {  // Ignore moves that easily caused a cutoff last search.
    const Evaluation deltaPerDepth = 70;
    SearchResult<TURN> r = qsearch<TURN>(pos, 0, alpha, beta);
    if (r.score >= beta + deltaPerDepth * depth) {
      return r;
    }
    if (r.score <= alpha - deltaPerDepth * depth) {
      return r;
    }
  }

  Bitboard ourPieces = pos->colorBitboards_[TURN] & ~pos->pieceBitboards_[coloredPiece<TURN, Piece::PAWN>()];
  const bool inCheck = can_enemy_attack<TURN>(*pos, lsb(pos->pieceBitboards_[moverKing]));

  if (depth >= 3 && std::popcount(ourPieces) > 1 && !inCheck) {
    make_nullmove<TURN>(pos);
    SearchResult<TURN> a = flip(search<opposingColor>(pos, depth - 3, -beta, -alpha, RecommendedMoves()));
    if (a.score >= beta && a.move != kNullMove) {
      undo_nullmove<TURN>(pos);
      return SearchResult<TURN>(beta + 1, kNullMove);
    }
    undo_nullmove<TURN>(pos);
  }

  Move lastFoundBestMove = (it != gCache.end() ? it->second.bestMove : kNullMove);

  #ifndef NDEBUG
  std::string fen0 = pos->fen();
  #endif

  SearchResult<TURN> r(
    kMinEval + 1,
    kNullMove
  );

  ExtMove moves[kMaxNumMoves];
  ExtMove *end = compute_moves<TURN, MoveGenType::ALL_MOVES>(*pos, moves);

  if (end - moves == 0) {
    if (inCheck) {
      return r;
    } else {
      return SearchResult<TURN>(Evaluation(0), kNullMove);
    }
  }

  const Bitboard ourTargets = compute_my_targets<TURN>(*pos);
  const Bitboard theirTargets = compute_my_targets<opposingColor>(*pos);
  const Bitboard theirHanging = ourTargets & pos->colorBitboards_[opposingColor] & ~theirTargets;
  const Bitboard ourHanging = theirTargets & pos->colorBitboards_[TURN] & ~ourTargets;

  const Move lastMove = pos->history_.size() > 0 ? pos->history_.back().move : kNullMove;

  for (ExtMove *move = moves; move < end; ++move) {
    move->score = 0;

    const bool isCapture = (move->capture != Piece::NO_PIECE);
    const bool areWeHanging = ((bb(move->move.from) & ourHanging) > 0);
    const bool areTheyHanging = isCapture && ((bb(move->move.to) & theirHanging) > 0);

    move->score += kMovePieceBonus[move->piece];
    move->score += kMovePieceBonus_Capture[move->piece * (move->capture != Piece::NO_PIECE)];
    move->score += kMovePieceBonus_WeAreHanging[move->piece * areWeHanging];
    move->score += kCapturePieceBonus[move->capture];
    move->score += kCapturePieceBonus_Hanging[move->capture * areTheyHanging];

    move->score += areWeHanging * 258;
    move->score += areTheyHanging * 607;
    move->score += (move->move == lastFoundBestMove) * (depth == 1) * 592;
    move->score += (move->move == lastFoundBestMove) * (depth == 2) * 420;
    move->score += (move->move == lastFoundBestMove) * (depth >= 3) * 465;
    // move->score += (kNullMove == lastFoundBestMove) * -67;  // unnecessary

    move->score += (move->move == recommendedMoves.moves[0]) * 237;
    // move->score += (kNullMove == recommendedMoves.moves[0]) * 65;
    move->score += (move->move == recommendedMoves.moves[1]) * 300;
    // move->score += (kNullMove == recommendedMoves.moves[1]) * 44;
    move->score += (move->move.to == lastMove.to) * 250;
    move->score += isCapture * 518;

    const int32_t history = gHistoryHeuristicTable[TURN][move->move.from][move->move.to];
    // move->score += depth * 40;
    move->score += (history > 0) * 87;
    move->score += (history > 4) * 66;
    move->score += (history > 16) * 98;
    move->score += (history > 64) * 126;
    move->score += (history > 256) * 153;
  }

  #if GENERATE_MOVE_ORDERING_DATA
  size_t foo = rand() % 50000;
  if (foo == 0 || (depth > 1 && foo < 5) || (depth > 2 && foo < 25)) {
    const Bitboard ourTargets = compute_my_targets<TURN>(*pos);
    const Bitboard theirTargets = compute_my_targets<opposingColor>(*pos);

    const Bitboard theirHanging = ourTargets & pos->colorBitboards_[opposingColor] & ~theirTargets;
    const Bitboard ourHanging = theirTargets & pos->colorBitboards_[TURN] & ~ourTargets;

    std::cout << "<movedata>" << std::endl;
    std::cout << pos->fen() << std::endl;
    for (ExtMove *move = moves; move < end; ++move) {

      const bool isCapture = (move->capture != Piece::NO_PIECE);
      const bool areWeHanging = ((bb(move->move.from) & ourHanging) > 0);
      const bool areTheyHanging = isCapture && ((bb(move->move.to) & theirHanging) > 0);

      std::vector<int> features;
      // [-107 -128 -272 -186 -267 -124]
      features.push_back(move->piece == Piece::PAWN);
      features.push_back(move->piece == Piece::KNIGHT);
      features.push_back(move->piece == Piece::BISHOP);
      features.push_back(move->piece == Piece::ROOK);
      features.push_back(move->piece == Piece::QUEEN);
      features.push_back(move->piece == Piece::KING);

      //  [ 547    6  120   30 -184   -1]
      features.push_back((move->piece == Piece::PAWN) && isCapture);
      features.push_back((move->piece == Piece::KNIGHT) && isCapture);
      features.push_back((move->piece == Piece::BISHOP) && isCapture);
      features.push_back((move->piece == Piece::ROOK) && isCapture);
      features.push_back((move->piece == Piece::QUEEN) && isCapture);
      features.push_back((move->piece == Piece::KING) && isCapture);

      // [ -81  124  145  -68 -194  334]
      features.push_back((move->piece == Piece::PAWN) && areWeHanging);
      features.push_back((move->piece == Piece::KNIGHT) && areWeHanging);
      features.push_back((move->piece == Piece::BISHOP) && areWeHanging);
      features.push_back((move->piece == Piece::ROOK) && areWeHanging);
      features.push_back((move->piece == Piece::QUEEN) && areWeHanging);
      features.push_back((move->piece == Piece::KING) && areWeHanging);

      // [-423   -5   51  278  617    0]
      features.push_back(move->capture == Piece::PAWN);
      features.push_back(move->capture == Piece::KNIGHT);
      features.push_back(move->capture == Piece::BISHOP);
      features.push_back(move->capture == Piece::ROOK);
      features.push_back(move->capture == Piece::QUEEN);
      features.push_back(move->capture == Piece::KING);

      // [-543  142  221  254  533   -1]
      features.push_back((move->piece == Piece::PAWN) * areTheyHanging);
      features.push_back((move->piece == Piece::KNIGHT) * areTheyHanging);
      features.push_back((move->piece == Piece::BISHOP) * areTheyHanging);
      features.push_back((move->piece == Piece::ROOK) * areTheyHanging);
      features.push_back((move->piece == Piece::QUEEN) * areTheyHanging);
      features.push_back((move->piece == Piece::KING) * areTheyHanging);

      // [ 258  607  592  420  465  -67]
      features.push_back(areWeHanging);
      features.push_back(areTheyHanging);
      features.push_back((move->move == lastFoundBestMove) * (depth == 1));
      features.push_back((move->move == lastFoundBestMove) * (depth == 2));
      features.push_back((move->move == lastFoundBestMove) * (depth >= 3));
      features.push_back(lastFoundBestMove == kNullMove);

      // [ 237   65  300   44  250  518]
      features.push_back(recommendedMoves.moves[0] == move->move);
      features.push_back(recommendedMoves.moves[0] == kNullMove);
      features.push_back(recommendedMoves.moves[1] == move->move);
      features.push_back(recommendedMoves.moves[1] == kNullMove);
      features.push_back(move->move.to == lastMove.to);
      features.push_back(isCapture);

      // [  -6   87   66   98  126  153]
      features.push_back(depth);

      features.push_back(gHistoryHeuristicTable[TURN][move->move.from][move->move.to] > 0);
      features.push_back(gHistoryHeuristicTable[TURN][move->move.from][move->move.to] > 4);
      features.push_back(gHistoryHeuristicTable[TURN][move->move.from][move->move.to] > 16);
      features.push_back(gHistoryHeuristicTable[TURN][move->move.from][move->move.to] > 64);
      features.push_back(gHistoryHeuristicTable[TURN][move->move.from][move->move.to] > 256);

      std::cout << move->move.uci();
      for (size_t i = 0; i < features.size(); ++i) {
        std::cout << " " << features[i];
      }
      std::cout << std::endl;
    }
    std::cout << "</movedata>" << std::endl;
    exit(0);
  }
  #endif

  std::sort(moves, end, [](ExtMove a, ExtMove b) {
    return a.score > b.score;
  });

  RecommendedMoves recommendationsForChildren;

  NodeType nodeType = NodeTypePV;

  size_t numValidMoves = 0;
  bool isPV = true;
  for (ExtMove *extMove = moves; extMove < end; ++extMove) {

    #ifndef NDEBUG
    gStackDebug.push_back(extMove->uci());
    pos->assert_valid_state("a " + extMove->uci());
    #endif

    #ifndef NDEBUG
    const size_t h0 = pos->hash_;
    #endif

    make_move<TURN>(pos, extMove->move);

    // Don't move into check.
    if (can_enemy_attack<TURN>(*pos, lsb(pos->pieceBitboards_[moverKing]))) {
      undo<TURN>(pos);
      continue;
    }

    ++numValidMoves;

    SearchResult<TURN> a = flip(search<opposingColor>(pos, depth - 1, -beta, -alpha, recommendationsForChildren));
    a.score -= (a.score > -kLongestForcedMate);
    a.score += (a.score < kLongestForcedMate);

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

    if (a.score > r.score) {
      r.score = a.score;
      r.move = extMove->move;
      recommendationsForChildren.add(a.move);
      if (r.score >= beta) {
        nodeType = NodeTypeCut_LowerBound;
        gHistoryHeuristicTable[TURN][r.move.from][r.move.to] += depth * depth;
        break;
      }
      if (r.score > alpha) {
        alpha = r.score;
        isPV = false;
      }

    }
  }

  if (r.score <= alpha) {
    nodeType = NodeTypeAll_UpperBound;
  }

  if (numValidMoves == 0) {
    r.score = inCheck ? kCheckmate : 0;
    r.move = kNullMove;
  }

  it = gCache.find(pos->hash_);  // Need to re-search since the iterator may have changed when searching my children.
  if (it == gCache.end() || depth > it->second.depth) {
    const CacheResult cr = CacheResult{
      depth,
      r.score,
      r.move,
      nodeType,
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

  return r;
}

template<Color TURN>
SearchResult<TURN> search_with_aspiration_window(Position* pos, Depth depth, SearchResult<TURN> lastResult) {
  // TODO: the aspiration window technique used here should probably be implemented for internal nodes too.
  // Even just using this at the root node gives my engine a +0.25 (n=100) score against itself.
  // Table of historical experiments (program with window vs program without)
  // kBuffer  |  Score
  //      75  |  +0.10 (n=100)
  //      50  |  +0.25 (n=100)
  //      25  |  +0.14 (n=100)
  // TODO: only widen the bounds on the side that fails?
  constexpr Evaluation kBuffer = 50;
  SearchResult<TURN> r = search<TURN>(pos, depth, lastResult.score - kBuffer, lastResult.score + kBuffer, RecommendedMoves());
  if (r.score > lastResult.score - kBuffer && r.score < lastResult.score + kBuffer) {
    return r;
  }
  return search<TURN>(pos, depth, kMinEval, kMaxEval, RecommendedMoves());
}

// Gives scores from white's perspective
SearchResult<Color::WHITE> search(Position* pos, Depth depth, SearchResult<Color::WHITE> lastResult) {
  if (pos->turn_ == Color::WHITE) {
    return search_with_aspiration_window<Color::WHITE>(pos, depth, lastResult);
  } else {
    return flip(search_with_aspiration_window<Color::BLACK>(pos, depth, flip(lastResult)));
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
void print_feature_vec(Position *pos, const std::string& originalFen, bool humanReadable, bool makeQuiet) {
  if (makeQuiet) {
    SearchResult<Color::WHITE> r = to_white(qsearch<TURN>(pos, 0, kMinEval, kMaxEval));
    if (r.score > kMaxEval - 100 || r.score < kMinEval + 100) {
      std::cout << "PRINT FEATURE VEC FAIL (MATE)" << std::endl;
      return;
    }
    if (r.move != kNullMove) {
      make_move<TURN>(pos, r.move);
      print_feature_vec<opposite_color<TURN>()>(pos, originalFen, humanReadable, true);
      undo<TURN>(pos);
      return;
    }
  }

  gEvaluator.features[EF::OUR_PAWNS] = 10;
  Evaluation e = gEvaluator.score<TURN>(*pos);
  if (gEvaluator.features[EF::OUR_PAWNS] == 10) {
    std::cout << "PRINT FEATURE VEC FAIL (SHORT-CIRCUIT)" << std::endl;
  }
  if (humanReadable) {
    std::cout << "ORIGINAL_FEN " << originalFen << std::endl;
    std::cout << "FEN " << pos->fen() << std::endl;
    std::cout << "SCORE " << e << std::endl;
    const int32_t t = gEvaluator.features[EF::TIME];
    for (size_t i = 0; i < EF::NUM_EVAL_FEATURES; ++i) {
      const int32_t x = gEvaluator.features[i];
      const int32_t s = (kEarlyW0[i] * x * t + kLateW0[i] * x * (16 - t)) / 16;
      std::cout << gEvaluator.features[i] << " " << std::setfill(' ') << std::setw(4) << s << " " << EFSTR[i] << std::endl;
    }
  } else {
    std::cout << pos->fen() << std::endl;
    for (size_t i = 0; i < EF::NUM_EVAL_FEATURES; ++i) {
      if (i != 0) {
        std::cout << " ";
      }
      std::cout << gEvaluator.features[i];
    }
    std::cout << std::endl;
  }
}

bool is_uci(const std::string& text) {
  if (text.size() < 4) {
    return false;
  }
  if (text.size() > 5) {
    return false;
  }
  if (text[0] < 'a' | text[0] > 'h') {
    return false;
  }
  if (text[1] < '1' | text[1] > '8') {
    return false;
  }
  if (text[2] < 'a' | text[2] > 'h') {
    return false;
  }
  if (text[3] < '1' | text[3] > '8') {
    return false;
  }
  if (text.size() == 4) {
    return true;
  }
  return text[4] == 'n' || text[4] == 'b' || text[4] == 'r' || text[4] == 'q';
}

void mymain(std::vector<Position>& positions, const std::string& mode, double timeLimitMs, Depth depth, uint64_t nodeLimit, bool makeQuiet) {
  if (mode == "printvec" || mode == "printvec-cpu") {
    for (auto pos : positions) {
      if (pos.turn_ == Color::WHITE) {
        print_feature_vec<Color::WHITE>(&pos, pos.fen(), mode == "printvec", makeQuiet);
      } else {
        print_feature_vec<Color::BLACK>(&pos, pos.fen(), mode == "printvec", makeQuiet);
      }
    }
    return;
  } else if (mode == "analyze") {
    for (auto pos : positions) {
      reset_stuff();
      SearchResult<Color::WHITE> results(Evaluation(0), kNullMove);
      time_t tstart = clock();
      for (size_t i = 1; i <= depth; ++i) {
        results = search(&pos, i, results);
        if (positions.size() == 1) {
          const double secs = double(clock() - tstart)/CLOCKS_PER_SEC;
          std::cout << i << " : " << results.move << " : " << results.score << " (" << secs << " secs, " << gNodeCounter / secs / 1000 << " kNodes/sec)" << std::endl;
        }
        if (gNodeCounter >= nodeLimit) {
          break;
        }
        if (double(clock() - tstart)/CLOCKS_PER_SEC*1000 >= timeLimitMs) {
          break;
        }
      }

      auto it = gCache.find(pos.hash_);
      std::vector<uint64_t> oldHashes = {pos.hash_};
      size_t i = 0;
      while (it != gCache.end()) {
        if (++i > 40) {
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
        if (std::find(oldHashes.begin(), oldHashes.end(), pos.hash_) != oldHashes.end()) {
          oldHashes.push_back(pos.hash_);
          std::cout << "loop" << std::endl;
          break;
        }
        oldHashes.push_back(pos.hash_);
        it = gCache.find(pos.hash_);
      }
    }
  } else if (mode == "play") {
    for (auto pos : positions) {
      while (true) {
        reset_stuff();
        SearchResult<Color::WHITE> results(Evaluation(0), kNullMove);
        time_t tstart = clock();
        for (size_t i = 1; i <= depth; ++i) {
          results = search(&pos, i, results);
          if (gNodeCounter >= nodeLimit) {
            break;
          }
          if (double(clock() - tstart)/CLOCKS_PER_SEC*1000 >= timeLimitMs) {
            break;
          }
        }
        if (results.move == kNullMove) {
          break;
        }
        std::cout << results.move << " " << std::flush;
        if (pos.turn_ == Color::WHITE) {
          make_move<Color::WHITE>(&pos, results.move);
        } else {
          make_move<Color::BLACK>(&pos, results.move);
        }
      }
      std::cout << std::endl;
    }
  }
}

void make_uci_move(Position *pos, std::string uciMove) {
  ExtMove moves[256];
  ExtMove *end;
  if (pos->turn_ == Color::WHITE) {
    end = compute_legal_moves<Color::WHITE>(pos, moves);
  } else {
    end = compute_legal_moves<Color::BLACK>(pos, moves);
  }
  ExtMove *move;
  for (move = moves; move < end; ++move) {
    if (move->move.uci() == uciMove) {
      break;
    }
  }
  if (move->move.uci() != uciMove) {
    throw std::runtime_error("Unrecognized uci move \"" + uciMove + "\"");
    exit(1);
  }
  if (pos->turn_ == Color::WHITE) {
    make_move<Color::WHITE>(pos, move->move);
  } else {
    make_move<Color::BLACK>(pos, move->move);
  }
}

int main(int argc, char *argv[]) {
  signal(SIGSEGV, handler);
  #ifndef NDEBUG
  signal(SIGABRT, handler);
  #endif

  initialize_geometry();
  initialize_zorbrist();
  initialize_sliding();

  std::vector<std::string> args;
  for (size_t i = 1; i < argc; ++i) {
    args.push_back(argv[i]);
  }

  // test1();
  // test_moves();

  Depth depth = 50;
  std::string mode = "analyze";
  double timeLimitMs = 60000.0;
  std::string fenFile;
  std::string fen;
  uint64_t limitfens = 999999999;
  bool makeQuiet = false;
  std::vector<std::string> uciMoves;
  size_t nodeLimit = -1;

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
    } else if (args.size() >= 2 && args[0] == "nodes") {
      nodeLimit = std::stoi(args[1]);
      if (nodeLimit < 1) {
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
    } else if (args.size() >= 2 && args[0] == "limitfens") {
      limitfens = std::stoi(args[1]);
      args = std::vector<std::string>(args.begin() + 2, args.end());
    } else if (args.size() >= 1 && args[0] == "moves") {
      size_t i = 0;
      while (++i < args.size() && is_uci(args[i])) {
        uciMoves.push_back(args[i]);
      }
      args = std::vector<std::string>(args.begin() + uciMoves.size() + 1, args.end());
    } else if (args.size() >= 2 && args[0] == "makequiet") {
      if (args[1] != "0" && args[1] != "1") {
        std::cout << "makequiet must be \"0\" or \"1\" but is \"" << args[1] << "\"" << std::endl;
        return 1;
      }
      makeQuiet = (args[1] == "1");
      args = std::vector<std::string>(args.begin() + 2, args.end());
    } else {
      std::cout << "Cannot understand arguments" << std::endl;
      return 1;
    }
  }

  if (mode != "evaluate" && mode != "analyze" && mode != "play" && mode != "printvec" && mode != "printvec-cpu") {
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

  std::vector<Position> positions;

  if (fenFile.size() > 0) {
    std::ifstream infile(fenFile);
    std::string line;
    while (std::getline(infile, line)) {
      positions.push_back(Position(line));
      if (positions.size() >= limitfens) {
        break;
      }
    }
  } else {
    positions.push_back(Position(fen));
  }

  if (uciMoves.size() != 0 && positions.size() != 1) {
    throw std::runtime_error("cannot provide moves if there is more than one fen");
  }

  if (uciMoves.size() > 0) {
    for (size_t i = 0; i < uciMoves.size(); ++i) {
      make_uci_move(&positions[0], uciMoves[i]);
    }
  }

  mymain(positions, mode, timeLimitMs, depth, nodeLimit, makeQuiet);

  return 0;
}
