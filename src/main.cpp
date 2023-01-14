 // Production:
 // g++ src/*.cpp -std=c++20 -O3 -DNDEBUG
 // 
 // Debug:
 // g++ src/*.cpp -std=c++20 -std=c++20 -rdynamic -g1

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

using namespace ChessEngine;

// 1713 / 3056

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

constexpr Bitboard kCenter16 = (kFiles[2] | kFiles[3] | kFiles[4] | kFiles[5]) & (kRanks[2] | kRanks[3] | kRanks[4] | kRanks[5]);
constexpr Bitboard kCenter4 = (kFiles[3] | kFiles[4]) & (kRanks[3] | kRanks[4]);

std::vector<std::string> gEvalVecNames;

class EvalVecRType {
 public:
  Evaluation value;
  EvalVecRType(Evaluation e) {
    gEvalVecNames.clear();
    value = e;
    assert(e == 0);
  }
  std::vector<Evaluation> deltas;
  std::vector<bool> isIncrement;

  EvalVecRType operator+=(Evaluation delta) {
    value += delta;
    deltas.push_back(delta);
    isIncrement.push_back(true);
    return *this;
  }
  EvalVecRType operator-=(Evaluation delta) {
    value -= delta;
    deltas.push_back(delta);
    isIncrement.push_back(false);
    return *this;
  }

  std::string str() const {
    std::string r = "";
    assert(deltas.size() == gEvalVecNames.size() * 2);
    assert(deltas.size() % 2 == 0);
    for (size_t i = 0; i < deltas.size(); i += 2) {
      assert(isIncrement[i]);
      assert(!isIncrement[i + 1]);
      r += gEvalVecNames[i/2] + " " + std::to_string(deltas[i] - deltas[i + 1]) + "\n";
    }
    return r;
  }
};

inline Evaluation interpolate(Evaluation early, Evaluation late, Evaluation time) {
  return (early * (16 - time) + late * time) / 16;
}

inline Bitboard fatten(Bitboard b) {
  return shift<Direction::WEST>(b) | b | shift<Direction::EAST>(b);
}

enum EF {
  PAWNS,
  KNIGHTS,
  BISHOPS,
  ROOKS,
  QUEENS,

  YOUR_TURN,

  IN_CHECK,
  KING_ON_BACK_RANK,

  PASSED_PAWNS,
  ISOLATED_PAWNS,
  DOUBLED_PAWNS,

  PAWNS_CENTER_16,
  PAWNS_CENTER_4,
  ADVANCED_PASSED_PAWNS_1,
  ADVANCED_PASSED_PAWNS_2,
  ADVANCED_PASSED_PAWNS_3,
  ADVANCED_PASSED_PAWNS_4,
  PAWN_MINOR_CAPTURES,
  PAWN_MAJOR_CAPTURES,
  PROTECTED_PAWNS,
  PROTECTED_PASSED_PAWNS,
  PAWN_SPREAD,

  BISHOPS_DEVELOPED,
  BLOCKADED_BISHOPS,
  SCARY_BISHOPS,
  SCARIER_BISHOPS,

  BLOCKADED_ROOKS,
  SCARY_ROOKS,
  ROOKS_ON_7TH,

  KNIGHTS_DEVELOPED,
  KNIGHT_MAJOR_CAPTURES,
  KNIGHTS_CENTER_16,
  KNIGHTS_CENTER_4,
  KNIGHT_ON_ENEMY_SIDE,

  HANGING_WHITE_PAWNS,
  HANGING_WHITE_KNIGHTS,
  HANGING_WHITE_BISHOPS,
  HANGING_WHITE_ROOKS,
  HANGING_WHITE_QUEENS,

  HANGING_BLACK_PAWNS,
  HANGING_BLACK_KNIGHTS,
  HANGING_BLACK_BISHOPS,
  HANGING_BLACK_ROOKS,
  HANGING_BLACK_QUEENS,

  NUM_TARGET_SQUARES,

  TIME,

  NUM_EVAL_FEATURES,
};

std::string EFSTR[] = {
  "PAWNS",
  "KNIGHTS",
  "BISHOPS",
  "ROOKS",
  "QUEENS",
  "YOUR_TURN",
  "IN_CHECK",
  "KING_ON_BACK_RANK",
  "PASSED_PAWNS",
  "ISOLATED_PAWNS",
  "DOUBLED_PAWNS",
  "PAWNS_CENTER_16",
  "PAWNS_CENTER_4",
  "ADVANCED_PASSED_PAWNS_1",
  "ADVANCED_PASSED_PAWNS_2",
  "ADVANCED_PASSED_PAWNS_3",
  "ADVANCED_PASSED_PAWNS_4",
  "PAWN_MINOR_CAPTURES",
  "PAWN_MAJOR_CAPTURES",
  "PROTECTED_PAWNS",
  "PROTECTED_PASSED_PAWNS",
  "PAWN_SPREAD",
  "BISHOPS_DEVELOPED",
  "BLOCKADED_BISHOPS",
  "SCARY_BISHOPS",
  "SCARIER_BISHOPS",
  "BLOCKADED_ROOKS",
  "SCARY_ROOKS",
  "ROOKS_ON_7TH",
  "KNIGHTS_DEVELOPED",
  "KNIGHT_MAJOR_CAPTURES",
  "KNIGHTS_CENTER_16",
  "KNIGHTS_CENTER_4",
  "KNIGHT_ON_ENEMY_SIDE",
  "HANGING_WHITE_PAWNS",
  "HANGING_WHITE_KNIGHTS",
  "HANGING_WHITE_BISHOPS",
  "HANGING_WHITE_ROOKS",
  "HANGING_WHITE_QUEENS",
  "HANGING_BLACK_PAWNS",
  "HANGING_BLACK_KNIGHTS",
  "HANGING_BLACK_BISHOPS",
  "HANGING_BLACK_ROOKS",
  "HANGING_BLACK_QUEENS",
  "TIME",
  "NUM_TARGET_SQUARES",
};

struct Evaluator {
  Evaluator() {}
  Evaluation score(const Position& pos) {
    if (std::popcount(pos.pieceBitboards_[ColoredPiece::WHITE_KING]) == 0) {
      return kMinEval;
    }
    if (std::popcount(pos.pieceBitboards_[ColoredPiece::BLACK_KING]) == 0) {
      return kMaxEval;
    }

    const Square whiteKingSq = lsb(pos.pieceBitboards_[ColoredPiece::WHITE_KING]);
    const Square blackKingSq = lsb(pos.pieceBitboards_[ColoredPiece::BLACK_KING]);

    const Bitboard whitePawns = pos.pieceBitboards_[ColoredPiece::WHITE_PAWN];
    const Bitboard whiteKnights = pos.pieceBitboards_[ColoredPiece::WHITE_KNIGHT];
    const Bitboard whiteBishops = pos.pieceBitboards_[ColoredPiece::WHITE_BISHOP];
    const Bitboard whiteRooks = pos.pieceBitboards_[ColoredPiece::WHITE_ROOK];
    const Bitboard whiteQueens = pos.pieceBitboards_[ColoredPiece::WHITE_QUEEN];

    const Bitboard blackPawns = pos.pieceBitboards_[ColoredPiece::BLACK_PAWN];
    const Bitboard blackKnights = pos.pieceBitboards_[ColoredPiece::BLACK_KNIGHT];
    const Bitboard blackBishops = pos.pieceBitboards_[ColoredPiece::BLACK_BISHOP];
    const Bitboard blackRooks = pos.pieceBitboards_[ColoredPiece::BLACK_ROOK];
    const Bitboard blackQueens = pos.pieceBitboards_[ColoredPiece::BLACK_QUEEN];

    const Bitboard royalWhite = pos.pieceBitboards_[ColoredPiece::WHITE_QUEEN] | pos.pieceBitboards_[ColoredPiece::WHITE_KING];
    const Bitboard royalBlack = pos.pieceBitboards_[ColoredPiece::BLACK_QUEEN] | pos.pieceBitboards_[ColoredPiece::BLACK_KING];
    const Bitboard majorWhite = pos.pieceBitboards_[ColoredPiece::WHITE_ROOK] | royalWhite;
    const Bitboard majorBlack = pos.pieceBitboards_[ColoredPiece::BLACK_ROOK] | royalBlack;
    const Bitboard minorWhite = pos.pieceBitboards_[ColoredPiece::WHITE_KNIGHT] | pos.pieceBitboards_[ColoredPiece::WHITE_BISHOP];
    const Bitboard minorBlack = pos.pieceBitboards_[ColoredPiece::BLACK_KNIGHT] | pos.pieceBitboards_[ColoredPiece::BLACK_BISHOP];
    const Bitboard whitePieces = majorWhite | minorWhite;
    const Bitboard blackPieces = majorBlack | minorBlack;

    const Bitboard whitePawnTargets = compute_pawn_targets<Color::WHITE>(pos);
    const Bitboard blackPawnTargets = compute_pawn_targets<Color::BLACK>(pos);
    const Bitboard whiteKnightTargets = compute_knight_targets<Color::WHITE>(pos);
    const Bitboard blackKnightTargets = compute_knight_targets<Color::BLACK>(pos);
    const Bitboard whiteBishopTargets = compute_bishoplike_targets<Color::WHITE>(pos, whiteBishops);
    const Bitboard blackBishopTargets = compute_bishoplike_targets<Color::BLACK>(pos, blackBishops);
    const Bitboard whiteRookTargets = compute_rooklike_targets<Color::WHITE>(pos, whiteRooks);
    const Bitboard blackRookTargets = compute_rooklike_targets<Color::BLACK>(pos, blackRooks);
    const Bitboard whiteQueenTargets = compute_bishoplike_targets<Color::WHITE>(pos, pos.pieceBitboards_[ColoredPiece::WHITE_QUEEN]) | compute_rooklike_targets<Color::WHITE>(pos, pos.pieceBitboards_[ColoredPiece::WHITE_QUEEN]);
    const Bitboard blackQueenTargets = compute_bishoplike_targets<Color::BLACK>(pos, pos.pieceBitboards_[ColoredPiece::BLACK_QUEEN]) | compute_rooklike_targets<Color::BLACK>(pos, pos.pieceBitboards_[ColoredPiece::BLACK_QUEEN]);

    constexpr Bitboard kWhiteSide = kRanks[4] | kRanks[5] | kRanks[6] | kRanks[7];
    constexpr Bitboard kBlackSide = kRanks[0] | kRanks[1] | kRanks[2] | kRanks[3];

    const Bitboard whiteTargets = whitePawnTargets | whiteKnightTargets | whiteBishopTargets | whiteRookTargets | whiteQueenTargets;
    const Bitboard blackTargets = blackPawnTargets | blackKnightTargets | blackBishopTargets | blackRookTargets | blackQueenTargets;

    features[EF::PAWNS] = std::popcount(whitePawns) - std::popcount(blackPawns);
    features[EF::KNIGHTS] = std::popcount(whiteKnights) - std::popcount(blackKnights);
    features[EF::BISHOPS] = std::popcount(whiteBishops) - std::popcount(blackBishops);
    features[EF::ROOKS] = std::popcount(whiteRooks) - std::popcount(blackRooks);
    features[EF::QUEENS] = std::popcount(whiteQueens) - std::popcount(blackQueens);

    features[EF::YOUR_TURN] = (pos.turn_ == Color::WHITE) * 2 - 1;
    features[EF::IN_CHECK] = can_enemy_attack<Color::WHITE>(pos, whiteKingSq) - can_enemy_attack<Color::BLACK>(pos, blackKingSq);
    features[EF::KING_ON_BACK_RANK] = (7 - blackKingSq / 8) - (whiteKingSq / 8);

    // Pawns
    const Bitboard blockadedBlackPawns = shift<Direction::NORTH>(whitePawns) & blackPawns;
    const Bitboard blockadedWhitePawns = shift<Direction::SOUTH>(blockadedBlackPawns);
    const Bitboard protectedWhitePawns = whitePawns & whitePawnTargets;
    const Bitboard protectedBlackPawns = whitePawns & whitePawnTargets;
    {
      const Bitboard filledWhite = northFill(whitePawns);
      const Bitboard filledBlack = southFill(blackPawns);
      const Bitboard fatWhitePawns = fatten(filledWhite);
      const Bitboard fatBlackPawns = fatten(filledBlack);
      const Bitboard filesWithWhitePawns = southFill(filledWhite);
      const Bitboard filesWithBlackPawns = northFill(filledBlack);
      const Bitboard filesWithoutWhitePawns = ~filesWithWhitePawns;
      const Bitboard filesWithoutBlackPawns = ~filesWithBlackPawns;
      const Bitboard passedWhitePawns = whitePawns & ~fatten(filesWithBlackPawns);
      const Bitboard passedBlackPawns = blackPawns & ~fatten(filesWithWhitePawns);
      const Bitboard isolatedWhitePawns = whitePawns & shift<Direction::WEST>(filesWithoutWhitePawns) & shift<Direction::EAST>(filesWithoutWhitePawns);
      const Bitboard isolatedBlackPawns = blackPawns & shift<Direction::WEST>(filesWithoutBlackPawns) & shift<Direction::EAST>(filesWithoutBlackPawns);
      const Bitboard doubledWhitePawns = whitePawns & (filledWhite >> 8);
      const Bitboard doubledBlackPawns = blackPawns & (filledBlack << 8);
      constexpr Bitboard kRookPawns = kFiles[0] | kFiles[7];

      features[EF::PAWNS_CENTER_16] = std::popcount(whitePawns & kCenter16) - std::popcount(blackPawns & kCenter16);
      features[EF::PAWNS_CENTER_16] = std::popcount(whitePawns & kCenter16) - std::popcount(blackPawns & kCenter16);
      features[EF::PAWNS_CENTER_4] = std::popcount(whitePawns & kCenter4) - std::popcount(blackPawns & kCenter4);
      features[EF::PASSED_PAWNS] = std::popcount(passedWhitePawns) - std::popcount(passedBlackPawns);
      features[EF::ISOLATED_PAWNS] = std::popcount(isolatedWhitePawns) - std::popcount(isolatedBlackPawns);
      features[EF::DOUBLED_PAWNS] = std::popcount(doubledWhitePawns) - std::popcount(doubledBlackPawns);
      features[EF::ADVANCED_PASSED_PAWNS_1] = std::popcount(passedWhitePawns & kRanks[7]) - std::popcount(passedBlackPawns & kRanks[1]);
      features[EF::ADVANCED_PASSED_PAWNS_2] = std::popcount(passedWhitePawns & kRanks[6]) - std::popcount(passedBlackPawns & kRanks[2]);
      features[EF::ADVANCED_PASSED_PAWNS_3] = std::popcount(passedWhitePawns & kRanks[5]) - std::popcount(passedBlackPawns & kRanks[3]);
      features[EF::ADVANCED_PASSED_PAWNS_4] = std::popcount(passedWhitePawns & kRanks[4]) - std::popcount(passedBlackPawns & kRanks[4]);
      features[EF::PAWN_MINOR_CAPTURES] = std::popcount(whitePawnTargets & minorBlack) - std::popcount(blackPawnTargets & minorWhite);
      features[EF::PAWN_MAJOR_CAPTURES] = std::popcount(whitePawnTargets & majorBlack) - std::popcount(blackPawnTargets & majorWhite);
      features[EF::PROTECTED_PAWNS] = std::popcount(whitePawns & whitePawnTargets) - std::popcount(blackPawns & blackPawnTargets);
      features[EF::PROTECTED_PASSED_PAWNS] = std::popcount(passedWhitePawns & whitePawnTargets) - std::popcount(passedBlackPawns & blackPawnTargets);

      features[EF::PAWN_SPREAD] = 0;
      if (filesWithWhitePawns) {
        features[EF::PAWN_SPREAD] += msb(filesWithWhitePawns & 255) - lsb(filesWithWhitePawns & 255);
      }
      if (filesWithBlackPawns) {
        features[EF::PAWN_SPREAD] -= msb(filesWithBlackPawns & 255) - lsb(filesWithBlackPawns & 255);
      }
    }

    {  // Bishops
      const Bitboard whiteBishopTargetsIgnoringNonBlockades = compute_bishoplike_targets<Color::WHITE>(pos, whiteBishops, blockadedWhitePawns);
      const Bitboard blackBishopTargetsIgnoringNonBlockades = compute_bishoplike_targets<Color::BLACK>(pos, blackBishops, blockadedBlackPawns);
      features[EF::BISHOPS_DEVELOPED] = std::popcount(blackBishops & (bb( 2) | bb( 5))) - std::popcount(whiteBishops & (bb(58) | bb(61)));
      features[EF::BLOCKADED_BISHOPS] = std::popcount(whiteBishopTargetsIgnoringNonBlockades & (blockadedWhitePawns | protectedBlackPawns)) - std::popcount(blackBishopTargetsIgnoringNonBlockades & (blockadedBlackPawns | protectedWhitePawns));
      features[EF::SCARY_BISHOPS] = std::popcount(whiteBishopTargetsIgnoringNonBlockades & majorBlack) - std::popcount(blackBishopTargetsIgnoringNonBlockades & majorWhite);
      features[EF::SCARIER_BISHOPS] = std::popcount(whiteBishopTargetsIgnoringNonBlockades & royalBlack) - std::popcount(blackBishopTargetsIgnoringNonBlockades & royalWhite);

      // const Bitboard whitePawnsOnWhite = kWhiteSquares & whitePawns;
      // const Bitboard whitePawnsOnBlack = kBlackSquares & whitePawns;
      // const Bitboard blackPawnsOnWhite = kWhiteSquares & blackPawns;
      // const Bitboard blackPawnsOnBlack = kBlackSquares & blackPawns;
    }

    {  // Rooks
      features[EF::BLOCKADED_ROOKS] = std::popcount(whiteRookTargets & whitePawns) - std::popcount(blackRookTargets & blackPawns);
      features[EF::SCARY_ROOKS] = std::popcount(whiteRookTargets & royalBlack) - std::popcount(blackRookTargets & royalWhite);
      features[EF::ROOKS_ON_7TH] = std::popcount(whiteRooks & kRanks[1]) - std::popcount(blackRooks & kRanks[6]);
    }

    {  // Knights
      features[EF::KNIGHTS_DEVELOPED] = std::popcount(blackKnights & (bb( 1) | bb( 6))) - std::popcount(whiteKnights & (bb(57) | bb(62)));
      features[EF::KNIGHT_MAJOR_CAPTURES] = std::popcount(compute_knight_targets<Color::WHITE>(pos) & majorBlack) - std::popcount(compute_knight_targets<Color::BLACK>(pos) & majorWhite);
      features[EF::KNIGHTS_CENTER_16] = std::popcount(whiteKnights & kCenter16) - std::popcount(blackKnights & kCenter16);
      features[EF::KNIGHTS_CENTER_4] = std::popcount(whiteKnights & kCenter4) - std::popcount(blackKnights & kCenter4);
      features[EF::KNIGHT_ON_ENEMY_SIDE] = std::popcount(whiteKnights & kBlackSide) - std::popcount(blackKnights & kWhiteSide);
    }

    {
      // Hanging pieces are more valuable if it is your turn since they're literally free material,
      // as opposed to threats. Also is a very useful heuristic so that leaf nodes don't sack a rook
      // for a pawn.
      const Bitboard whiteHanging = blackTargets & ~whiteTargets & pos.colorBitboards_[Color::WHITE];
      const Bitboard blackHanging = whiteTargets & ~blackTargets & pos.colorBitboards_[Color::BLACK];
      features[EF::HANGING_WHITE_PAWNS] = std::popcount(whitePawns & whiteHanging);
      features[EF::HANGING_BLACK_PAWNS] = std::popcount(blackPawns & blackHanging);
      features[EF::HANGING_WHITE_KNIGHTS] = std::popcount(whiteKnights & whiteHanging);
      features[EF::HANGING_BLACK_KNIGHTS] = std::popcount(blackKnights & blackHanging);
      features[EF::HANGING_WHITE_BISHOPS] = std::popcount(whiteBishops & whiteHanging);
      features[EF::HANGING_BLACK_BISHOPS] = std::popcount(blackBishops & blackHanging);
      features[EF::HANGING_WHITE_ROOKS] = std::popcount(whiteRooks & whiteHanging);
      features[EF::HANGING_BLACK_ROOKS] = std::popcount(blackRooks & blackHanging);
      features[EF::HANGING_WHITE_QUEENS] = std::popcount(whiteQueens & whiteHanging);
      features[EF::HANGING_BLACK_QUEENS] = std::popcount(blackQueens & blackHanging);

      features[EF::NUM_TARGET_SQUARES] = std::popcount(whiteTargets) - std::popcount(blackTargets);

    }

    const int16_t whitePiecesRemaining = std::popcount(pos.colorBitboards_[Color::WHITE] & ~pos.pieceBitboards_[ColoredPiece::WHITE_PAWN]) + std::popcount(pos.pieceBitboards_[ColoredPiece::WHITE_QUEEN]) - 1;
    const int16_t blackPiecesRemaining = std::popcount(pos.colorBitboards_[Color::BLACK] & ~pos.pieceBitboards_[ColoredPiece::BLACK_PAWN]) + std::popcount(pos.pieceBitboards_[ColoredPiece::WHITE_QUEEN]) - 1;

    // Use larger integer to make arithmetic safe.
    const int32_t earliness = (whitePiecesRemaining + blackPiecesRemaining);
    const int32_t early = this->early(pos);
    const int32_t late = this->late(pos);

    features[EF::TIME] = earliness;
    
    return (early * earliness + late * (16 - earliness)) / 16;
  }

  Evaluation early(const Position& pos) const {
    Evaluation r = 0;
    r += features[EF::PAWNS] * 100;
    r += features[EF::KNIGHTS] * 499;
    r += features[EF::BISHOPS] * 465;
    r += features[EF::ROOKS] * 365;
    r += features[EF::QUEENS] * 1093;
    r += features[EF::YOUR_TURN] * 20;
    r += features[EF::IN_CHECK] * -300;
    r += features[EF::KING_ON_BACK_RANK] * 50;

    r += features[EF::PASSED_PAWNS] * 0;
    r += features[EF::ISOLATED_PAWNS] * -30;
    r += features[EF::DOUBLED_PAWNS] * -10;
    r += features[EF::PAWNS_CENTER_16] * 10;
    r += features[EF::PAWNS_CENTER_4] * 58;
    r += features[EF::ADVANCED_PASSED_PAWNS_1] * 0;
    r += features[EF::ADVANCED_PASSED_PAWNS_2] * 0;
    r += features[EF::ADVANCED_PASSED_PAWNS_3] * 0;
    r += features[EF::ADVANCED_PASSED_PAWNS_4] * 0;
    r += features[EF::PAWN_MINOR_CAPTURES] * 50;
    r += features[EF::PAWN_MAJOR_CAPTURES] * 50;
    r += features[EF::PROTECTED_PAWNS] * 10;

    r += features[EF::BISHOPS_DEVELOPED] * 10;
    r += features[EF::BLOCKADED_BISHOPS] * -10;
    r += features[EF::SCARY_BISHOPS] * 10;
    r += features[EF::SCARIER_BISHOPS] * 30;

    r += features[EF::BLOCKADED_ROOKS] * -10;
    r += features[EF::SCARY_ROOKS] * 10;
    r += features[EF::ROOKS_ON_7TH] * 20;

    r += features[EF::KNIGHTS_DEVELOPED] * 20;
    r += features[EF::KNIGHT_MAJOR_CAPTURES] * 40;
    r += features[EF::KNIGHTS_CENTER_16] * 50;
    r += features[EF::KNIGHTS_CENTER_4] * 0;
    r += features[EF::KNIGHT_ON_ENEMY_SIDE] * 30;

    const Evaluation wScaleFactor = (pos.turn_ == Color::WHITE ? 20 : 1);
    const Evaluation bScaleFactor = (pos.turn_ == Color::BLACK ? -20 : -1);
    const Evaluation denominator = 20;

    r += features[EF::HANGING_WHITE_PAWNS] * 100 * bScaleFactor / denominator;
    r += features[EF::HANGING_WHITE_KNIGHTS] * 300 * bScaleFactor / denominator;
    r += features[EF::HANGING_WHITE_BISHOPS] * 300 * bScaleFactor / denominator;
    r += features[EF::HANGING_WHITE_ROOKS] * 500 * bScaleFactor / denominator;
    r += features[EF::HANGING_WHITE_QUEENS] * 900 * bScaleFactor / denominator;

    r += features[EF::HANGING_BLACK_PAWNS] * 100 * wScaleFactor / denominator;
    r += features[EF::HANGING_BLACK_KNIGHTS] * 300 * wScaleFactor / denominator;
    r += features[EF::HANGING_BLACK_BISHOPS] * 300 * wScaleFactor / denominator;
    r += features[EF::HANGING_BLACK_ROOKS] * 500 * wScaleFactor / denominator;
    r += features[EF::HANGING_BLACK_QUEENS] * 900 * wScaleFactor / denominator;

    r += features[EF::NUM_TARGET_SQUARES] * 4;

    return r;
  }
  Evaluation late(const Position& pos) const {
    Evaluation r = 0;
    r += features[EF::PAWNS] * 100;
    r += features[EF::KNIGHTS] * 436;
    r += features[EF::BISHOPS] * 539;
    r += features[EF::ROOKS] * 1561;
    r += features[EF::QUEENS] * 3101;
    r += features[EF::YOUR_TURN] * 10;
    r += features[EF::IN_CHECK] * -200;

    r += features[EF::PASSED_PAWNS] * 30;
    r += features[EF::DOUBLED_PAWNS] * -20;
    r += features[EF::ADVANCED_PASSED_PAWNS_1] * 140;
    r += features[EF::ADVANCED_PASSED_PAWNS_2] * 80;
    r += features[EF::ADVANCED_PASSED_PAWNS_3] * 40;
    r += features[EF::ADVANCED_PASSED_PAWNS_4] * 25;
    r += features[EF::PAWN_MINOR_CAPTURES] * 50;
    r += features[EF::PAWN_MAJOR_CAPTURES] * 50;
    // r += features[EF::PROTECTED_PAWNS] * 10;
    r += features[EF::PROTECTED_PASSED_PAWNS] * 10;
    // r += features[EF::PAWN_SPREAD] * 20;

    r += features[EF::BLOCKADED_BISHOPS] * -10;
    r += features[EF::SCARY_BISHOPS] * 30;

    r += features[EF::BLOCKADED_ROOKS] * -20;
    r += features[EF::SCARY_ROOKS] * 10;
    r += features[EF::ROOKS_ON_7TH] * 10;

    return r;
  }
  Evaluation features[NUM_EVAL_FEATURES];
};

Evaluator gEvaluator;

typedef int8_t Depth;
constexpr Depth kDepthScale = 4;

#ifndef NDEBUG
std::vector<std::string> gStackDebug;
#endif

/*

parent looking high value {
  sibling looking for low value: returns 10
  you: finds a 9; stop looking (parent will choose other sibling)
}

*/

struct CacheResult {
  Depth depth;
  Evaluation eval;
  Move bestMove;
  #ifndef NDEBUG
  std::string fen;
  #endif
};

std::string history_string(Position* pos) {
  std::string r = "<HISTORY>";
  for (size_t i = 0; i < pos->history_.size(); ++i) {
    if (i != 0) {
      r += " ";
    }
    r += pos->history_[i].uci();
  }
  r += "</HISTORY>";
  return r;
}

std::unordered_map<uint64_t, CacheResult> gCache;

constexpr int kSimplePieceValues[7] = {
  // 0, 100, 300, 300, 500, 900, 1000,
  0, 100, 450, 500, 1000, 2000
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

template<Color TURN>
std::pair<Evaluation, Move> qsearch(Position *pos) {
  constexpr Color opposingColor = opposite_color<TURN>();
  constexpr ColoredPiece moverKing = coloredPiece<TURN, Piece::KING>();

  ExtMove moves[kMaxNumMoves];
  ExtMove *end = compute_moves<TURN, MoveGenType::CAPTURES>(*pos, moves);

  if (moves == end) {
    return std::make_pair(gEvaluator.score(*pos), kNullMove);
  }

  const Bitboard theirTargets = compute_my_targets<opposingColor>(*pos);
  const Bitboard theirHanging = ~theirTargets & pos->colorBitboards_[opposingColor];

  for (ExtMove *move = moves; move < end; ++move) {
    move->score = kSimplePieceValues[move->capture] - kSimplePieceValues[move->piece];
    move->score += kSimplePieceValues[move->piece] * ((theirHanging & bb(move->move.to)) > 0);
  }

  std::sort(moves, end, [](ExtMove a, ExtMove b) {
    return a.score > b.score;
  });

  Evaluation selfEval = gEvaluator.score(*pos);
  if (moves[0].score < 0) {
    return std::make_pair(selfEval, kNullMove);
  }

  make_move<TURN>(pos, moves[0].move);

  std::pair<Evaluation, Move> child = qsearch<opposingColor>(pos);

  undo<TURN>(pos);

  if (TURN == Color::WHITE) {
    if (child.first > selfEval) {
      return std::make_pair(child.first, moves[0].move);
    } else {
      return std::make_pair(selfEval, kNullMove);
    }
  } else {
    if (child.first < selfEval) {
      return std::make_pair(child.first, moves[0].move);
    } else {
      return std::make_pair(selfEval, kNullMove);
    }
  }
}

template<Color TURN>
std::pair<Evaluation, Move> search(Position* pos, Depth depth, const Evaluation bestSibling, RecommendedMoves recommendedMoves) {
  constexpr Color opposingColor = opposite_color<TURN>();
  constexpr ColoredPiece moverKing = coloredPiece<TURN, Piece::KING>();

  const bool inCheck = can_enemy_attack<TURN>(*pos, lsb(pos->pieceBitboards_[moverKing]));
  if (inCheck) {
    depth += kDepthScale;
  }

  if (depth <= 0) {
    ++leafCounter;
    return qsearch<TURN>(pos);
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

  if (TURN == Color::WHITE) {
    if (std::popcount(pos->pieceBitboards_[ColoredPiece::WHITE_KING]) == 0) {
      return std::make_pair(kMinEval, kNullMove);
    }
  } else {
    if (std::popcount(pos->pieceBitboards_[ColoredPiece::BLACK_KING]) == 0) {
      return std::make_pair(kMaxEval, kNullMove);
    }
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
    TURN == Color::WHITE ? kMinEval : kMaxEval,
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

    if (can_enemy_attack<TURN>(*pos, lsb(pos->pieceBitboards_[moverKing]))) {
      undo<TURN>(pos);
      continue;
    }
    ++numValidMoves;

    std::pair<Evaluation, Move> a = search<opposingColor>(pos, depth - depthReduction, r.first, recommendationsForChildren);

    if (TURN == Color::WHITE) {
      if (a.first > r.first) {
        r.first = a.first;
        r.second = extMove->move;
        recommendationsForChildren.add(a.second);
      }
    }
    else {
      if (a.first < r.first) {
        r.first = a.first;
        r.second = extMove->move;
        recommendationsForChildren.add(a.second);
      }
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

    if (TURN == Color::WHITE) {
      if (r.first >= bestSibling) {
        return r;
      }
    }
    if (TURN == Color::BLACK) {
      if (r.first <= bestSibling) {
        return r;
      }
    }
  }

  if (numValidMoves == 0) {
    if (inCheck) {
      if (TURN == Color::WHITE) {
        return std::make_pair(kMinEval, kNullMove);
      } else {
        return std::make_pair(kMaxEval, kNullMove);
      }
    } else {
      return std::make_pair(0, kNullMove);
    }
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

  return r;
}

std::pair<Evaluation, Move> search(Position* pos, Depth depth) {
  if (pos->turn_ == Color::WHITE) {
    return search<Color::WHITE>(pos, depth, kMaxEval, RecommendedMoves());
  } else {
    return search<Color::BLACK>(pos, depth, kMinEval, RecommendedMoves());
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

int main(int argc, char *argv[]) {
  signal(SIGSEGV, handler);
  #ifndef NDEBUG
  signal(SIGABRT, handler);
  #endif

  std::vector<std::string> args;
  for (size_t i = 1; i < argc; ++i) {
    args.push_back(argv[i]);
  }

  initialize_geometry();
  initialize_zorbrist();
  // test1();
  // test_moves();

  Position pos = Position::init();
  Depth depth = 99;
  bool loop = false;
  bool silent = false;
  double timeLimitMs = 1000000000.0;
  std::string fenFile;

  while (args.size() > 0) {
    if (args.size() >= 7 && args[0] == "fen") {
      std::vector<std::string> fenVec(args.begin() + 1, args.begin() + 7);
      args = std::vector<std::string>(args.begin() + 7, args.end());

      std::string fen = join(fenVec, " ");
      pos = Position(fen);
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
    } else if (args.size() >= 2 && args[0] == "loop") {
      loop = (args[1][0] == '1');
      args = std::vector<std::string>(args.begin() + 2, args.end());
    } else if (args.size() >= 2 && args[0] == "silent") {
      silent = (args[1] == "1");
      args = std::vector<std::string>(args.begin() + 2, args.end());
    } else if (args.size() >= 1 && args[0] == "evalvec") {
      Evaluation e = gEvaluator.score(pos);
      for (size_t i = 0; i < EF::NUM_EVAL_FEATURES; ++i) {
        std::cout << gEvaluator.features[i] << " " << EFSTR[i] << std::endl;
      }
      return 0;
    } else if (args.size() >= 2 && args[0] == "fens") {
      fenFile = args[1];
      args = std::vector<std::string>(args.begin() + 2, args.end());
    } else {
      std::cout << "Cannot understand arguments" << std::endl;
      return 1;
    }
  }

  if (fenFile.size() > 0) {
    std::ifstream infile(fenFile);
    std::string line;
    size_t a = 0;
    size_t b = 0;
    size_t totalTime = 0;
    while (std::getline(infile, line)) {
      std::vector<std::string> parts = split(line, ':');
      if (parts.size() != 2) {
        throw std::runtime_error("bad fen file");
      }
      std::string fen = parts[0];
      std::string bestMove = parts[1];
      Position pos(fen);

      const time_t t0 = clock();
      std::pair<Evaluation, Move> results;
      for (size_t i = 1; i <= depth; ++i) {
        results = search(&pos, i * kDepthScale);
        if (results.second.uci() == bestMove) {
          break;
        }
      }
      const time_t t1 = clock();

      totalTime += t1 - t0;
      if (results.second.uci() == bestMove) {
        a += 1;
      }
      b += 1;
      if (b % 1000 == 0) {
        std::cout << a << " / " << b << std::endl;
      }
    }
    std::cout << a << " / " << b << std::endl;
    std::cout << totalTime / 1000 << "ms" << std::endl;
    std::cout << "leafCounter = " << leafCounter << std::endl;
    std::cout << "nodeCounter = " << nodeCounter << std::endl;
    return 0;
  }

  if (loop) {
    if (depth <= 0) {
      throw std::runtime_error("invalid depth");
    }
    while (true) {
      gCache.clear();
      std::pair<Evaluation, Move> results;
      time_t tstart = clock();
      for (size_t i = 1; i <= depth; ++i) {
        results = search(&pos, i * kDepthScale);
        if (double(clock() - tstart)/CLOCKS_PER_SEC*1000 > timeLimitMs) {
          break;
        }
      }
      if (results.second == kNullMove || results.first == kMaxEval || results.first == kMinEval) {
        std::cout << std::endl;
        return 0;
      }
      if (pos.turn_ == Color::WHITE) {
        make_move<Color::WHITE>(&pos, results.second);
      } else {
        make_move<Color::BLACK>(&pos, results.second);
      }
      std::cout << results.second << " " << std::flush;
    }
  }

  if (!silent) {
    std::cout << "hash = " << pos.hash_ << std::endl;
  }

  time_t tstart = clock();
  std::pair<Evaluation, Move> results;
  for (size_t i = 1; i <= depth; ++i) {
    time_t t0 = clock();
    results = search(&pos, i * kDepthScale);
    time_t t1 = clock();
    if (!silent) {
      std::cout << i << " : " << results.second << " : " << results.first << " (" << double(t1 - t0)/CLOCKS_PER_SEC << " secs)" << std::endl;
    }
    if (double(clock() - tstart)/CLOCKS_PER_SEC*1000 > timeLimitMs) {
      break;
    }
  }
  time_t tend = clock();
  if (!silent) {
    std::cout << "total time = " << double(tend - tstart)/CLOCKS_PER_SEC << " secs" << std::endl;
    std::cout << "leafCounter = " << leafCounter << std::endl;
    std::cout << "nodeCounter = " << nodeCounter << std::endl;
    std::cout << "eval = " << results.first << std::endl;
  }
  std::cout << "bestMove = " << results.second << std::endl;

  if (!silent) {
    std::cout << "hash = " << pos.hash_ << std::endl;
  }

  auto it = gCache.find(pos.hash_);
  size_t i = 0;
  while (it != gCache.end()) {
    if (++i > 10) {
      break;
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

  return 0;
}