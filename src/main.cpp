 // Production:
 // g++ src/*.cpp -std=c++20 -O3 -DNDEBUG
 // 
 // Debug:
 // g++ src/*.cpp -std=c++20 -std=c++20 -rdynamic -g1

// "./a.out fen r1b1k1nr/pppp1ppp/5q2/2b5/2B2B2/P1N1P3/1PP2PPP/R2QK1NR b KQkq - 0 8 depth 3"
// Why do we sack a queen?

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

std::string history_string(const Position& pos) {
  std::string r = "<HISTORY>";
  for (size_t i = 0; i < pos.history_.size(); ++i) {
    if (i != 0) {
      r += " ";
    }
    r += pos.history_[i].uci();
  }
  r += "</HISTORY>";
  return r;
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

inline Bitboard fatten(Bitboard b) {
  return shift<Direction::WEST>(b) | b | shift<Direction::EAST>(b);
}

enum EF {
  PAWNS,
  KNIGHTS,
  BISHOPS,
  ROOKS,
  QUEENS,

  IN_CHECK,
  KING_ON_BACK_RANK,
  KING_ACTIVE,
  THREATS_NEAR_KING_2,
  THREATS_NEAR_KING_3,

  PASSED_PAWNS,
  ISOLATED_PAWNS,
  DOUBLED_PAWNS,
  DOUBLE_ISOLATED_PAWNS,

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

  BISHOPS_DEVELOPED,
  BISHOP_PAIR,
  BLOCKADED_BISHOPS,
  SCARY_BISHOPS,
  SCARIER_BISHOPS,

  BLOCKADED_ROOKS,
  SCARY_ROOKS,
  INFILTRATING_ROOKS,

  KNIGHTS_DEVELOPED,
  KNIGHT_MAJOR_CAPTURES,
  KNIGHTS_CENTER_16,
  KNIGHTS_CENTER_4,
  KNIGHT_ON_ENEMY_SIDE,

  OUR_HANGING_PAWNS,
  OUR_HANGING_KNIGHTS,
  OUR_HANGING_BISHOPS,
  OUR_HANGING_ROOKS,
  OUR_HANGING_QUEENS,

  THEIR_HANGING_PAWNS,
  THEIR_HANGING_KNIGHTS,
  THEIR_HANGING_BISHOPS,
  THEIR_HANGING_ROOKS,
  THEIR_HANGING_QUEENS,

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
  "IN_CHECK",
  "KING_ON_BACK_RANK",
  "KING_ACTIVE",
  "THREATS_NEAR_KING_2",
  "THREATS_NEAR_KING_3",
  "PASSED_PAWNS",
  "ISOLATED_PAWNS",
  "DOUBLED_PAWNS",
  "DOUBLE_ISOLATED_PAWNS",
  "PAWNS_CENTER_16",
  "PAWNS_CENTER_4",
  "ADVANCED_PASSED_PAWNS_2",
  "ADVANCED_PASSED_PAWNS_3",
  "ADVANCED_PASSED_PAWNS_4",
  "PAWN_MINOR_CAPTURES",
  "PAWN_MAJOR_CAPTURES",
  "PROTECTED_PAWNS",
  "PROTECTED_PASSED_PAWNS",
  "BISHOPS_DEVELOPED",
  "BISHOP_PAIR",
  "BLOCKADED_BISHOPS",
  "SCARY_BISHOPS",
  "SCARIER_BISHOPS",
  "BLOCKADED_ROOKS",
  "SCARY_ROOKS",
  "INFILTRATING_ROOKS",
  "KNIGHTS_DEVELOPED",
  "KNIGHT_MAJOR_CAPTURES",
  "KNIGHTS_CENTER_16",
  "KNIGHTS_CENTER_4",
  "KNIGHT_ON_ENEMY_SIDE",
  "OUR_HANGING_PAWNS",
  "OUR_HANGING_KNIGHTS",
  "OUR_HANGING_BISHOPS",
  "OUR_HANGING_ROOKS",
  "OUR_HANGING_QUEENS",
  "THEIR_HANGING_PAWNS",
  "THEIR_HANGING_KNIGHTS",
  "THEIR_HANGING_BISHOPS",
  "THEIR_HANGING_ROOKS",
  "THEIR_HANGING_QUEENS",
  "TIME",
  "NUM_TARGET_SQUARES",
};

// 5584 / 11387
// 2677ms
// leafCounter = 1579099
// nodeCounter = 323425

struct Evaluator {
  Evaluator() {}

  template<Color US>
  Evaluation score(const Position& pos) {
    assert(US == pos.turn_);
    constexpr Color THEM = opposite_color<US>();
    if (std::popcount(pos.pieceBitboards_[coloredPiece<US, Piece::KING>()]) == 0) {
      return kMinEval;
    }
    if (std::popcount(pos.pieceBitboards_[coloredPiece<THEM, Piece::KING>()]) == 0) {
      return kMaxEval;
    }

    const Square ourKingSq = lsb(pos.pieceBitboards_[coloredPiece<US, Piece::KING>()]);
    const Square theirKingSq = lsb(pos.pieceBitboards_[coloredPiece<THEM, Piece::KING>()]);

    const Bitboard ourPawns = pos.pieceBitboards_[coloredPiece<US, Piece::PAWN>()];
    const Bitboard ourKnights = pos.pieceBitboards_[coloredPiece<US, Piece::KNIGHT>()];
    const Bitboard ourBishops = pos.pieceBitboards_[coloredPiece<US, Piece::BISHOP>()];
    const Bitboard ourRooks = pos.pieceBitboards_[coloredPiece<US, Piece::ROOK>()];
    const Bitboard ourQueens = pos.pieceBitboards_[coloredPiece<US, Piece::QUEEN>()];
    const Bitboard ourKings = pos.pieceBitboards_[coloredPiece<US, Piece::KING>()];

    const Bitboard theirPawns = pos.pieceBitboards_[coloredPiece<THEM, Piece::PAWN>()];
    const Bitboard theirKnights = pos.pieceBitboards_[coloredPiece<THEM, Piece::KNIGHT>()];
    const Bitboard theirBishops = pos.pieceBitboards_[coloredPiece<THEM, Piece::BISHOP>()];
    const Bitboard theirRooks = pos.pieceBitboards_[coloredPiece<THEM, Piece::ROOK>()];
    const Bitboard theirQueens = pos.pieceBitboards_[coloredPiece<THEM, Piece::QUEEN>()];
    const Bitboard theirKings = pos.pieceBitboards_[coloredPiece<THEM, Piece::KING>()];

    const Bitboard royalUs = ourQueens | ourKings;
    const Bitboard royalThem = theirQueens | theirKings;
    const Bitboard majorUs = ourRooks | royalUs;
    const Bitboard majorThem = theirRooks | royalThem;
    const Bitboard minorUs = ourKnights | ourBishops;
    const Bitboard minorThem = theirKnights | theirBishops;
    const Bitboard usPieces = majorUs | minorUs;
    const Bitboard themPieces = majorThem | minorThem;

    const Bitboard ourPawnTargets = compute_pawn_targets<US>(pos);
    const Bitboard theirPawnTargets = compute_pawn_targets<THEM>(pos);
    const Bitboard ourKnightTargets = compute_knight_targets<US>(pos);
    const Bitboard theirKnightTargets = compute_knight_targets<THEM>(pos);
    const Bitboard usBishopTargets = compute_bishoplike_targets<US>(pos, ourBishops);
    const Bitboard themBishopTargets = compute_bishoplike_targets<THEM>(pos, theirBishops);
    const Bitboard usRookTargets = compute_rooklike_targets<US>(pos, ourRooks);
    const Bitboard themRookTargets = compute_rooklike_targets<THEM>(pos, theirRooks);
    const Bitboard usQueenTargets = compute_bishoplike_targets<US>(pos, ourQueens) | compute_rooklike_targets<US>(pos, ourQueens);
    const Bitboard themQueenTargets = compute_bishoplike_targets<THEM>(pos, theirQueens) | compute_rooklike_targets<THEM>(pos, theirQueens);

    constexpr Bitboard kOurSide = (US == Color::WHITE ? kWhiteSide : kBlackSide);
    constexpr Bitboard kTheirSide = (US == Color::WHITE ? kBlackSide : kWhiteSide);
    constexpr Direction kForward = (US == Color::WHITE ? Direction::NORTH : Direction::SOUTH);
    constexpr Direction kBackward = opposite_dir(kForward);

    const Bitboard usTargets = ourPawnTargets | ourKnightTargets | usBishopTargets | usRookTargets | usQueenTargets;
    const Bitboard themTargets = theirPawnTargets | theirKnightTargets | themBishopTargets | themRookTargets | themQueenTargets;

    features[EF::PAWNS] = std::popcount(ourPawns) - std::popcount(theirPawns);
    features[EF::KNIGHTS] = std::popcount(ourKnights) - std::popcount(theirKnights);
    features[EF::BISHOPS] = std::popcount(ourBishops) - std::popcount(theirBishops);
    features[EF::ROOKS] = std::popcount(ourRooks) - std::popcount(theirRooks);
    features[EF::QUEENS] = std::popcount(ourQueens) - std::popcount(theirQueens);

    // Note that TURN (the side to move) often gets larger bonuses since they can take advantage of threats better.

    bool isUsInCheck = can_enemy_attack<US>(pos, ourKingSq);
    bool isThemInCheck = can_enemy_attack<THEM>(pos, theirKingSq);
    // Note: in a rigorous engine isThemInCheck would always be false on your move. For this engine,
    // it means they have moved into check (e.g. moving a pinned piece). Unfortunately we cannot
    // immediately return kMaxEval since it's possible they had had legal moves (and were in stalemate).
    features[EF::IN_CHECK] = isUsInCheck - isThemInCheck;
    if (US == Color::WHITE) {
      features[EF::KING_ON_BACK_RANK] = (ourKingSq / 8 == 7) - (theirKingSq / 8 == 0);
      features[EF::KING_ACTIVE] = (ourKingSq / 8 < 5) - (theirKingSq / 8 > 2);
    } else {
      features[EF::KING_ON_BACK_RANK] = (ourKingSq / 8 == 0) - (theirKingSq / 8 == 7);
      features[EF::KING_ACTIVE] = (ourKingSq / 8 > 2) - (theirKingSq / 8 < 5);
    }
    features[EF::THREATS_NEAR_KING_2] = std::popcount(kNearby[2][ourKingSq] & themTargets & ~usTargets) - std::popcount(kNearby[2][theirKingSq] & usTargets & ~themTargets);
    features[EF::THREATS_NEAR_KING_3] = std::popcount(kNearby[3][ourKingSq] & themTargets & ~usTargets) - std::popcount(kNearby[2][theirKingSq] & usTargets & ~themTargets);

    // Pawns
    const Bitboard theirBlockadedPawns = shift<kForward>(ourPawns) & theirPawns;
    const Bitboard outBlockadedPawns = shift<kBackward>(theirPawns) & ourPawns;
    const Bitboard ourProtectedPawns = ourPawns & ourPawnTargets;
    const Bitboard theirProtectedPawns = theirPawns & theirPawnTargets;
    {
      Bitboard filledUs, filledThem;
      Bitboard filesWithOurPawns, filesWithTheirPawns;
      if (US == Color::WHITE) {
        filledUs = northFill(ourPawns);
        filledThem = southFill(theirPawns);
        filesWithOurPawns = southFill(filledUs);
        filesWithTheirPawns = northFill(filledThem);
      } else {
        filledUs = southFill(ourPawns);
        filledThem = northFill(theirPawns);
        filesWithOurPawns = northFill(filledUs);
        filesWithTheirPawns = southFill(filledThem);
      }
      const Bitboard fatUsPawns = fatten(filledUs);
      const Bitboard fatThemPawns = fatten(filledThem);
      const Bitboard filesWithoutUsPawns = ~filesWithOurPawns;
      const Bitboard filesWithoutThemPawns = ~filesWithTheirPawns;
      const Bitboard passedUsPawns = ourPawns & ~shift<kBackward>(fatten(filledThem));
      const Bitboard passedThemPawns = theirPawns & ~shift<kForward>(fatten(filledUs));
      const Bitboard isolatedUsPawns = ourPawns & ~shift<Direction::WEST>(filesWithOurPawns) & ~shift<Direction::EAST>(filesWithOurPawns);
      const Bitboard isolatedThemPawns = theirPawns & ~shift<Direction::WEST>(filesWithTheirPawns) & ~shift<Direction::EAST>(filesWithTheirPawns);
      const Bitboard doubledUsPawns = ourPawns & shift<kForward>(filledUs);
      const Bitboard doubledThemPawns = theirPawns & shift<kBackward>(filledThem);

      constexpr Bitboard kRookPawns = kFiles[0] | kFiles[7];

      features[EF::PAWNS_CENTER_16] = std::popcount(ourPawns & kCenter16) - std::popcount(theirPawns & kCenter16);
      features[EF::PAWNS_CENTER_16] = std::popcount(ourPawns & kCenter16) - std::popcount(theirPawns & kCenter16);
      features[EF::PAWNS_CENTER_4] = std::popcount(ourPawns & kCenter4) - std::popcount(theirPawns & kCenter4);
      features[EF::PASSED_PAWNS] = std::popcount(passedUsPawns) * 2 - std::popcount(passedThemPawns);
      features[EF::ISOLATED_PAWNS] = std::popcount(isolatedUsPawns) - std::popcount(isolatedThemPawns);
      features[EF::DOUBLED_PAWNS] = std::popcount(doubledUsPawns) - std::popcount(doubledThemPawns);
      features[EF::DOUBLE_ISOLATED_PAWNS] = std::popcount(doubledUsPawns & isolatedUsPawns) - std::popcount(doubledThemPawns & isolatedThemPawns);

      if (US == Color::WHITE) {
        features[EF::ADVANCED_PASSED_PAWNS_2] = std::popcount(passedUsPawns & kRanks[1]) * 2 - std::popcount(passedThemPawns & kRanks[6]);
        features[EF::ADVANCED_PASSED_PAWNS_3] = std::popcount(passedUsPawns & kRanks[2]) * 2 - std::popcount(passedThemPawns & kRanks[5]);
        features[EF::ADVANCED_PASSED_PAWNS_4] = std::popcount(passedUsPawns & kRanks[3]) * 2 - std::popcount(passedThemPawns & kRanks[4]);
      } else {
        features[EF::ADVANCED_PASSED_PAWNS_2] = std::popcount(passedUsPawns & kRanks[6]) * 2 - std::popcount(passedThemPawns & kRanks[1]);
        features[EF::ADVANCED_PASSED_PAWNS_3] = std::popcount(passedUsPawns & kRanks[5]) * 2 - std::popcount(passedThemPawns & kRanks[2]);
        features[EF::ADVANCED_PASSED_PAWNS_4] = std::popcount(passedUsPawns & kRanks[4]) * 2 - std::popcount(passedThemPawns & kRanks[3]);
      }

      features[EF::PAWN_MINOR_CAPTURES] = std::popcount(ourPawnTargets & minorThem) * 3 - std::popcount(theirPawnTargets & minorUs);
      features[EF::PAWN_MAJOR_CAPTURES] = std::popcount(ourPawnTargets & majorThem) * 3 - std::popcount(theirPawnTargets & majorUs);
      features[EF::PROTECTED_PAWNS] = std::popcount(ourPawns & ourPawnTargets) - std::popcount(theirPawns & theirPawnTargets);
      features[EF::PROTECTED_PASSED_PAWNS] = std::popcount(passedUsPawns & ourPawnTargets) * 2 - std::popcount(passedThemPawns & theirPawnTargets);
    }

    {  // Bishops
      const Bitboard ourBishopTargetsIgnoringNonBlockades = compute_bishoplike_targets<US>(pos, ourBishops, outBlockadedPawns);
      const Bitboard theirBishopTargetsIgnoringNonBlockades = compute_bishoplike_targets<THEM>(pos, theirBishops, theirBlockadedPawns);
      if (US == Color::WHITE) {
        features[EF::BISHOPS_DEVELOPED] = std::popcount(theirBishops & (bb( 2) | bb( 5))) - std::popcount(ourBishops & (bb(58) | bb(61)));
      } else {
        features[EF::BISHOPS_DEVELOPED] = std::popcount(theirBishops & (bb(58) | bb(61))) - std::popcount(ourBishops & (bb( 2) | bb( 5)));
      }
      features[EF::BISHOP_PAIR] = (std::popcount(ourBishops) >= 2) - (std::popcount(theirBishops) >= 2);
      features[EF::BLOCKADED_BISHOPS] = std::popcount(ourBishopTargetsIgnoringNonBlockades & (outBlockadedPawns | theirProtectedPawns)) - std::popcount(theirBishopTargetsIgnoringNonBlockades & (theirBlockadedPawns | ourProtectedPawns));
      features[EF::SCARY_BISHOPS] = std::popcount(ourBishopTargetsIgnoringNonBlockades & majorThem) - std::popcount(theirBishopTargetsIgnoringNonBlockades & majorUs);
      features[EF::SCARIER_BISHOPS] = std::popcount(ourBishopTargetsIgnoringNonBlockades & royalThem) - std::popcount(theirBishopTargetsIgnoringNonBlockades & royalUs);

      // const Bitboard ourPawnsOnUs = kUsSquares & ourPawns;
      // const Bitboard ourPawnsOnThem = kThemSquares & ourPawns;
      // const Bitboard theirPawnsOnUs = kUsSquares & theirPawns;
      // const Bitboard theirPawnsOnThem = kThemSquares & theirPawns;
    }

    constexpr Bitboard kOurBackRanks = (US == Color::WHITE ? kRanks[6] | kRanks[7] : kRanks[1] | kRanks[0]);
    constexpr Bitboard kTheirBackRanks = (US == Color::WHITE ? kRanks[1] | kRanks[0] : kRanks[6] | kRanks[7]);

    {  // Rooks
      features[EF::BLOCKADED_ROOKS] = std::popcount(usRookTargets & ourPawns) - std::popcount(themRookTargets & theirPawns);
      features[EF::SCARY_ROOKS] = std::popcount(usRookTargets & royalThem) - std::popcount(themRookTargets & royalUs);
      features[EF::INFILTRATING_ROOKS] = std::popcount(ourRooks & kTheirBackRanks) - std::popcount(theirRooks & kOurBackRanks);
    }

    {  // Knights
      if (US == Color::WHITE) {
        features[EF::KNIGHTS_DEVELOPED] = std::popcount(theirKnights & (bb( 1) | bb( 6))) - std::popcount(ourKnights & (bb(57) | bb(62)));
      } else {
        features[EF::KNIGHTS_DEVELOPED] = std::popcount(theirKnights & (bb(57) | bb(62))) - std::popcount(ourKnights & (bb( 1) | bb( 6)));
      }
      features[EF::KNIGHT_MAJOR_CAPTURES] = std::popcount(compute_knight_targets<US>(pos) & majorThem) * 2 - std::popcount(compute_knight_targets<THEM>(pos) & majorUs);
      features[EF::KNIGHTS_CENTER_16] = std::popcount(ourKnights & kCenter16) - std::popcount(theirKnights & kCenter16);
      features[EF::KNIGHTS_CENTER_4] = std::popcount(ourKnights & kCenter4) - std::popcount(theirKnights & kCenter4);
      features[EF::KNIGHT_ON_ENEMY_SIDE] = std::popcount(ourKnights & kTheirSide) * 2 - std::popcount(theirKnights & kOurSide);
    }

    {
      // Hanging pieces are more valuable if it is your turn since they're literally free material,
      // as opposed to threats. Also is a very useful heuristic so that leaf nodes don't sack a rook
      // for a pawn.
      const Bitboard usHanging = themTargets & ~usTargets & pos.colorBitboards_[US];
      const Bitboard themHanging = usTargets & ~themTargets & pos.colorBitboards_[THEM];
      features[EF::OUR_HANGING_PAWNS] = std::popcount(ourPawns & usHanging);
      features[EF::THEIR_HANGING_PAWNS] = std::popcount(theirPawns & themHanging);
      features[EF::OUR_HANGING_KNIGHTS] = std::popcount(ourKnights & usHanging);
      features[EF::THEIR_HANGING_KNIGHTS] = std::popcount(theirKnights & themHanging);
      features[EF::OUR_HANGING_BISHOPS] = std::popcount(ourBishops & usHanging);
      features[EF::THEIR_HANGING_BISHOPS] = std::popcount(theirBishops & themHanging);
      features[EF::OUR_HANGING_ROOKS] = std::popcount(ourRooks & usHanging);
      features[EF::THEIR_HANGING_ROOKS] = std::popcount(theirRooks & themHanging);
      features[EF::OUR_HANGING_QUEENS] = std::popcount(ourQueens & usHanging);
      features[EF::THEIR_HANGING_QUEENS] = std::popcount(theirQueens & themHanging);

      features[EF::NUM_TARGET_SQUARES] = std::popcount(usTargets) * 2 - std::popcount(themTargets);

    }

    const int16_t usPiecesRemaining = std::popcount(pos.colorBitboards_[US] & ~ourPawns) + std::popcount(ourQueens) - 1;
    const int16_t themPiecesRemaining = std::popcount(pos.colorBitboards_[THEM] & ~theirPawns) + std::popcount(theirQueens) - 1;

    // Use larger integer to make arithmetic safe.
    const int32_t earliness = (usPiecesRemaining + themPiecesRemaining);
    const int32_t early = this->early<US>(pos);
    const int32_t late = this->late<US>(pos);

    features[EF::TIME] = earliness;
    
    return (early * earliness + late * (16 - earliness)) / 16;
  }

  template<Color US>
  Evaluation early(const Position& pos) const {
    Evaluation r = 20;  // Bonus for being your turn.
    r += features[EF::PAWNS] * 100;
    r += features[EF::KNIGHTS] * 450;
    r += features[EF::BISHOPS] * 465;
    r += features[EF::ROOKS] * 365;
    r += features[EF::QUEENS] * 1093;
    r += features[EF::IN_CHECK] * -300;

    r += features[EF::KING_ON_BACK_RANK] * 50;
    r += features[EF::KING_ACTIVE] * -100;
    r += features[EF::THREATS_NEAR_KING_2] * -2;
    r += features[EF::THREATS_NEAR_KING_3] * -2;

    // r += features[EF::PASSED_PAWNS] * 0;
    r += features[EF::ISOLATED_PAWNS] * -30;
    r += features[EF::DOUBLED_PAWNS] * -20;
    r += features[EF::DOUBLE_ISOLATED_PAWNS] * -20;
    r += features[EF::PAWNS_CENTER_16] * 10;
    r += features[EF::PAWNS_CENTER_4] * 58;
    // r += features[EF::ADVANCED_PASSED_PAWNS_1] * 0;
    // r += features[EF::ADVANCED_PASSED_PAWNS_2] * 0;
    // r += features[EF::ADVANCED_PASSED_PAWNS_3] * 0;
    // r += features[EF::ADVANCED_PASSED_PAWNS_4] * 0;
    r += features[EF::PAWN_MINOR_CAPTURES] * 50;
    r += features[EF::PAWN_MAJOR_CAPTURES] * 50;
    r += features[EF::PROTECTED_PAWNS] * 10;

    r += features[EF::BISHOPS_DEVELOPED] * 10;
    // r += features[EF::BISHOP_PAIR] * 20;
    r += features[EF::BLOCKADED_BISHOPS] * -10;
    r += features[EF::SCARY_BISHOPS] * 10;
    r += features[EF::SCARIER_BISHOPS] * 30;

    r += features[EF::BLOCKADED_ROOKS] * -10;
    r += features[EF::SCARY_ROOKS] * 10;
    r += features[EF::INFILTRATING_ROOKS] * 20;

    r += features[EF::KNIGHTS_DEVELOPED] * 20;
    r += features[EF::KNIGHT_MAJOR_CAPTURES] * 40;
    r += features[EF::KNIGHTS_CENTER_16] * 50;
    r += features[EF::KNIGHTS_CENTER_4] * 0;
    r += features[EF::KNIGHT_ON_ENEMY_SIDE] * 30;

    const Evaluation wScaleFactor = 10;
    const Evaluation bScaleFactor = -1;
    const Evaluation denominator = 10;

    r += features[EF::OUR_HANGING_PAWNS] * 100 * bScaleFactor / denominator;
    r += features[EF::OUR_HANGING_KNIGHTS] * 300 * bScaleFactor / denominator;
    r += features[EF::OUR_HANGING_BISHOPS] * 300 * bScaleFactor / denominator;
    r += features[EF::OUR_HANGING_ROOKS] * 500 * bScaleFactor / denominator;
    r += features[EF::OUR_HANGING_QUEENS] * 900 * bScaleFactor / denominator;

    r += features[EF::THEIR_HANGING_PAWNS] * 100 * wScaleFactor / denominator;
    r += features[EF::THEIR_HANGING_KNIGHTS] * 300 * wScaleFactor / denominator;
    r += features[EF::THEIR_HANGING_BISHOPS] * 300 * wScaleFactor / denominator;
    r += features[EF::THEIR_HANGING_ROOKS] * 500 * wScaleFactor / denominator;
    r += features[EF::THEIR_HANGING_QUEENS] * 900 * wScaleFactor / denominator;

    r += features[EF::NUM_TARGET_SQUARES] * 4;

    return r;
  }

  template<Color US>
  Evaluation late(const Position& pos) const {
    constexpr Color THEM = opposite_color<US>();
    Evaluation r = 0;
    r += features[EF::PAWNS] * 100;
    r += features[EF::KNIGHTS] * 436;
    r += features[EF::BISHOPS] * 539;
    r += features[EF::ROOKS] * 1561;
    r += features[EF::QUEENS] * 3101;
    r += features[EF::IN_CHECK] * -200;

    r += features[EF::KING_ON_BACK_RANK] * -50;
    r += features[EF::KING_ACTIVE] * 50;

    r += features[EF::PASSED_PAWNS] * 50;
    r += features[EF::DOUBLED_PAWNS] * -20;
    r += features[EF::ADVANCED_PASSED_PAWNS_2] * 250;
    r += features[EF::ADVANCED_PASSED_PAWNS_3] * 100;
    r += features[EF::ADVANCED_PASSED_PAWNS_4] * 50;
    r += features[EF::PAWN_MINOR_CAPTURES] * 50;
    r += features[EF::PAWN_MAJOR_CAPTURES] * 50;
    // r += features[EF::PROTECTED_PAWNS] * 10;
    r += features[EF::PROTECTED_PASSED_PAWNS] * 10;

    r += features[EF::BLOCKADED_BISHOPS] * -10;
    r += features[EF::SCARY_BISHOPS] * 30;

    r += features[EF::BLOCKADED_ROOKS] * -20;
    r += features[EF::SCARY_ROOKS] * 10;
    r += features[EF::INFILTRATING_ROOKS] * 10;

    const Square theirKingSq = lsb(pos.colorBitboards_[THEM]);
    const Square ourKingSq = lsb(pos.colorBitboards_[US]);
    const int wx = ourKingSq % 8;
    const int wy = ourKingSq / 8;
    const int bx = theirKingSq % 8;
    const int by = theirKingSq / 8;
    const int dist = std::max(std::abs(wx - bx), std::abs(wy - by));
    // If them's king is the last them piece alive, us wants to drive it near the edge.
    // If them's king is the last them piece alive, us wants to be near him.
    r -= kDistToCorner[theirKingSq] * 2 * (std::popcount(pos.colorBitboards_[THEM]) == 1);
    r -= (std::popcount(pos.colorBitboards_[THEM]) == 1) * dist * 10;
    r += kDistToCorner[ourKingSq] * 2 * (std::popcount(pos.colorBitboards_[US]) == 1);
    r += (std::popcount(pos.colorBitboards_[US]) == 1) * dist * 10;

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

template<Color TURN>
std::pair<Evaluation, Move> qsearch(Position *pos, int32_t depth) {
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

  Evaluation selfEval = gEvaluator.score<TURN>(*pos);
  if (moves[0].score < 0) {
    return std::make_pair(selfEval, kNullMove);
  }

  std::pair<Evaluation, Move> bestChild(Evaluation(kMinEval), kNullMove);
  for (ExtMove *move = moves; move < end; ++move) {
    make_move<TURN>(pos, moves[0].move);

    std::pair<Evaluation, Move> child = qsearch<opposingColor>(pos, depth + 1);
    child.first *= -1;

    if (child.first > bestChild.first) {
      bestChild = child;
    }

    undo<TURN>(pos);

    // Only looking at the best move seems to work fine.
    break;
  }

  if (bestChild.first > selfEval) {
    return std::make_pair(bestChild.first, moves[0].move);
  } else {
    return std::make_pair(selfEval, kNullMove);
  }
}

template<Color TURN>
std::pair<Evaluation, Move> search(Position* pos, Depth depth, const Evaluation bestSibling, RecommendedMoves recommendedMoves) {
  constexpr Color opposingColor = opposite_color<TURN>();
  constexpr ColoredPiece moverKing = coloredPiece<TURN, Piece::KING>();

  if (std::popcount(pos->pieceBitboards_[coloredPiece<TURN, Piece::KING>()]) == 0) {
    return std::make_pair(kMinEval, kNullMove);
  }

  const bool inCheck = can_enemy_attack<TURN>(*pos, lsb(pos->pieceBitboards_[moverKing]));

  if (depth <= 0) {
    ++leafCounter;
    std::pair<Evaluation, Move> r = qsearch<TURN>(pos, 0);
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

    std::pair<Evaluation, Move> a = search<opposingColor>(pos, depth - depthReduction, -r.first, recommendationsForChildren);
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

    if (r.first >= bestSibling) {
      return r;
    }
  }

  if (numValidMoves == 0) {
    r.first = inCheck ? kMinEval + 1 : 0;
    r.second = kNullMove;
  }

  // This technically improves the engine but presumably makes optimizing
  // it really annoying to think about, so commenting out for now.
  // if (it != gCache.end()) {
  //   r.first = float(r.first) * 1.1 - float(it->second.eval) * 0.1;
  // }

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
    return search<Color::WHITE>(pos, depth, kMaxEval, RecommendedMoves());
  } else {
    std::pair<Evaluation, Move> r = search<Color::BLACK>(pos, depth, kMaxEval, RecommendedMoves());
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
void print_feature_vec(Position *pos) {
  std::pair<Evaluation, Move> r = qsearch<TURN>(pos, 0);
  if (r.second != kNullMove) {
    make_move<TURN>(pos, r.second);
    print_feature_vec<opposite_color<TURN>()>(pos);
    undo<TURN>(pos);
    return;
  }

  Evaluation e;
  if (pos->turn_ == Color::WHITE) {
    e = gEvaluator.score<Color::WHITE>(*pos);
  } else {
    e = -gEvaluator.score<Color::BLACK>(*pos);
  }
  std::cout << pos->fen() << std::endl;
  for (size_t i = 0; i < EF::NUM_EVAL_FEATURES; ++i) {
    std::cout << gEvaluator.features[i] << " " << EFSTR[i] << std::endl;
  }
  std::cout << "score: " << e << std::endl;
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
      if (pos.turn_ == Color::WHITE) {
        print_feature_vec<Color::WHITE>(&pos);
      } else {
        print_feature_vec<Color::BLACK>(&pos);
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
  for (size_t i = 0; i <= depth; ++i) {
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

  return 0;
}