#ifndef EVALUATOR_H
#define EVALUATOR_H

#include <cstdint>

#include <algorithm>
#include <fstream>
#include <string>

#include "geometry.h"
#include "kpvk.h"
#include "utils.h"
#include "Threats.h"
#include "Position.h"
#include "movegen.h"
#include "piece_maps.h"
#include "PawnAnalysis.h"

namespace ChessEngine {

// Arbitrary constant that our "special_boosts" will never accidentally return.
static const Evaluation kKnownDraw = kMinEval + 10;

/**
 * Engine is bad at
 * 1) knowing when trapped pieces (e.g. knights in corners) aren't long for this world
 * 2) knowing when very extended/isolated pawns will probably be captured in endgames
 *    (bc one king is close to it and the other is far)
 * 3) more generally, understanding what squares the king wants to get to, and rewarding
 *    the king for being close to them (namely promotion squares and isolated pawns).
 * 4) pawn storms
 */

enum EF {
  OUR_PAWNS,
  OUR_KNIGHTS,
  OUR_BISHOPS,
  OUR_ROOKS,
  OUR_QUEENS,
  THEIR_PAWNS,
  THEIR_KNIGHTS,
  THEIR_BISHOPS,
  THEIR_ROOKS,
  THEIR_QUEENS,

  IN_CHECK,
  KING_ON_BACK_RANK,

  KING_ON_CENTER_FILE,  // +0.2578 ± 0.0532
  KING_ACTIVE,
  THREATS_NEAR_KING_2,
  THREATS_NEAR_KING_3,
  PASSED_PAWNS,

  ISOLATED_PAWNS,
  DOUBLED_PAWNS,
  DOUBLE_ISOLATED_PAWNS,
  PAWNS_CENTER_16,
  PAWNS_CENTER_4,
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
  LONELY_KING_IN_CENTER,

  LONELY_KING_AWAY_FROM_ENEMY_KING,
  TIME,
  KPVK_OPPOSITION,
  SQUARE_RULE,
  ADVANCED_PAWNS_1,
  ADVANCED_PAWNS_2,
  OPEN_ROOKS,
  ROOKS_ON_THEIR_SIDE,

  KING_IN_FRONT_OF_PASSED_PAWN,
  KING_IN_FRONT_OF_PASSED_PAWN2,
  OUR_MATERIAL_THREATS,
  THEIR_MATERIAL_THREATS,
  LONELY_KING_ON_EDGE,

  OUTPOSTED_KNIGHTS,
  OUTPOSTED_BISHOPS,
  PAWN_MOVES,
  KNIGHT_MOVES,
  BISHOP_MOVES,
  ROOK_MOVES,

  QUEEN_MOVES,
  PAWN_MOVES_ON_THEIR_SIDE,
  KNIGHT_MOVES_ON_THEIR_SIDE,
  BISHOP_MOVES_ON_THEIR_SIDE,
  ROOK_MOVES_ON_THEIR_SIDE,
  QUEEN_MOVES_ON_THEIR_SIDE,

  KING_HOME_QUALITY,
  BISHOPS_BLOCKING_KNIGHTS,
  OUR_HANGING_PAWNS_2,
  OUR_HANGING_KNIGHTS_2,
  OUR_HANGING_BISHOPS_2,
  OUR_HANGING_ROOKS_2,

  OUR_HANGING_QUEENS_2,
  THEIR_HANGING_PAWNS_2,
  THEIR_HANGING_KNIGHTS_2,
  THEIR_HANGING_BISHOPS_2,
  THEIR_HANGING_ROOKS_2,
  THEIR_HANGING_QUEENS_2,

  QUEEN_THREATS_NEAR_KING,
  MISSING_FIANCHETTO_BISHOP,

  NUM_BAD_SQUARES_FOR_PAWNS,
  NUM_BAD_SQUARES_FOR_MINORS,
  NUM_BAD_SQUARES_FOR_ROOKS,
  NUM_BAD_SQUARES_FOR_QUEENS,
  IN_TRIVIAL_CHECK,
  IN_DOUBLE_CHECK,

  THREATS_NEAR_OUR_KING,
  THREATS_NEAR_THEIR_KING,
  NUM_PIECES_HARRASSABLE_BY_PAWNS,  // 0.0322 ± 0.0174 (96.8%)

  PAWN_CHECKS,
  KNIGHT_CHECKS,
  BISHOP_CHECKS,
  ROOK_CHECKS,
  QUEEN_CHECKS,

  BACK_RANK_MATE_THREAT_AGAINST_US,
  BACK_RANK_MATE_THREAT_AGAINST_THEM,

  OUR_KING_HAS_0_ESCAPE_SQUARES,
  THEIR_KING_HAS_0_ESCAPE_SQUARES,
  OUR_KING_HAS_1_ESCAPE_SQUARES,
  THEIR_KING_HAS_1_ESCAPE_SQUARES,
  OUR_KING_HAS_2_ESCAPE_SQUARES,
  THEIR_KING_HAS_2_ESCAPE_SQUARES,

  OPPOSITE_SIDE_KINGS_PAWN_STORM,
  IN_CHECK_AND_OUR_HANGING_QUEENS,
  PROMOTABLE_PAWN,
  PINNED_PIECES,

  NUM_EVAL_FEATURES,
};

constexpr Bitboard kWhiteKingCorner = bb(Square::H1) | bb(Square::H2) | bb(Square::G1) | bb(Square::G2) | bb(Square::F1);
constexpr Bitboard kWhiteQueenCorner = bb(Square::A1) | bb(Square::A2) | bb(Square::B1) | bb(Square::B2) | bb(Square::C1);
constexpr Bitboard kBlackKingCorner = bb(Square::H8) | bb(Square::H7) | bb(Square::G8) | bb(Square::G7) | bb(Square::F8);
constexpr Bitboard kBlackQueenCorner = bb(Square::A8) | bb(Square::A7) | bb(Square::B8) | bb(Square::B7) | bb(Square::C8);

std::string EFSTR[] = {
  "OUR_PAWNS",
  "OUR_KNIGHTS",
  "OUR_BISHOPS",
  "OUR_ROOKS",
  "OUR_QUEENS",
  "THEIR_PAWNS",
  "THEIR_KNIGHTS",
  "THEIR_BISHOPS",
  "THEIR_ROOKS",
  "THEIR_QUEENS",
  "IN_CHECK",
  "KING_ON_BACK_RANK",
  "KING_ON_CENTER_FILE",
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
  "LONELY_KING_IN_CENTER",
  "LONELY_KING_AWAY_FROM_ENEMY_KING",
  "TIME",
  "KPVK_OPPOSITION",
  "SQUARE_RULE",
  "ADVANCED_PAWNS_1",
  "ADVANCED_PAWNS_2",
  "OPEN_ROOKS",
  "ROOKS_ON_THEIR_SIDE",
  "KING_IN_FRONT_OF_PASSED_PAWN",
  "KING_IN_FRONT_OF_PASSED_PAWN2",
  "OUR_MATERIAL_THREATS",
  "THEIR_MATERIAL_THREATS",
  "LONELY_KING_ON_EDGE",
  "OUTPOSTED_KNIGHTS",
  "OUTPOSTED_BISHOPS",
  "PAWN_MOVES",
  "KNIGHT_MOVES",
  "BISHOP_MOVES",
  "ROOK_MOVES",
  "QUEEN_MOVES",
  "PAWN_MOVES_ON_THEIR_SIDE",
  "KNIGHT_MOVES_ON_THEIR_SIDE",
  "BISHOP_MOVES_ON_THEIR_SIDE",
  "ROOK_MOVES_ON_THEIR_SIDE",
  "QUEEN_MOVES_ON_THEIR_SIDE",
  "KING_HOME_QUALITY",
  "BISHOPS_BLOCKING_KNIGHTS",
  "OUR_HANGING_PAWNS_2",
  "OUR_HANGING_KNIGHTS_2",
  "OUR_HANGING_BISHOPS_2",
  "OUR_HANGING_ROOKS_2",
  "OUR_HANGING_QUEENS_2",
  "THEIR_HANGING_PAWNS_2",
  "THEIR_HANGING_KNIGHTS_2",
  "THEIR_HANGING_BISHOPS_2",
  "THEIR_HANGING_ROOKS_2",
  "THEIR_HANGING_QUEENS_2",
  "QUEEN_THREATS_NEAR_KING",
  "MISSING_FIANCHETTO_BISHOP",
  "NUM_BAD_SQUARES_FOR_PAWNS",
  "NUM_BAD_SQUARES_FOR_MINORS",
  "NUM_BAD_SQUARES_FOR_ROOKS",
  "NUM_BAD_SQUARES_FOR_QUEENS",
  "IN_TRIVIAL_CHECK",
  "IN_DOUBLE_CHECK",
  "THREATS_NEAR_OUR_KING",
  "THREATS_NEAR_THEIR_KING",
  "NUM_PIECES_HARRASSABLE_BY_PAWNS",
  "PAWN_CHECKS",
  "KNIGHT_CHECKS",
  "BISHOP_CHECKS",
  "ROOK_CHECKS",
  "QUEEN_CHECKS",
  "BACK_RANK_MATE_THREAT_AGAINST_US",
  "BACK_RANK_MATE_THREAT_AGAINST_THEM",
  "OUR_KING_HAS_0_ESCAPE_SQUARES",
  "THEIR_KING_HAS_0_ESCAPE_SQUARES",
  "OUR_KING_HAS_1_ESCAPE_SQUARES",
  "THEIR_KING_HAS_1_ESCAPE_SQUARES",
  "OUR_KING_HAS_2_ESCAPE_SQUARES",
  "THEIR_KING_HAS_2_ESCAPE_SQUARES",
  "OPPOSITE_SIDE_KINGS_PAWN_STORM",
  "IN_CHECK_AND_OUR_HANING_QUEENS",
  "PROMOTABLE_PAWN",
  "PINNED_PIECES",
};

// captures = difference in values divided by 2

struct Evaluator {
  Evaluator() {
    earlyB = 0;
    std::fill_n(earlyW, EF::NUM_EVAL_FEATURES, 0);
    lateB = 0;
    std::fill_n(lateW, EF::NUM_EVAL_FEATURES, 0);
    clippedB = 0;
    std::fill_n(clippedW, EF::NUM_EVAL_FEATURES, 0);
  }

  int32_t earlyB;
  int32_t earlyW[EF::NUM_EVAL_FEATURES];
  int32_t lateB;
  int32_t lateW[EF::NUM_EVAL_FEATURES];
  int32_t clippedB;
  int32_t clippedW[EF::NUM_EVAL_FEATURES];

  void save_weights_to_file(std::ofstream& myfile) {
    myfile << lpad(earlyB) << "  // early bias" << std::endl;
    for (size_t i = 0; i < EF::NUM_EVAL_FEATURES; ++i) {
      myfile << lpad(earlyW[i]) << "  // early " << EFSTR[i] << std::endl;
    }

    myfile << lpad(lateB) << "  // late bias" << std::endl;
    for (size_t i = 0; i < EF::NUM_EVAL_FEATURES; ++i) {
      myfile << lpad(lateW[i]) << "  // late " << EFSTR[i] << std::endl;
    }

    myfile << lpad(clippedB) << "  // clipped bias" << std::endl;
    for (size_t i = 0; i < EF::NUM_EVAL_FEATURES; ++i) {
      myfile << lpad(clippedW[i]) << "  // clipped " << EFSTR[i] << std::endl;
    }
  }

  void load_weights_from_file(std::ifstream &myfile) {
    std::string line;
    std::vector<std::string> params;


    getline(myfile, line);
    try {
      earlyB = stoi(process_with_file_line(line));
    } catch (std::invalid_argument& err) {
      std::cout << "error loading earlyB" << std::endl;
      throw err;
    }
    for (size_t i = 0; i < EF::NUM_EVAL_FEATURES; ++i) {
      getline(myfile, line);
      try {
        earlyW[i] = stoi(process_with_file_line(line));
      } catch (std::invalid_argument& err) {
        std::cout << "error loading earlyW[i]" << std::endl;
        throw err;
      }
    }

    getline(myfile, line);
    try {
      lateB = stoi(process_with_file_line(line));
    } catch (std::invalid_argument& err) {
      std::cout << "error loading lateB" << std::endl;
      throw err;
    }
    for (size_t i = 0; i < EF::NUM_EVAL_FEATURES; ++i) {
      getline(myfile, line);
      try {
        lateW[i] = stoi(process_with_file_line(line));
      } catch (std::invalid_argument& err) {
        std::cout << "error loading lateW[i]" << std::endl;
        throw err;
      }
    }

    getline(myfile, line);
    try {
      clippedB = stoi(process_with_file_line(line));
    } catch (std::invalid_argument& err) {
      std::cout << "error loading clippedB" << std::endl;
      throw err;
    }
    for (size_t i = 0; i < EF::NUM_EVAL_FEATURES; ++i) {
      getline(myfile, line);
      try {
        clippedW[i] = stoi(process_with_file_line(line));
      } catch (std::invalid_argument& err) {
        std::cout << "error loading clippedW[" << i << "]" << std::endl;
        throw err;
      }
    }
  }

  Evaluation pawnValue() const {
    return this->earlyW[EF::OUR_PAWNS] + this->clippedW[EF::OUR_PAWNS];
  }

  template<Color US>
  Evaluation score(const Position& pos) {
    assert(pos.pieceBitboards_[ColoredPiece::WHITE_KING] > 0);
    assert(pos.pieceBitboards_[ColoredPiece::BLACK_KING] > 0);
    Threats<US> threats(pos);
    return this->score<US>(pos, threats);
  }

  bool is_material_draw(const Position& pos) const {
    const Bitboard everyone = pos.colorBitboards_[Color::WHITE] | pos.colorBitboards_[Color::BLACK];
    const Bitboard everyoneButKings = everyone & ~(pos.pieceBitboards_[ColoredPiece::WHITE_KING] | pos.pieceBitboards_[ColoredPiece::BLACK_KING]);
    const bool isThreeManEndgame = std::popcount(everyone) == 3;
    bool isDraw = false;
    isDraw |= (everyoneButKings == 0);
    isDraw |= (everyoneButKings == (pos.pieceBitboards_[ColoredPiece::WHITE_KNIGHT] | pos.pieceBitboards_[ColoredPiece::BLACK_KNIGHT])) && isThreeManEndgame;
    isDraw |= (everyoneButKings == (pos.pieceBitboards_[ColoredPiece::WHITE_BISHOP] | pos.pieceBitboards_[ColoredPiece::BLACK_BISHOP])) && isThreeManEndgame;
    return isDraw;
  }

  template<Color US>
  Evaluation score(const Position& pos, const Threats<US>& threats) {
    float score = pos.network->slowforward();
    return Evaluation(std::round(score * 100));
  }

  template<Color US>
  Evaluation special_boosts(const Position& pos) {
    constexpr Color THEM = opposite_color<US>();
    constexpr Evaluation kKnownWinBonus = 1000;

    const Square ourKingSq = lsb(pos.pieceBitboards_[coloredPiece<US, Piece::KING>()]);
    const Square theirKingSq = lsb(pos.pieceBitboards_[coloredPiece<THEM, Piece::KING>()]);

    const Bitboard ourPawns = pos.pieceBitboards_[coloredPiece<US, Piece::PAWN>()];
    const Bitboard ourKings = pos.pieceBitboards_[coloredPiece<US, Piece::KING>()];

    const Bitboard theirPawns = pos.pieceBitboards_[coloredPiece<THEM, Piece::PAWN>()];
    const Bitboard theirKings = pos.pieceBitboards_[coloredPiece<THEM, Piece::KING>()];

    const Square whiteKingSq = (US == Color::WHITE ? ourKingSq : theirKingSq);
    const Square blackKingSq = (US == Color::BLACK ? ourKingSq : theirKingSq);

    const Bitboard ourMen = pos.colorBitboards_[US];
    const Bitboard theirMen = pos.colorBitboards_[THEM];
    const Bitboard everyone = ourMen | theirMen;

    const bool doWeOnlyHavePawnsLeft = (ourMen & ~(ourPawns | ourKings)) == 0;
    const bool doTheyOnlyHavePawnsLeft = (theirMen & ~(theirPawns | theirKings)) == 0;
    const bool isKingPawnEndgame = doWeOnlyHavePawnsLeft && doTheyOnlyHavePawnsLeft;

    if (isKingPawnEndgame && (std::popcount(ourPawns) == 1) && (theirPawns == 0)) {
      int result;
      if (US == Color::WHITE) {
        result = known_kpvk_result(ourKingSq, theirKingSq, lsb(ourPawns), true);
      } else {
        result = known_kpvk_result(Square(63 - ourKingSq), Square(63 - theirKingSq), Square(63 - lsb(ourPawns)), true);
      }
      if (result == 0) {
        return kKnownDraw;
      }
      if (result == 2) {
        return 1000;
      }
    }
    else if (isKingPawnEndgame && (std::popcount(theirPawns) == 1) && (ourPawns == 0)) {
      int result;
      if (US == Color::BLACK) {
        result = known_kpvk_result(theirKingSq, ourKingSq, lsb(theirPawns), false);
      } else {
        result = known_kpvk_result(Square(63 - theirKingSq), Square(63 - ourKingSq), Square(63 - lsb(theirPawns)), false);
      }
      if (result == 0) {
        return kKnownDraw;
      }
      if (result == 2) {
        return -1000;
      }
    }

    const int wx = ourKingSq % 8;
    const int wy = ourKingSq / 8;
    const int bx = theirKingSq % 8;
    const int by = theirKingSq / 8;

    Evaluation r = 0;

    {
      const bool theyHaveLonelyKing = (theirMen == theirKings);
      const bool weHaveLonelyKing = (ourMen == ourKings);
      const bool weHaveMaterialToMate = (
        std::popcount(pos.pieceBitboards_[coloredPiece<US, Piece::KNIGHT>()]) > 2
        || std::popcount(pos.pieceBitboards_[coloredPiece<US, Piece::BISHOP>()]) > 1
        || std::popcount(pos.pieceBitboards_[coloredPiece<US, Piece::ROOK>()]) > 0
        || std::popcount(pos.pieceBitboards_[coloredPiece<US, Piece::QUEEN>()]) > 0
      );
      const bool theyHaveMaterialToMate = (
        std::popcount(pos.pieceBitboards_[coloredPiece<THEM, Piece::KNIGHT>()]) > 2
        || std::popcount(pos.pieceBitboards_[coloredPiece<THEM, Piece::BISHOP>()]) > 1
        || std::popcount(pos.pieceBitboards_[coloredPiece<THEM, Piece::ROOK>()]) > 0
        || std::popcount(pos.pieceBitboards_[coloredPiece<THEM, Piece::QUEEN>()]) > 0
      );

      r += value_or_zero(theyHaveLonelyKing && weHaveMaterialToMate, kKnownWinBonus);
      r -= value_or_zero(weHaveLonelyKing && theyHaveMaterialToMate, kKnownWinBonus);

      r += value_or_zero(theyHaveLonelyKing, (3 - kDistToEdge[theirKingSq]) * 50);
      r -= value_or_zero(  weHaveLonelyKing, (3 - kDistToEdge[ourKingSq]) * 50);
      r += value_or_zero(theyHaveLonelyKing, (3 - kDistToCorner[theirKingSq]) * 50);
      r -= value_or_zero(  weHaveLonelyKing, (3 - kDistToCorner[ourKingSq]) * 50);

      int dx = std::abs(wx - bx);
      int dy = std::abs(wy - by);
      const bool opposition = (dx == 2 && dy == 0) || (dx == 0 && dy == 2);

      // We don't want it to be our turn if they have the opposition.
      r -= value_or_zero(weHaveLonelyKing && opposition, 75);

      // And put our king next to the enemy king.
      r += value_or_zero(theyHaveLonelyKing, (8 - std::max(dx, dy)) * 50);
      r -= value_or_zero(  weHaveLonelyKing, (8 - std::max(dx, dy)) * 50);
      r += value_or_zero(theyHaveLonelyKing, (8 - std::min(dx, dy)) * 25);
      r -= value_or_zero(  weHaveLonelyKing, (8 - std::min(dx, dy)) * 25);
    }

    return r;
  }

  template<Color US>
  int32_t early(const Position& pos) const {
    int32_t r = earlyB;
    for (size_t i = 0; i < EF::NUM_EVAL_FEATURES; ++i) {
      r += features[i] * earlyW[i];
    }
    return r;
  }

  template<Color US>
  int32_t late(const Position& pos) const {
    int32_t r = lateB;
    for (size_t i = 0; i < EF::NUM_EVAL_FEATURES; ++i) {
      r += features[i] * lateW[i];
    }

    r += features[EF::SQUARE_RULE] * 2000;

    return r;
  }

  template<Color US>
  int32_t clipped(const Position& pos) const {
    int32_t r = clippedB;
    for (size_t i = 0; i < EF::NUM_EVAL_FEATURES; ++i) {
      r += features[i] * clippedW[i];
    }
    return std::max(-100, std::min(100, r));
  }

  void zero_() {
    for (size_t i = 0; i < EF::NUM_EVAL_FEATURES; ++i) {
      earlyW[i] = 0;
      lateW[i] = 0;
      clippedW[i] = 0;
    }
    earlyB = 0;
    lateB = 0;
    clippedB = 0;
  }

  Evaluation features[NUM_EVAL_FEATURES];
};

}  // namespace ChessEngine

#endif  // EVALUATOR_H
