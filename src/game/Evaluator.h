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

  // Endgame bonuses
  KNOWN_KPVK_DRAW,
  KNOWN_KPVK_WIN,
  LONELY_KING_ON_EDGE_AND_NOT_DRAW,
  LONELY_KING_IN_CORNER_AND_NOT_DRAW,
  LONELY_KING_OPPOSITION_AND_NOT_DRAW,
  LONELY_KING_ACHIEVABLE_OPPOSITION_AND_NOT_DRAW,
  LONELY_KING_NEXT_TO_ENEMY_KING,
  KING_TROPISM,

  PAWNS_X_QUEENS,
  PAWNS_X_QUEENS_2,
  PAWNS_X_KNIGHTS,
  PAWNS_X_KNIGHTS_2,
  PAWNS_X_BISHOPS,
  PAWNS_X_BISHOPS_2,
  PAWNS_X_ROOKS,
  PAWNS_X_ROOKS_2,
  KNIGHTS_X_QUEENS,
  KNIGHTS_X_QUEENS_2,
  BISHOPS_X_QUEENS,
  BISHOPS_X_QUEENS_2,
  ROOKS_X_QUEENS,
  ROOKS_X_QUEENS_2,

  KNOWN_DRAW,

  NUM_EVAL_FEATURES,
};

constexpr Bitboard kWhiteKingCorner = bb(SafeSquare::SH1) | bb(SafeSquare::SH2) | bb(SafeSquare::SG1) | bb(SafeSquare::SG2) | bb(SafeSquare::SF1);
constexpr Bitboard kWhiteQueenCorner = bb(SafeSquare::SA1) | bb(SafeSquare::SA2) | bb(SafeSquare::SB1) | bb(SafeSquare::SB2) | bb(SafeSquare::SC1);
constexpr Bitboard kBlackKingCorner = bb(SafeSquare::SH8) | bb(SafeSquare::SH7) | bb(SafeSquare::SG8) | bb(SafeSquare::SG7) | bb(SafeSquare::SF8);
constexpr Bitboard kBlackQueenCorner = bb(SafeSquare::SA8) | bb(SafeSquare::SA7) | bb(SafeSquare::SB8) | bb(SafeSquare::SB7) | bb(SafeSquare::SC8);

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
  "IN_CHECK_AND_OUR_HANGING_QUEENS",
  "PROMOTABLE_PAWN",
  "PINNED_PIECES",
  "KNOWN_KPVK_DRAW",
  "KNOWN_KPVK_WIN",
  "LONELY_KING_ON_EDGE_AND_NOT_DRAW",
  "LONELY_KING_IN_CORNER_AND_NOT_DRAW",
  "LONELY_KING_OPPOSITION_AND_NOT_DRAW",
  "LONELY_KING_ACHIEVABLE_OPPOSITION_AND_NOT_DRAW",
  "LONELY_KING_NEXT_TO_ENEMY_KING",
  "KING_TROPISM",
  "PAWNS_X_QUEENS",
  "PAWNS_X_QUEENS_2",
  "PAWNS_X_KNIGHTS",
  "PAWNS_X_KNIGHTS_2",
  "PAWNS_X_BISHOPS",
  "PAWNS_X_BISHOPS_2",
  "PAWNS_X_ROOKS",
  "PAWNS_X_ROOKS_2",
  "KNIGHTS_X_QUEENS",
  "KNIGHTS_X_QUEENS_2",
  "BISHOPS_X_QUEENS",
  "BISHOPS_X_QUEENS_2",
  "ROOKS_X_QUEENS",
  "ROOKS_X_QUEENS_2",
  "KNOWN_DRAW"
};

template<size_t N>
struct Flood {
  Bitboard result[N];
};
inline Bitboard _king_fill(Bitboard b) {
  return shift<Direction::EAST>(b)
      |  shift<Direction::WEST>(b)
      |  shift<Direction::NORTH>(b)
      |  shift<Direction::SOUTH>(b)
      |  shift<Direction::NORTH_EAST>(b)
      |  shift<Direction::NORTH_WEST>(b)
      |  shift<Direction::SOUTH_EAST>(b)
      |  shift<Direction::SOUTH_WEST>(b);
}
template<size_t N>
Flood<N> king_flood(SafeSquare kingSq, Bitboard blocked) {
  Flood<N> result;
  result[0] = bb(kingSq);
  for (int i = 1; i < N; ++i) {
    result[i] = _king_fill(result[i - 1]) & ~blocked;
  }
  return result;
}

struct Evaluator {
  Evaluator() {
    earlyB = 0;
    std::fill_n(earlyW, EF::NUM_EVAL_FEATURES, 0);
    lateB = 0;
    std::fill_n(lateW, EF::NUM_EVAL_FEATURES, 0);
    ineqB = 0;
    std::fill_n(ineqW, EF::NUM_EVAL_FEATURES, 0);
  }

  int32_t earlyB;
  int32_t earlyW[EF::NUM_EVAL_FEATURES];
  int32_t lateB;
  int32_t lateW[EF::NUM_EVAL_FEATURES];
  int32_t ineqB;
  int32_t ineqW[EF::NUM_EVAL_FEATURES];

  void save_weights_to_file(std::ostream& myfile) {
    myfile << lpad(earlyB) << "  // early bias" << std::endl;
    for (size_t i = 0; i < EF::NUM_EVAL_FEATURES; ++i) {
      myfile << lpad(earlyW[i]) << "  // early " << EFSTR[i] << std::endl;
    }

    myfile << lpad(lateB) << "  // late bias" << std::endl;
    for (size_t i = 0; i < EF::NUM_EVAL_FEATURES; ++i) {
      myfile << lpad(lateW[i]) << "  // late " << EFSTR[i] << std::endl;
    }

    myfile << lpad(ineqB) << "  // ineq bias" << std::endl;
    for (size_t i = 0; i < EF::NUM_EVAL_FEATURES; ++i) {
      myfile << lpad(ineqW[i]) << "  // ineq " << EFSTR[i] << std::endl;
    }
  }

  void load_weights_from_file(std::istream &myfile) {
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
      ineqB = stoi(process_with_file_line(line));
    } catch (std::invalid_argument& err) {
      std::cout << "error loading ineqB" << std::endl;
      throw err;
    }
    for (size_t i = 0; i < EF::NUM_EVAL_FEATURES; ++i) {
      getline(myfile, line);
      try {
        ineqW[i] = stoi(process_with_file_line(line));
      } catch (std::invalid_argument& err) {
        std::cout << "error loading ineqW[" << i << "]" << std::endl;
        throw err;
      }
    }
  }

  Evaluation pawnValue() const {
    return this->earlyW[EF::OUR_PAWNS] + this->ineqW[EF::OUR_PAWNS];
  }

  template<Color US>
  Evaluation score(const Position& pos) {
    assert(pos.pieceBitboards_[ColoredPiece::WHITE_KING] > 0);
    assert(pos.pieceBitboards_[ColoredPiece::BLACK_KING] > 0);
    Threats<US> threats(pos);
    return this->score<US>(pos, threats);
  }

  template<Color US>
  Evaluation score(const Position& pos, const Threats<US>& threats) {
    constexpr Color THEM = opposite_color<US>();

    assert(pos.pieceBitboards_[ColoredPiece::WHITE_KING] > 0);
    assert(pos.pieceBitboards_[ColoredPiece::BLACK_KING] > 0);

    if (pos.is_material_draw()) {
      return 0;
    }

    const SafeSquare ourKingSq = lsb_i_promise_board_is_not_empty(pos.pieceBitboards_[coloredPiece<US, Piece::KING>()]);
    const SafeSquare theirKingSq = lsb_i_promise_board_is_not_empty(pos.pieceBitboards_[coloredPiece<THEM, Piece::KING>()]);

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

    const SafeSquare whiteKingSq = (US == Color::WHITE ? ourKingSq : theirKingSq);
    const SafeSquare blackKingSq = (US == Color::BLACK ? ourKingSq : theirKingSq);

    const Bitboard ourMen = pos.colorBitboards_[US];
    const Bitboard theirMen = pos.colorBitboards_[THEM];
    const Bitboard everyone = ourMen | theirMen;

    // TODO: penalty for double attacks near king

    const Bitboard ourRoyalty = ourQueens | ourKings;
    const Bitboard theirRoyalty = theirQueens | theirKings;
    const Bitboard ourMajors = ourRooks | ourRoyalty;
    const Bitboard theirMajors = theirRooks | theirRoyalty;
    const Bitboard ourMinors = ourKnights | ourBishops;
    const Bitboard theirMinors = theirKnights | theirBishops;
    const Bitboard ourPieces = ourMajors | ourMinors;
    const Bitboard theirPieces = theirMajors | theirMinors;

    constexpr Bitboard kOurSide = (US == Color::WHITE ? kWhiteSide : kBlackSide);
    constexpr Bitboard kTheirSide = (US == Color::WHITE ? kBlackSide : kWhiteSide);
    constexpr Direction kForward = (US == Color::WHITE ? Direction::NORTH : Direction::SOUTH);
    constexpr Direction kForward2 = (US == Color::WHITE ? Direction::NORTHx2 : Direction::SOUTHx2);
    constexpr Direction kBackward = opposite_dir<kForward>();
    constexpr Direction kBackward2 = opposite_dir<kForward2>();
    constexpr Bitboard kOurBackRanks = (US == Color::WHITE ? kRanks[6] | kRanks[7] : kRanks[1] | kRanks[0]);
    constexpr Bitboard kTheirBackRanks = (US == Color::WHITE ? kRanks[1] | kRanks[0] : kRanks[6] | kRanks[7]);

    features[EF::OUR_PAWNS] = std::popcount(ourPawns);
    features[EF::OUR_KNIGHTS] = std::popcount(ourKnights);
    features[EF::OUR_BISHOPS] = std::popcount(ourBishops);
    features[EF::OUR_ROOKS] = std::popcount(ourRooks);
    features[EF::OUR_QUEENS] = std::popcount(ourQueens);
    features[EF::THEIR_PAWNS] = std::popcount(theirPawns);
    features[EF::THEIR_KNIGHTS] = std::popcount(theirKnights);
    features[EF::THEIR_BISHOPS] = std::popcount(theirBishops);
    features[EF::THEIR_ROOKS] = std::popcount(theirRooks);
    features[EF::THEIR_QUEENS] = std::popcount(theirQueens);

    features[EF::PAWNS_X_QUEENS] = features[EF::OUR_PAWNS] * features[EF::OUR_QUEENS] - features[EF::THEIR_PAWNS] * features[EF::THEIR_QUEENS];
    features[EF::PAWNS_X_QUEENS_2] = features[EF::OUR_PAWNS] * features[EF::THEIR_QUEENS] - features[EF::THEIR_PAWNS] * features[EF::OUR_QUEENS];
    features[EF::PAWNS_X_KNIGHTS] = features[EF::OUR_PAWNS] * features[EF::OUR_KNIGHTS] - features[EF::THEIR_PAWNS] * features[EF::THEIR_KNIGHTS];
    features[EF::PAWNS_X_KNIGHTS_2] = features[EF::OUR_PAWNS] * features[EF::THEIR_KNIGHTS] - features[EF::THEIR_PAWNS] * features[EF::OUR_KNIGHTS];
    features[EF::PAWNS_X_BISHOPS] = features[EF::OUR_PAWNS] * features[EF::OUR_BISHOPS] - features[EF::THEIR_PAWNS] * features[EF::THEIR_BISHOPS];
    features[EF::PAWNS_X_BISHOPS_2] = features[EF::OUR_PAWNS] * features[EF::THEIR_BISHOPS] - features[EF::THEIR_PAWNS] * features[EF::OUR_BISHOPS];
    features[EF::PAWNS_X_ROOKS] = features[EF::OUR_PAWNS] * features[EF::OUR_ROOKS] - features[EF::THEIR_PAWNS] * features[EF::THEIR_ROOKS];
    features[EF::PAWNS_X_ROOKS_2] = features[EF::OUR_PAWNS] * features[EF::THEIR_ROOKS] - features[EF::THEIR_PAWNS] * features[EF::OUR_ROOKS];
    features[EF::KNIGHTS_X_QUEENS] = features[EF::OUR_KNIGHTS] * features[EF::OUR_QUEENS] - features[EF::THEIR_KNIGHTS] * features[EF::THEIR_QUEENS];
    features[EF::KNIGHTS_X_QUEENS_2] = features[EF::OUR_KNIGHTS] * features[EF::THEIR_QUEENS] - features[EF::THEIR_KNIGHTS] * features[EF::OUR_QUEENS];
    features[EF::BISHOPS_X_QUEENS] = features[EF::OUR_BISHOPS] * features[EF::OUR_QUEENS] - features[EF::THEIR_BISHOPS] * features[EF::THEIR_QUEENS];
    features[EF::BISHOPS_X_QUEENS_2] = features[EF::OUR_BISHOPS] * features[EF::THEIR_QUEENS] - features[EF::THEIR_BISHOPS] * features[EF::OUR_QUEENS];
    features[EF::ROOKS_X_QUEENS] = features[EF::OUR_ROOKS] * features[EF::OUR_QUEENS] - features[EF::THEIR_ROOKS] * features[EF::THEIR_QUEENS];
    features[EF::ROOKS_X_QUEENS_2] = features[EF::OUR_ROOKS] * features[EF::THEIR_QUEENS] - features[EF::THEIR_ROOKS] * features[EF::OUR_QUEENS];

    const int16_t ourPiecesRemaining = std::popcount(pos.colorBitboards_[US] & ~ourPawns) + std::popcount(ourQueens) * 2 - 1;
    const int16_t theirPiecesRemaining = std::popcount(pos.colorBitboards_[THEM] & ~theirPawns) + std::popcount(theirQueens) * 2 - 1;
    const int32_t time = std::max(0, std::min(18, 18 - (ourPiecesRemaining + theirPiecesRemaining)));

    // Note that TURN (the side to move) often gets larger bonuses since they can take advantage of threats better.
    {
      Bitboard attackingOurKing = compute_attackers<THEM>(pos, ourKingSq);
      // Note: in a rigorous engine isThemInCheck would always be false on your move. For this engine,
      // it means they have moved into check (e.g. moving a pinned piece). Unfortunately we cannot
      // immediately return kMaxEval since it's possible they had had legal moves (and were in stalemate).
      features[EF::IN_CHECK] = (attackingOurKing > 0);
      features[EF::IN_DOUBLE_CHECK] = std::popcount(attackingOurKing) > 1;

      attackingOurKing &= ~(theirQueens & threats.badForTheir[Piece::QUEEN]);
      attackingOurKing &= ~(theirRooks & threats.badForTheir[Piece::ROOK]);
      attackingOurKing &= ~(theirBishops & threats.badForTheir[Piece::BISHOP]);
      attackingOurKing &= ~(theirKnights & threats.badForTheir[Piece::KNIGHT]);
      attackingOurKing &= ~(theirPawns & threats.badForTheir[Piece::PAWN]);

      // The piece checking our king can simply be captured.
      features[EF::IN_TRIVIAL_CHECK] = features[EF::IN_CHECK] && (attackingOurKing == 0) && !features[EF::IN_DOUBLE_CHECK];
    }

    if (US == Color::WHITE) {
      features[EF::KING_ON_BACK_RANK] = (ourKingSq / 8 == 7) - (theirKingSq / 8 == 0);
      features[EF::KING_ACTIVE] = (ourKingSq / 8 < 5) - (theirKingSq / 8 > 2);
    } else {
      features[EF::KING_ON_BACK_RANK] = (ourKingSq / 8 == 0) - (theirKingSq / 8 == 7);
      features[EF::KING_ACTIVE] = (ourKingSq / 8 > 2) - (theirKingSq / 8 < 5);
    }
    features[EF::KING_ON_CENTER_FILE] = (ourKingSq % 8 == 3 || ourKingSq % 8 == 4) - (theirKingSq % 8 == 3 || theirKingSq % 8 == 4);
    features[EF::THREATS_NEAR_KING_2] = std::popcount(kNearby[2][ourKingSq] & threats.theirTargets & ~threats.ourTargets) - std::popcount(kNearby[2][theirKingSq] & threats.ourTargets & ~threats.theirTargets);
    features[EF::THREATS_NEAR_KING_3] = std::popcount(kNearby[3][ourKingSq] & threats.theirTargets & ~threats.ourTargets) - std::popcount(kNearby[3][theirKingSq] & threats.ourTargets & ~threats.theirTargets);
    features[EF::QUEEN_THREATS_NEAR_KING] = std::popcount(kNearby[1][ourKingSq] & threats.theirDoubleTargets & ~threats.ourDoubleTargets & threats.theirQueenTargets) - std::popcount(kNearby[1][theirKingSq] & threats.ourDoubleTargets & ~threats.theirDoubleTargets & threats.ourQueenTargets);
    features[EF::THREATS_NEAR_OUR_KING] = std::popcount(kNearby[1][ourKingSq] & threats.theirDoubleTargets & ~threats.ourDoubleTargets);
    features[EF::THREATS_NEAR_THEIR_KING] = std::popcount(kNearby[1][theirKingSq] & threats.ourDoubleTargets & ~threats.theirDoubleTargets);

    PinMasks ourPinnedMask = compute_pin_masks<US>(pos, ourKingSq);
    PinMasks theirPinnedMask = compute_pin_masks<THEM>(pos, theirKingSq);
    ourPinnedMask |= compute_pin_masks<US>(pos, lsb_or(ourQueens, ourKingSq));
    theirPinnedMask |= compute_pin_masks<THEM>(pos, lsb_or(theirQueens, theirKingSq));

    //   0: 21.8462 ± 0.0106847
    // -10: 21.826 ± 0.0106827
    // -20: 21.819  ± 0.0106812
    // -40: 21.8631 ± 0.0106809

    // -20: 21.8045 ± 0.0106802
    // -40: 21.7966 ± 0.0106796
    // -60: 21.7961 ± 0.0106792

    // Bishops being pinned by a bishop or rooks being pinned by a rook aren't
    // a big deal. We're concerned with pieces being pinned and not being able
    // to capture their pinner.
     // 0.021 ± 0.015
    features[EF::PINNED_PIECES] = 
      std::popcount((ourPinnedMask.horizontal | ourPinnedMask.vertical) & (ourPieces & ~ourRooks))
      + std::popcount((ourPinnedMask.northeast | ourPinnedMask.northwest) & (ourPieces & ~ourBishops))
      - std::popcount((theirPinnedMask.horizontal | theirPinnedMask.vertical) & (theirPieces & ~theirRooks))
      - std::popcount((theirPinnedMask.northeast | theirPinnedMask.northwest) & (theirPieces & ~theirBishops));

    {  // Add penalty if the king is in a fianchettoed corner and his bishop is not on the main diagonal.
      // Note: the "color" of a corner is the color of its fianchettoed bishop.
      constexpr Bitboard kOurWhiteCorner = (US == Color::WHITE ? kWhiteKingCorner : kBlackQueenCorner);
      constexpr Bitboard kOurBlackCorner = (US == Color::BLACK ? kWhiteQueenCorner : kBlackKingCorner);
      constexpr Bitboard kTheirWhiteCorner = (US != Color::WHITE ? kWhiteKingCorner : kBlackQueenCorner);
      constexpr Bitboard kTheirBlackCorner = (US != Color::BLACK ? kWhiteQueenCorner : kBlackKingCorner);
      constexpr Bitboard ourWhiteFianchettoPawn = bb(US == Color::WHITE ? SafeSquare::SG2 : SafeSquare::SB7);
      constexpr Bitboard ourBlackFianchettoPawn = bb(US == Color::WHITE ? SafeSquare::SB2 : SafeSquare::SG7);
      constexpr Bitboard theirWhiteFianchettoPawn = bb(US != Color::WHITE ? SafeSquare::SG2 : SafeSquare::SB7);
      constexpr Bitboard theirBlackFianchettoPawn = bb(US != Color::WHITE ? SafeSquare::SB2 : SafeSquare::SG7);
      features[EF::MISSING_FIANCHETTO_BISHOP] = 0;
      features[EF::MISSING_FIANCHETTO_BISHOP] += ((ourKings & kOurWhiteCorner) > 0) && ((kMainWhiteDiagonal & ourBishops) == 0) && ((kWhiteSquares & theirBishops) > 0) && ((ourPawns & ourWhiteFianchettoPawn) == 0);
      features[EF::MISSING_FIANCHETTO_BISHOP] += ((ourKings & kOurBlackCorner) > 0) && ((kMainBlackDiagonal & ourBishops) == 0) && ((kBlackSquares & theirBishops) > 0) && ((ourPawns & ourBlackFianchettoPawn) == 0);
      features[EF::MISSING_FIANCHETTO_BISHOP] -= ((theirKings & kTheirWhiteCorner) > 0) && ((kMainWhiteDiagonal & theirBishops) == 0) && ((kWhiteSquares & ourBishops) > 0) && ((theirPawns & theirWhiteFianchettoPawn) == 0);
      features[EF::MISSING_FIANCHETTO_BISHOP] -= ((theirKings & kTheirBlackCorner) > 0) && ((kMainBlackDiagonal & theirBishops) == 0) && ((kBlackSquares & ourBishops) > 0) && ((theirPawns & theirBlackFianchettoPawn) == 0);
    }

    // Pawns
    PawnAnalysis<US> pawnAnalysis(pos, threats);
    {
      features[EF::PAWNS_CENTER_16] = std::popcount(ourPawns & kCenter16) - std::popcount(theirPawns & kCenter16);
      features[EF::PAWNS_CENTER_16] = std::popcount(ourPawns & kCenter16) - std::popcount(theirPawns & kCenter16);
      features[EF::PAWNS_CENTER_4] = std::popcount(ourPawns & kCenter4) - std::popcount(theirPawns & kCenter4);
      features[EF::PASSED_PAWNS] = std::popcount(pawnAnalysis.ourPassedPawns) - std::popcount(pawnAnalysis.theirPassedPawns);
      features[EF::ISOLATED_PAWNS] = std::popcount(pawnAnalysis.ourIsolatedPawns) - std::popcount(pawnAnalysis.theirIsolatedPawns);
      features[EF::DOUBLED_PAWNS] = std::popcount(pawnAnalysis.ourDoubledPawns) - std::popcount(pawnAnalysis.theirDoubledPawns);
      features[EF::DOUBLE_ISOLATED_PAWNS] = std::popcount(pawnAnalysis.ourDoubledPawns & pawnAnalysis.ourIsolatedPawns) - std::popcount(pawnAnalysis.theirDoubledPawns & pawnAnalysis.theirIsolatedPawns);
      features[EF::ADVANCED_PAWNS_1] = std::popcount(ourPawns & kTheirBackRanks) - std::popcount(theirPawns & kOurBackRanks);
      features[EF::ADVANCED_PAWNS_2] = std::popcount(ourPawns & shift<kBackward>(kTheirBackRanks)) - std::popcount(theirPawns & shift<kForward>(kOurBackRanks));

      if (US == Color::WHITE) {
        features[EF::ADVANCED_PASSED_PAWNS_2] = std::popcount(pawnAnalysis.ourPassedPawns & kRanks[1]) * 2 - std::popcount(pawnAnalysis.theirPassedPawns & kRanks[6]);
        features[EF::ADVANCED_PASSED_PAWNS_3] = std::popcount(pawnAnalysis.ourPassedPawns & kRanks[2]) * 2 - std::popcount(pawnAnalysis.theirPassedPawns & kRanks[5]);
        features[EF::ADVANCED_PASSED_PAWNS_4] = std::popcount(pawnAnalysis.ourPassedPawns & kRanks[3]) * 2 - std::popcount(pawnAnalysis.theirPassedPawns & kRanks[4]);
      } else {
        features[EF::ADVANCED_PASSED_PAWNS_2] = std::popcount(pawnAnalysis.ourPassedPawns & kRanks[6]) * 2 - std::popcount(pawnAnalysis.theirPassedPawns & kRanks[1]);
        features[EF::ADVANCED_PASSED_PAWNS_3] = std::popcount(pawnAnalysis.ourPassedPawns & kRanks[5]) * 2 - std::popcount(pawnAnalysis.theirPassedPawns & kRanks[2]);
        features[EF::ADVANCED_PASSED_PAWNS_4] = std::popcount(pawnAnalysis.ourPassedPawns & kRanks[4]) * 2 - std::popcount(pawnAnalysis.theirPassedPawns & kRanks[3]);
      }

      features[EF::PAWN_MINOR_CAPTURES] = std::popcount(threats.ourPawnTargets & theirMinors) - std::popcount(threats.theirPawnTargets & ourMinors);
      features[EF::PAWN_MAJOR_CAPTURES] = std::popcount(threats.ourPawnTargets & theirMajors) - std::popcount(threats.theirPawnTargets & ourMajors);
      features[EF::PROTECTED_PAWNS] = std::popcount(ourPawns & threats.ourPawnTargets) - std::popcount(theirPawns & threats.theirPawnTargets);
      features[EF::PROTECTED_PASSED_PAWNS] = std::popcount(pawnAnalysis.ourPassedPawns & threats.ourPawnTargets) - std::popcount(pawnAnalysis.theirPassedPawns & threats.theirPawnTargets);

      features[EF::NUM_PIECES_HARRASSABLE_BY_PAWNS] = std::popcount(pawnAnalysis.piecesOurPawnsCanThreaten) - std::popcount(pawnAnalysis.piecesTheirPawnsCanThreaten);

      Bitboard whitePromoSquares = shift<Direction::NORTH>(pos.pieceBitboards_[ColoredPiece::WHITE_PAWN] & kRanks[1]);
      Bitboard blackPromoSquares = shift<Direction::SOUTH>(pos.pieceBitboards_[ColoredPiece::BLACK_PAWN] & kRanks[6]);
      if (US == Color::WHITE) {
        features[EF::PROMOTABLE_PAWN] =
        std::popcount(whitePromoSquares & ~threats.badForOur[Piece::QUEEN] & ~everyone)
        - std::popcount(blackPromoSquares & ~threats.badForTheir[Piece::QUEEN] & ~everyone);
      } else {
        features[EF::PROMOTABLE_PAWN] =
        std::popcount(blackPromoSquares & ~threats.badForOur[Piece::QUEEN & ~everyone])
        - std::popcount(whitePromoSquares & ~threats.badForTheir[Piece::QUEEN] & ~everyone);
      }
    }

    const Bitboard ourBishopTargetsIgnoringNonBlockades = compute_bishoplike_targets(ourBishops, pawnAnalysis.ourBlockadedPawns);
    const Bitboard theirBishopTargetsIgnoringNonBlockades = compute_bishoplike_targets(theirBishops, pawnAnalysis.theirBlockadedPawns);
    {  // Bishops
      if (US == Color::WHITE) {
        features[EF::BISHOPS_DEVELOPED] = std::popcount(theirBishops & (bb(SafeSquare::SC8) | bb(SafeSquare::SF8))) - std::popcount(ourBishops & (bb(SafeSquare::SC1) | bb(SafeSquare::SF1)));
      } else {
        features[EF::BISHOPS_DEVELOPED] = std::popcount(theirBishops & (bb(SafeSquare::SC1) | bb(SafeSquare::SF1))) - std::popcount(ourBishops & (bb(SafeSquare::SC8) | bb(SafeSquare::SF8)));
      }
      features[EF::BISHOP_PAIR] = (std::popcount(ourBishops) >= 2) - (std::popcount(theirBishops) >= 2);
      features[EF::BLOCKADED_BISHOPS] = std::popcount(ourBishopTargetsIgnoringNonBlockades & (pawnAnalysis.ourBlockadedPawns | pawnAnalysis.theirProtectedPawns)) - std::popcount(theirBishopTargetsIgnoringNonBlockades & (pawnAnalysis.theirBlockadedPawns | pawnAnalysis.ourProtectedPawns));
      features[EF::SCARY_BISHOPS] = std::popcount(ourBishopTargetsIgnoringNonBlockades & theirMajors) - std::popcount(theirBishopTargetsIgnoringNonBlockades & ourMajors);
      features[EF::SCARIER_BISHOPS] = std::popcount(ourBishopTargetsIgnoringNonBlockades & theirRoyalty) - std::popcount(theirBishopTargetsIgnoringNonBlockades & ourRoyalty);
      features[EF::OUTPOSTED_BISHOPS] = std::popcount(ourBishops & pawnAnalysis.possibleOutpostsForUs) - std::popcount(theirBishops & pawnAnalysis.possibleOutpostsForThem);
    }

    {  // Rooks
      const Bitboard openFiles = pawnAnalysis.filesWithoutOurPawns & pawnAnalysis.filesWithoutTheirPawns;
      features[EF::BLOCKADED_ROOKS] = std::popcount(ourRooks & pawnAnalysis.filesWithoutOurPawns) - std::popcount(theirRooks & pawnAnalysis.filesWithoutTheirPawns);
      features[EF::SCARY_ROOKS] = std::popcount(threats.ourRookTargets & theirRoyalty) - std::popcount(threats.theirRookTargets & ourRoyalty);
      features[EF::INFILTRATING_ROOKS] = std::popcount(ourRooks & kTheirBackRanks) - std::popcount(theirRooks & kOurBackRanks);
      features[EF::OPEN_ROOKS] = std::popcount(ourRooks & openFiles) - std::popcount(theirRooks & openFiles);
      features[EF::ROOKS_ON_THEIR_SIDE] = std::popcount(ourRooks & kTheirSide) - std::popcount(theirRooks & kOurSide);
    }

    {  // Knights
      if (US == Color::WHITE) {
        features[EF::KNIGHTS_DEVELOPED] = std::popcount(theirKnights & (bb(SafeSquare::SB8) | bb(SafeSquare::SG8))) - std::popcount(ourKnights & (bb(SafeSquare::SB1) | bb(SafeSquare::SG1)));
      } else {
        features[EF::KNIGHTS_DEVELOPED] = std::popcount(theirKnights & (bb(SafeSquare::SB1) | bb(SafeSquare::SG1))) - std::popcount(ourKnights & (bb(SafeSquare::SB8) | bb(SafeSquare::SG8)));
      }
      features[EF::KNIGHT_MAJOR_CAPTURES] = std::popcount(threats.ourKnightTargets & theirMajors) - std::popcount(threats.theirKnightTargets & ourMajors);
      features[EF::KNIGHTS_CENTER_16] = std::popcount(ourKnights & kCenter16) - std::popcount(theirKnights & kCenter16);
      features[EF::KNIGHTS_CENTER_4] = std::popcount(ourKnights & kCenter4) - std::popcount(theirKnights & kCenter4);
      features[EF::KNIGHT_ON_ENEMY_SIDE] = std::popcount(ourKnights & kTheirSide) - std::popcount(theirKnights & kOurSide);
      features[EF::OUTPOSTED_KNIGHTS] = std::popcount(ourKnights & pawnAnalysis.possibleOutpostsForUs & kTheirSide) - std::popcount(theirKnights & pawnAnalysis.possibleOutpostsForThem & kOurSide);

      // TODO: use kBackward for their knights.
      features[EF::BISHOPS_BLOCKING_KNIGHTS] = std::popcount(shift<kBackward>(shift<kBackward>(shift<kBackward>(theirKnights))) & ourBishops) - std::popcount(shift<kForward>(shift<kForward>(shift<kForward>(ourKnights))) & theirBishops);
    }

    const bool isOurKingLonely = (ourMen == ourKings);
    const bool isTheirKingLonely = (theirMen == theirKings);

    const CastlingRights cr = pos.currentState_.castlingRights;
    {
      // Hanging pieces are more valuable if it is your turn since they're literally free material,
      // as opposed to threats. Also is a very useful heuristic so that leaf nodes don't sack a rook
      // for a pawn.
      const Bitboard usHanging = threats.theirTargets & ~threats.ourTargets & pos.colorBitboards_[US];
      const Bitboard themHanging = threats.ourTargets & ~threats.theirTargets & pos.colorBitboards_[THEM];

      bool anyOfUsHanging = false;
      bool anyOfThemHanging = false;

      features[EF::THEIR_HANGING_QUEENS] = std::popcount(theirQueens & themHanging);
      features[EF::OUR_HANGING_QUEENS] = std::popcount(ourQueens & usHanging);
      anyOfThemHanging |= features[EF::THEIR_HANGING_QUEENS];
      anyOfUsHanging |= features[EF::OUR_HANGING_QUEENS];

      features[EF::THEIR_HANGING_ROOKS] = std::popcount(theirRooks & themHanging) && !anyOfThemHanging;
      features[EF::OUR_HANGING_ROOKS] = std::popcount(ourRooks & usHanging) && !anyOfUsHanging;
      anyOfThemHanging |= features[EF::THEIR_HANGING_ROOKS];
      anyOfUsHanging |= features[EF::OUR_HANGING_ROOKS];

      features[EF::THEIR_HANGING_BISHOPS] = std::popcount(theirBishops & themHanging) && !anyOfThemHanging;
      features[EF::OUR_HANGING_BISHOPS] = std::popcount(ourBishops & usHanging) && !anyOfUsHanging;
      anyOfThemHanging |= features[EF::THEIR_HANGING_BISHOPS];
      anyOfUsHanging |= features[EF::OUR_HANGING_BISHOPS];

      features[EF::THEIR_HANGING_KNIGHTS] = std::popcount(theirKnights & themHanging) && !anyOfThemHanging;
      features[EF::OUR_HANGING_KNIGHTS] = std::popcount(ourKnights & usHanging) && !anyOfUsHanging;
      anyOfThemHanging |= features[EF::THEIR_HANGING_KNIGHTS];
      anyOfUsHanging |= features[EF::OUR_HANGING_KNIGHTS];

      features[EF::THEIR_HANGING_PAWNS] = std::popcount(theirPawns & themHanging) && !anyOfThemHanging;
      features[EF::OUR_HANGING_PAWNS] = std::popcount(ourPawns & usHanging) && !anyOfUsHanging;

      features[EF::IN_CHECK_AND_OUR_HANGING_QUEENS] = features[EF::OUR_HANGING_QUEENS] && features[EF::IN_CHECK];

      const int wx = ourKingSq % 8;
      const int wy = ourKingSq / 8;
      const int bx = theirKingSq % 8;
      const int by = theirKingSq / 8;
      const int kingsDist = std::max(std::abs(wx - bx), std::abs(wy - by));
      features[EF::LONELY_KING_IN_CENTER] = value_or_zero(std::popcount(pos.colorBitboards_[THEM]) == 1, 3 - kDistToCorner[theirKingSq]);
      features[EF::LONELY_KING_IN_CENTER] -= value_or_zero(std::popcount(pos.colorBitboards_[US]) == 1, 3 - kDistToCorner[ourKingSq]);

      features[EF::LONELY_KING_AWAY_FROM_ENEMY_KING] = value_or_zero(isTheirKingLonely, 8 - kingsDist);
      features[EF::LONELY_KING_AWAY_FROM_ENEMY_KING] -= value_or_zero(isOurKingLonely, 8 - kingsDist);
    }

    {
      const Bitboard usHanging = threats.theirDoubleTargets & ~threats.ourDoubleTargets & pos.colorBitboards_[US];
      const Bitboard themHanging = threats.ourDoubleTargets & ~threats.theirDoubleTargets & pos.colorBitboards_[THEM];
      features[EF::OUR_HANGING_PAWNS_2] = std::popcount(ourPawns & usHanging);
      features[EF::THEIR_HANGING_PAWNS_2] = std::popcount(theirPawns & themHanging);
      features[EF::OUR_HANGING_KNIGHTS_2] = std::popcount(ourKnights & usHanging);
      features[EF::THEIR_HANGING_KNIGHTS_2] = std::popcount(theirKnights & themHanging);
      features[EF::OUR_HANGING_BISHOPS_2] = std::popcount(ourBishops & usHanging);
      features[EF::THEIR_HANGING_BISHOPS_2] = std::popcount(theirBishops & themHanging);
      features[EF::OUR_HANGING_ROOKS_2] = std::popcount(ourRooks & usHanging);
      features[EF::THEIR_HANGING_ROOKS_2] = std::popcount(theirRooks & themHanging);
      features[EF::OUR_HANGING_QUEENS_2] = std::popcount(ourQueens & usHanging);
      features[EF::THEIR_HANGING_QUEENS_2] = std::popcount(theirQueens & themHanging);
    }

    const bool isKingPawnEndgame = (ourKings == ourPieces) && (theirKings == theirPieces);
    // TODO: move this negative into the corresponding weights.
    features[EF::KPVK_OPPOSITION] = value_or_zero(isKingPawnEndgame, -((shift<kForward>(shift<kForward>(ourKings)) & theirKings) > 0));

    features[EF::SQUARE_RULE] = 0;
    features[EF::SQUARE_RULE] += value_or_zero(isKingPawnEndgame, (kSquareRuleTheirTurn[US][theirKingSq] & pawnAnalysis.ourPassedPawns) > 0);
    features[EF::SQUARE_RULE] -= value_or_zero(isKingPawnEndgame, (kSquareRuleYourTurn[THEM][ourKingSq] & pawnAnalysis.theirPassedPawns) > 0);

    Bitboard aheadOfOurPassedPawnsFat, aheadOfTheirPassedPawnsFat;
    if (US == Color::WHITE) {
      aheadOfOurPassedPawnsFat = fatten(northFill(pawnAnalysis.ourPassedPawns));
      aheadOfTheirPassedPawnsFat = fatten(southFill(pawnAnalysis.theirPassedPawns));
    } else {
      aheadOfOurPassedPawnsFat = fatten(southFill(pawnAnalysis.ourPassedPawns));
      aheadOfTheirPassedPawnsFat = fatten(northFill(pawnAnalysis.theirPassedPawns));
    }

    // We split these into two features, the idea being that being ahead of your pawns while your opponent's
    // queen is on the board is dangerous, but being ahead of your opponent's passed pawns is not.
    features[EF::KING_IN_FRONT_OF_PASSED_PAWN] = ((ourKings & aheadOfOurPassedPawnsFat) > 0 && theirQueens == 0);
    features[EF::KING_IN_FRONT_OF_PASSED_PAWN] -= ((theirKings & aheadOfTheirPassedPawnsFat) > 0 && ourQueens == 0);
    features[EF::KING_IN_FRONT_OF_PASSED_PAWN2] = (ourKings & aheadOfTheirPassedPawnsFat) > 0;
    features[EF::KING_IN_FRONT_OF_PASSED_PAWN2] -= (theirKings & aheadOfOurPassedPawnsFat) > 0;

    // loser in center: 0
    // they're lonely on edge: 3
    //   we're lonely on edge: -3
    features[EF::LONELY_KING_ON_EDGE] = value_or_zero(isTheirKingLonely, 3 - kDistToEdge[theirKingSq]);
    features[EF::LONELY_KING_ON_EDGE] -= value_or_zero(isOurKingLonely, 3 - kDistToEdge[ourKingSq]);

    {
      Bitboard ourMaterialThreats = 0;
      ourMaterialThreats |= threats.ourPawnTargets & theirPieces;
      ourMaterialThreats |= threats.ourKnightTargets & theirMajors;
      ourMaterialThreats |= threats.ourBishopTargets & theirMajors;
      ourMaterialThreats |= threats.ourRookTargets & theirRoyalty;
      ourMaterialThreats |= threats.ourQueenTargets & theirKings;

      Bitboard theirMaterialThreats = 0;
      theirMaterialThreats |= threats.theirPawnTargets & ourPieces;
      theirMaterialThreats |= threats.theirKnightTargets & ourMajors;
      theirMaterialThreats |= threats.theirBishopTargets & ourMajors;
      theirMaterialThreats |= threats.theirRookTargets & ourRoyalty;
      theirMaterialThreats |= threats.theirQueenTargets & ourKings;

      features[EF::OUR_MATERIAL_THREATS] = std::popcount(ourMaterialThreats);
      features[EF::THEIR_MATERIAL_THREATS] = std::popcount(theirMaterialThreats);
    }

    {
      const Bitboard ourGoodKnightTargets = threats.ourKnightTargets & ~threats.badForOur[Piece::KNIGHT] & ~ourMen;
      const Bitboard theirGoodKnightTargets = threats.theirKnightTargets & ~threats.badForTheir[Piece::KNIGHT] & ~theirMen;
      const Bitboard ourGoodBishopTargets = ourBishopTargetsIgnoringNonBlockades & ~threats.badForOur[Piece::BISHOP] & ~ourMen;
      const Bitboard theirGoodBishopTargets = theirBishopTargetsIgnoringNonBlockades & ~threats.badForTheir[Piece::BISHOP] & ~theirMen;
      const Bitboard ourGoodRookTargets = threats.ourRookTargets & ~threats.badForOur[Piece::ROOK] & ~ourMen;
      const Bitboard theirGoodRookTargets = threats.theirRookTargets & ~threats.badForTheir[Piece::ROOK] & ~theirMen;
      const Bitboard ourGoodQueenTargets = threats.ourQueenTargets & ~threats.badForOur[Piece::QUEEN] & ~ourMen;
      const Bitboard theirGoodQueenTargets = threats.theirQueenTargets & ~threats.badForTheir[Piece::QUEEN] & ~theirMen;
      features[EF::PAWN_MOVES] = std::popcount(threats.ourPawnTargets) - std::popcount(threats.theirPawnTargets);
      features[EF::KNIGHT_MOVES] = std::popcount(ourGoodKnightTargets) - std::popcount(theirGoodKnightTargets);
      features[EF::BISHOP_MOVES] = std::popcount(ourGoodBishopTargets) - std::popcount(theirGoodBishopTargets);
      features[EF::ROOK_MOVES] = std::popcount(ourGoodRookTargets) - std::popcount(theirGoodRookTargets);
      features[EF::QUEEN_MOVES] = std::popcount(ourGoodQueenTargets) - std::popcount(theirGoodQueenTargets);
      features[EF::PAWN_MOVES_ON_THEIR_SIDE] = std::popcount(threats.ourPawnTargets & kTheirSide) - std::popcount(threats.theirPawnTargets & kOurSide);
      features[EF::KNIGHT_MOVES_ON_THEIR_SIDE] = std::popcount(ourGoodKnightTargets & kTheirSide) - std::popcount(theirGoodKnightTargets & kOurSide);
      features[EF::BISHOP_MOVES_ON_THEIR_SIDE] = std::popcount(ourGoodBishopTargets & kTheirSide) - std::popcount(theirGoodBishopTargets & kOurSide);
      features[EF::ROOK_MOVES_ON_THEIR_SIDE] = std::popcount(ourGoodRookTargets & kTheirSide) - std::popcount(theirGoodRookTargets & kOurSide);
      features[EF::QUEEN_MOVES_ON_THEIR_SIDE] = std::popcount(ourGoodQueenTargets & kTheirSide) - std::popcount(theirGoodQueenTargets & kOurSide);
    }

    // Bonus for king having pawns in front of him, or having pawns in front of him once he castles.
    features[EF::KING_HOME_QUALITY] = std::popcount(kKingHome[ourKingSq] & ourPawns) - std::popcount(kKingHome[theirKingSq] & theirPawns);
    const Evaluation whitePotentialHome = std::max(
      value_or_zero((cr & kCastlingRights_WhiteKing) > 0, std::popcount(kKingHome[SafeSquare::SG1] & ourPawns)),
      value_or_zero((cr & kCastlingRights_WhiteQueen) > 0, std::popcount(kKingHome[SafeSquare::SB1] & ourPawns))
    );
    const Evaluation blackPotentialHome = std::max(
      value_or_zero((cr & kCastlingRights_BlackKing) > 0, std::popcount(kKingHome[SafeSquare::SG8] & theirPawns)),
      value_or_zero((cr & kCastlingRights_BlackQueen) > 0, std::popcount(kKingHome[SafeSquare::SB8] & theirPawns))
    );
    if (US == Color::WHITE) {
      features[EF::KING_HOME_QUALITY] += whitePotentialHome - blackPotentialHome;
    } else {
      features[EF::KING_HOME_QUALITY] += blackPotentialHome - whitePotentialHome;
    }

    features[EF::NUM_BAD_SQUARES_FOR_PAWNS] = std::popcount(threats.badForOur[Piece::PAWN] & kCenter16) - std::popcount(threats.badForTheir[Piece::PAWN] & kCenter16);
    features[EF::NUM_BAD_SQUARES_FOR_MINORS] = std::popcount(threats.badForOur[Piece::KNIGHT] & kCenter16) - std::popcount(threats.badForTheir[Piece::KNIGHT] & kCenter16);
    features[EF::NUM_BAD_SQUARES_FOR_ROOKS] = std::popcount(threats.badForOur[Piece::ROOK] & kCenter16) - std::popcount(threats.badForTheir[Piece::ROOK] & kCenter16);
    features[EF::NUM_BAD_SQUARES_FOR_QUEENS] = std::popcount(threats.badForOur[Piece::QUEEN] & kCenter16) - std::popcount(threats.badForTheir[Piece::QUEEN] & kCenter16);

    // Checks
    // TODO: check map for our king too.
    const CheckMap checkMap = compute_potential_attackers<US>(pos, theirKingSq);
    {
      const Bitboard forwardPawns = shift<kForward>(ourPawns) & ~everyone;
      const Bitboard thirdRank = (US == Color::WHITE ? kRanks[5] : kRanks[2]);
      const Bitboard pawnCaptureChecks = checkMap.data[Piece::PAWN] & threats.ourPawnTargets & theirMen;
      Bitboard pawnPushChecks = checkMap.data[Piece::PAWN] & (forwardPawns | (shift<kForward>(forwardPawns & thirdRank) & ~everyone));
      const Bitboard knightChecks = checkMap.data[Piece::KNIGHT] & threats.ourKnightTargets;
      const Bitboard bishopChecks = checkMap.data[Piece::BISHOP] & threats.ourBishopTargets;
      const Bitboard rookChecks = checkMap.data[Piece::ROOK] & threats.ourRookTargets;
      const Bitboard queenChecks = checkMap.data[Piece::QUEEN] & threats.ourQueenTargets;

      // NOTE: we can't use badForOur[Piece::BISHOP] (or any piece) since the piece delivering the check
      // stops protecting the square when it moves.
      Bitboard safe = ~threats.theirTargets;
      safe |= (threats.ourTargets & ~threats.theirTargets) | (threats.ourDoubleTargets & ~threats.theirDoubleTargets);

      // NOTE: we *can* use threats.badForOur[Piece::PAWN], since it doesn't count the pawn's forward
      // push when determining square safety.

      features[EF::PAWN_CHECKS] = std::popcount((pawnPushChecks & ~threats.badForOur[Piece::PAWN]) | (pawnCaptureChecks & safe));
      features[EF::KNIGHT_CHECKS] = std::popcount(knightChecks & safe);
      features[EF::BISHOP_CHECKS] = std::popcount(bishopChecks & safe);
      features[EF::ROOK_CHECKS] = std::popcount(rookChecks & safe);
      features[EF::QUEEN_CHECKS] = std::popcount(queenChecks & safe);
    }

    {
      const Bitboard whitePawns = (US == Color::WHITE ? ourPawns : theirPawns);
      const Bitboard blackPawns = (US == Color::BLACK ? ourPawns : theirPawns);
      const Bitboard whiteKingCannotMoveTo = threats.template badFor<ColoredPiece::WHITE_KING>();
      const Bitboard blackKingCannotMoveTo = threats.template badFor<ColoredPiece::BLACK_KING>();
      const Bitboard whiteKingEscapes = compute_king_targets<Color::WHITE>(pos, whiteKingSq) & ~(whiteKingCannotMoveTo | whitePawns);
      const Bitboard blackKingEscapes = compute_king_targets<Color::BLACK>(pos, blackKingSq) & ~(blackKingCannotMoveTo | blackPawns);

      const bool backRankMateThreatAgainstWhite = (whiteKingSq >= 56 && ((whiteKingEscapes & kRanks[7]) == whiteKingEscapes) && ((threats.template targets<ColoredPiece::BLACK_ROOK>() & kRanks[7]) > 0));
      const bool backRankMateThreatAgainstBlack = (blackKingSq <=  7 && ((blackKingEscapes & kRanks[0]) == blackKingEscapes) && ((threats.template targets<ColoredPiece::WHITE_ROOK>() & kRanks[0]) > 0));

      if (US == Color::WHITE) {
        features[EF::BACK_RANK_MATE_THREAT_AGAINST_US] = backRankMateThreatAgainstWhite;
        features[EF::BACK_RANK_MATE_THREAT_AGAINST_THEM] = backRankMateThreatAgainstBlack;
      } else {
        features[EF::BACK_RANK_MATE_THREAT_AGAINST_US] = backRankMateThreatAgainstBlack;
        features[EF::BACK_RANK_MATE_THREAT_AGAINST_THEM] = backRankMateThreatAgainstWhite;
      }

      const Bitboard whitePieces = (US == Color::WHITE ? ourPieces : theirPieces);
      const Bitboard blackPieces = (US == Color::BLACK ? ourPieces : theirPieces);
      const int ourKingEscapes = std::popcount(US == Color::WHITE ? (whiteKingEscapes & ~whitePieces) : (blackKingEscapes & ~blackPieces));
      const int theirKingEscapes = std::popcount(US == Color::BLACK ? (whiteKingEscapes & ~whitePieces) : (blackKingEscapes & ~blackPieces));

      // TODO: probably should remove this
      features[EF::OUR_KING_HAS_0_ESCAPE_SQUARES] = (ourKingEscapes == 0);
      features[EF::THEIR_KING_HAS_0_ESCAPE_SQUARES] = (theirKingEscapes == 0);
      features[EF::OUR_KING_HAS_1_ESCAPE_SQUARES] = (ourKingEscapes == 1);
      features[EF::THEIR_KING_HAS_1_ESCAPE_SQUARES] = (theirKingEscapes == 1);
      features[EF::OUR_KING_HAS_2_ESCAPE_SQUARES] = (ourKingEscapes == 2);
      features[EF::THEIR_KING_HAS_2_ESCAPE_SQUARES] = (theirKingEscapes == 2);
    }

    const int kingsOnOppositesSideOfBoard = std::min(std::abs(whiteKingSq % 8 - blackKingSq % 8), 6);
    Bitboard aheadOfOurKing = fatten(ourKings);
    Bitboard aheadOfTheirKing = fatten(theirKings);
    if (US == Color::WHITE) {
      aheadOfOurKing >>= 8;
      aheadOfOurKing |= aheadOfOurKing >> 8;
      aheadOfOurKing |= aheadOfOurKing >> 16;
      aheadOfTheirKing <<= 8;
      aheadOfTheirKing |= aheadOfTheirKing << 8;
      aheadOfTheirKing |= aheadOfTheirKing << 16;
    } else {
      aheadOfOurKing <<= 8;
      aheadOfOurKing |= aheadOfOurKing << 8;
      aheadOfOurKing |= aheadOfOurKing << 16;
      aheadOfTheirKing >>= 8;
      aheadOfTheirKing |= aheadOfTheirKing >> 8;
      aheadOfTheirKing |= aheadOfTheirKing >> 16;
    }
    features[EF::OPPOSITE_SIDE_KINGS_PAWN_STORM] = kingsOnOppositesSideOfBoard * (std::popcount(aheadOfTheirKing & ourPawns) - std::popcount(aheadOfOurKing & theirPawns));

    features[EF::TIME] = time;

    const int wx = ourKingSq % 8;
    const int wy = ourKingSq / 8;
    const int bx = theirKingSq % 8;
    const int by = theirKingSq / 8;

    {
      features[EF::KNOWN_KPVK_DRAW] = 0;
      features[EF::KNOWN_KPVK_WIN] = 0;
      const bool isOurKPPVK = (theirMen == theirKings) && (std::popcount(pawnAnalysis.ourPassedPawns) >= 1);
      const bool isTheirKPPVK = (ourMen == ourKings) && (std::popcount(pawnAnalysis.theirPassedPawns) >= 1);
      // KPVK games are winning if square rule is true.
      if (isOurKPPVK && std::popcount(pawnAnalysis.ourPassedPawns) >= 1) {
        if (US == Color::WHITE) {
          features[EF::KNOWN_KPVK_WIN] = is_kpvk_win(ourKingSq, theirKingSq, lsb_i_promise_board_is_not_empty(pawnAnalysis.ourPassedPawns), true);
          features[EF::KNOWN_KPVK_DRAW] = (ourPieces == ourKings) && (std::popcount(ourPawns) == 1) && is_kpvk_draw(ourKingSq, theirKingSq, lsb_i_promise_board_is_not_empty(pawnAnalysis.ourPassedPawns), true);
        } else {
          features[EF::KNOWN_KPVK_WIN] = is_kpvk_win(SafeSquare(63 - ourKingSq), SafeSquare(63 - theirKingSq), SafeSquare(63 - msb_i_promise_board_is_not_empty(pawnAnalysis.ourPassedPawns)), true);
          features[EF::KNOWN_KPVK_DRAW] = (ourPieces == ourKings) && (std::popcount(ourPawns) == 1) && is_kpvk_draw(SafeSquare(63 - ourKingSq), SafeSquare(63 - theirKingSq), SafeSquare(63 - msb_i_promise_board_is_not_empty(pawnAnalysis.ourPassedPawns)), true);
        }
      }
      if (isTheirKPPVK && std::popcount(pawnAnalysis.theirPassedPawns) >= 1) {
        if (US == Color::BLACK) {
          features[EF::KNOWN_KPVK_WIN] = is_kpvk_win(theirKingSq, ourKingSq, lsb_i_promise_board_is_not_empty(pawnAnalysis.theirPassedPawns), false);
          features[EF::KNOWN_KPVK_DRAW] = (theirPieces == theirKings) && (std::popcount(theirPawns) == 1) && is_kpvk_draw(theirKingSq, ourKingSq, lsb_i_promise_board_is_not_empty(pawnAnalysis.theirPassedPawns), false);
        } else {
          features[EF::KNOWN_KPVK_WIN] = is_kpvk_win(SafeSquare(63 - theirKingSq), SafeSquare(63 - ourKingSq), SafeSquare(63 - msb_i_promise_board_is_not_empty(pawnAnalysis.theirPassedPawns)), false);
          features[EF::KNOWN_KPVK_DRAW] = (theirPieces == theirKings) && (std::popcount(theirPawns) == 1) && is_kpvk_draw(SafeSquare(63 - theirKingSq), SafeSquare(63 - ourKingSq), SafeSquare(63 - msb_i_promise_board_is_not_empty(pawnAnalysis.theirPassedPawns)), false);
        }
      }
    }
    {  // KRvK, KQvK, KBBvK, KBNvK
      const bool theyHaveLonelyKing = (theirMen == theirKings) && (std::popcount(ourKnights) > 2 || std::popcount(ourBishops) > 1 || std::popcount(ourRooks) > 0 || std::popcount(ourQueens) > 0);
      const bool weHaveLonelyKing = (ourMen == ourKings) && (std::popcount(theirKnights) > 2 || std::popcount(theirBishops) > 1 || std::popcount(theirRooks) > 0 || std::popcount(theirQueens) > 0);

      features[EF::LONELY_KING_ON_EDGE_AND_NOT_DRAW] = value_or_zero(theyHaveLonelyKing, (3 - kDistToEdge[theirKingSq]))
                                                     - value_or_zero(  weHaveLonelyKing, (3 - kDistToEdge[ourKingSq]));
      features[EF::LONELY_KING_IN_CORNER_AND_NOT_DRAW] = value_or_zero(theyHaveLonelyKing, (3 - kDistToCorner[theirKingSq]))
                                                       - value_or_zero(  weHaveLonelyKing, (3 - kDistToCorner[ourKingSq]));

      int dx = std::abs(wx - bx);
      int dy = std::abs(wy - by);
      const bool opposition = (dx == 2 && dy == 0) || (dx == 0 && dy == 2);
      features[EF::LONELY_KING_OPPOSITION_AND_NOT_DRAW] = weHaveLonelyKing && opposition;
      features[EF::LONELY_KING_ACHIEVABLE_OPPOSITION_AND_NOT_DRAW] = theyHaveLonelyKing && (
        (dx == 3 && dy <= 1)
        || (dy == 3 && dx <= 1)
      );

      features[EF::LONELY_KING_NEXT_TO_ENEMY_KING] = value_or_zero(theyHaveLonelyKing, (8 - std::max(dx, dy)) * 2)
      - value_or_zero(  weHaveLonelyKing, (8 - std::max(dx, dy)) * 2)
      + value_or_zero(theyHaveLonelyKing, (8 - std::min(dx, dy)) * 1)
      - value_or_zero(  weHaveLonelyKing, (8 - std::min(dx, dy)) * 1);
    }

    {
      // https://www.chessprogramming.org/King_Pawn_Tropism
      // Penalize king for being far away from pawns. Positive score is *good* for the mover.
      // Note: passed pawns are implicility prioritized, since we consider distance to the passed
      //       pawn, and the two squares ahead of it (so they're weighted 3x).
      const Bitboard passedPawns = pawnAnalysis.ourPassedPawns | pawnAnalysis.theirPassedPawns;
      const Bitboard otherPawns = (ourPawns | theirPawns) & ~passedPawns;
      const Bitboard aheadOfPassedPawns = passedPawns | shift<kForward>(pawnAnalysis.ourPassedPawns) | shift<kBackward>(pawnAnalysis.theirPassedPawns)
      | shift<kForward2>(pawnAnalysis.ourPassedPawns) | shift<kBackward2>(pawnAnalysis.theirPassedPawns);
      features[EF::KING_TROPISM] = 1;  // Small bonus for your turn.
      for (int i = 0; i < 15; ++i) {
        features[EF::KING_TROPISM] += std::popcount(aheadOfPassedPawns & kManhattanDist[i][ourKingSq]);
        features[EF::KING_TROPISM] += std::popcount(otherPawns & kManhattanDist[i][ourKingSq]);

        features[EF::KING_TROPISM] -= std::popcount(kManhattanDist[i][theirKingSq] & aheadOfPassedPawns);
        features[EF::KING_TROPISM] -= std::popcount(kManhattanDist[i][theirKingSq] & otherPawns);
      }
      features[EF::KING_TROPISM] *= std::max(time - 12, 0);
      features[EF::KING_TROPISM] /= 2;
    }

    // Handle some typically drawn endgames. ELO_STDERR(+8, +25)
    bool isDrawn = false;
    const int numOurMen = std::popcount(ourMen);
    const int numTheirMen = std::popcount(theirMen);
    const int numOurMinors = std::popcount(ourMinors);
    const int numTheirMinors = std::popcount(theirMinors);
    if (numOurMen == 1 && numTheirMen == 2) {
      isDrawn |= (numTheirMinors == 1);
    }
    if (numOurMen == 2 && numTheirMen == 1) {
      isDrawn |= (numOurMinors == 1);
    }
    if (numOurMen == 2 && numTheirMen == 2) {
      isDrawn |= (numOurMinors) && (numTheirMinors == 1);
      isDrawn |= (std::popcount(ourBishops) == 1) && (std::popcount(theirPawns) == 1);
      isDrawn |= (std::popcount(ourPawns) == 1) && (std::popcount(theirBishops) == 1);
      isDrawn |= (std::popcount(ourKnights) == 1) && (std::popcount(theirPawns) == 1);
      isDrawn |= (std::popcount(ourPawns) == 1) && (std::popcount(theirKnights) == 1);
      isDrawn |= (std::popcount(ourRooks) == 1) && (std::popcount(theirRooks) == 1);
      isDrawn |= (std::popcount(ourRooks) == 1) && (theirKnights & kCenter16) && (kDistToEdge[theirKingSq] > 0);
      isDrawn |= (std::popcount(theirRooks) == 1) && (ourKnights & kCenter16) && (kDistToEdge[ourKingSq] > 0);
      isDrawn |= (std::popcount(ourRooks) == 1) && (theirBishops) && (kDistToEdge[theirKingSq] > 0);
      isDrawn |= (std::popcount(theirRooks) == 1) && (ourBishops) && (kDistToEdge[ourKingSq] > 0);
    }
    features[EF::KNOWN_DRAW] = isDrawn;

    if (features[EF::KNOWN_KPVK_DRAW] || isDrawn) {
      return 0;
    }

    // TODO: bonus for controlling squares ahead of your own pawns

    // Use larger integer to make arithmetic safe.
    int32_t early = this->early<US>(pos);
    const int32_t late = this->late<US>(pos);
    const int32_t ineq = this->ineq<US>(pos);

    // Piece-square tables ELO_STDERR(+20, +37)
    int32_t pieceMap = pos.pieceMapScores[PieceMapType::PieceMapTypeEarly] * (18 - time) + pos.pieceMapScores[PieceMapType::PieceMapTypeLate] * time;
    if (US == Color::BLACK) {
      pieceMap *= -1;
    }

    // ELO_STDERR(+0, +16)
    const int kSafetyDist = 2;
    int ourKingDanger = 0;
    {
      ourKingDanger += (threats.theirPawnTargets & kNearby[kSafetyDist][ourKingSq]) > 0;
      ourKingDanger += (threats.theirKnightTargets & kNearby[kSafetyDist][ourKingSq]) > 0;
      ourKingDanger += (threats.theirBishopTargets & kNearby[kSafetyDist][ourKingSq] & kBlackSquares) > 0;
      ourKingDanger += (threats.theirBishopTargets & kNearby[kSafetyDist][ourKingSq] & kWhiteSquares) > 0;
      ourKingDanger += (threats.theirRookTargets & kNearby[kSafetyDist][ourKingSq]) > 0;
      ourKingDanger += (threats.theirQueenTargets & kNearby[kSafetyDist][ourKingSq]) > 0;
      ourKingDanger += (threats.theirKingTargets & kNearby[kSafetyDist][ourKingSq]) > 0;

      ourKingDanger -= (threats.ourPawnTargets & kNearby[kSafetyDist][ourKingSq]) > 0;
      ourKingDanger -= (threats.ourKnightTargets & kNearby[kSafetyDist][ourKingSq]) > 0;
      ourKingDanger -= (threats.ourBishopTargets & kNearby[kSafetyDist][ourKingSq] & kWhiteSquares) > 0;
      ourKingDanger -= (threats.ourBishopTargets & kNearby[kSafetyDist][ourKingSq] & kBlackSquares) > 0;
      ourKingDanger -= (threats.ourRookTargets & kNearby[kSafetyDist][ourKingSq]) > 0;
      ourKingDanger -= (threats.ourQueenTargets & kNearby[kSafetyDist][ourKingSq]) > 0;

      ourKingDanger = std::max(0, ourKingDanger - 1);
    }
    int theirKingDanger = 0;
    {
      theirKingDanger += (threats.ourPawnTargets & kNearby[kSafetyDist][theirKingSq]) > 0;
      theirKingDanger += (threats.ourKnightTargets & kNearby[kSafetyDist][theirKingSq]) > 0;
      theirKingDanger += (threats.ourBishopTargets & kNearby[kSafetyDist][theirKingSq] & kBlackSquares) > 0;
      theirKingDanger += (threats.ourBishopTargets & kNearby[kSafetyDist][theirKingSq] & kWhiteSquares) > 0;
      theirKingDanger += (threats.ourRookTargets & kNearby[kSafetyDist][theirKingSq]) > 0;
      theirKingDanger += (threats.ourQueenTargets & kNearby[kSafetyDist][theirKingSq]) > 0;
      theirKingDanger += (threats.ourKingTargets & kNearby[kSafetyDist][theirKingSq]) > 0;

      theirKingDanger -= (threats.theirPawnTargets & kNearby[kSafetyDist][theirKingSq]) > 0;
      theirKingDanger -= (threats.theirKnightTargets & kNearby[kSafetyDist][theirKingSq]) > 0;
      theirKingDanger -= (threats.theirBishopTargets & kNearby[kSafetyDist][theirKingSq] & kWhiteSquares) > 0;
      theirKingDanger -= (threats.theirBishopTargets & kNearby[kSafetyDist][theirKingSq] & kBlackSquares) > 0;
      theirKingDanger -= (threats.theirRookTargets & kNearby[kSafetyDist][theirKingSq]) > 0;
      theirKingDanger -= (threats.theirQueenTargets & kNearby[kSafetyDist][theirKingSq]) > 0;

      theirKingDanger = std::max(0, theirKingDanger - 1);
    }
    early -= (ourKingDanger - theirKingDanger) * 15;

    int32_t eval = (early * (18 - time) + late * time + ineq * 18 + pieceMap) / 18;

    eval = std::min(int32_t(-kQLongestForcedMate), std::max(int32_t(kQLongestForcedMate), eval));

    #ifdef PRINT_LEAVES
      if (rand() % 10000 == 0) {
        std::string t = "";
        t += pos.fen() + "\n";
        t += std::to_string(features[0]);
        for (size_t i = 1; i < EF::NUM_EVAL_FEATURES; ++i) {
          t += " " + std::to_string(features[i]);
        }
        std::cout << t << std::endl;
      }
    #endif

    return eval;
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
    return r;
  }

  template<Color US>
  int32_t ineq(const Position& pos) const {
    int32_t r = ineqB;
    for (size_t i = 0; i < EF::NUM_EVAL_FEATURES; ++i) {
      r += features[i] * ineqW[i];
    }

    int32_t inequality = (features[EF::OUR_KNIGHTS] - features[EF::THEIR_KNIGHTS]) * 3
    + (features[EF::OUR_BISHOPS] - features[EF::THEIR_BISHOPS]) * 3
    + (features[EF::OUR_ROOKS] - features[EF::THEIR_ROOKS]) * 5
    + (features[EF::OUR_QUEENS] - features[EF::THEIR_QUEENS]) * 9;
    inequality = std::min<int32_t>(1, std::max<int32_t>(-1, inequality));

    return r * inequality;
  }

  void zero_() {
    for (size_t i = 0; i < EF::NUM_EVAL_FEATURES; ++i) {
      earlyW[i] = 0;
      lateW[i] = 0;
      ineqW[i] = 0;
    }
    earlyB = 0;
    lateB = 0;
    ineqB = 0;
  }

  Evaluation features[NUM_EVAL_FEATURES];
};

#if NNUE_EVAL
template<Color US>
Evaluation nnue_evaluate(const Position& pos) {
  #if SLOW
  Evaluation score = pos.network->slowforward();
  #else
  Evaluation score = pos.network->fastforward();
  #endif
  if (US == Color::BLACK) {
    score *= -1;
  }
  return std::min<Evaluation>(-kQLongestForcedMate, std::max(kQLongestForcedMate, score));
}
#endif

}  // namespace ChessEngine

#endif  // EVALUATOR_H
