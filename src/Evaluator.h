#ifndef EVALUATOR_H
#define EVALUATOR_H

#include <cstdint>

#include <string>
#include <algorithm>

#include "geometry.h"
#include "utils.h"
#include "Position.h"
#include "movegen.h"
#include "piece_maps.h"

namespace ChessEngine {

/*

Add a feature for how many friendly pawns are on your bishop's color

Add a feature for the mobility of each piece

*/

float sigmoid(float x) {
  return 1.0f / (1.0f + std::exp(-x));
}

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
  KING_ON_CENTER_FILE,
  KING_ACTIVE,
  THREATS_NEAR_KING_2,
  THREATS_NEAR_KING_3,

  OUR_PASSED_PAWNS,
  THEIR_PASSED_PAWNS,
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

  NUM_TARGET_SQUARES,
  TIME,

  PAWN_PM,
  KNIGHT_PM,
  BISHOP_PM,
  ROOK_PM,
  QUEEN_PM,
  KING_PM,

  KPVK_OPPOSITION,
  KPVK_IN_FRONT_OF_PAWN,
  KPVK_OFFENSIVE_KEY_SQUARES,
  KPVK_DEFENSIVE_KEY_SQUARES,
  SQUARE_RULE,

  ADVANCED_PAWNS_1,
  ADVANCED_PAWNS_2,
  OPEN_ROOKS,
  ROOKS_ON_THEIR_SIDE,
  KING_CASTLED,
  CASTLING_RIGHTS,

  KING_IN_FRONT_OF_PASSED_PAWN,
  KING_IN_FRONT_OF_PASSED_PAWN2,

  PAWN_V_LONELY_KING,
  KNIGHTS_V_LONELY_KING,
  BISHOPS_V_LONELY_KING,
  ROOK_V_LONELY_KING,
  QUEEN_V_LONELY_KING,

  OUR_MATERIAL_THREATS,
  THEIR_MATERIAL_THREATS,

  LONELY_KING_ON_EDGE,

  NUM_EVAL_FEATURES,
};

const int32_t kEarlyB0 = 3;
const int32_t kEarlyW0[84] = {
  82, 482, 472, 519,1001, -76,
-478,-464,-505,-1005, -52,  -2,
 -48,  62, -61, -12, -24,  38,
 -29,   9,  28,  12,  32, 227,
 -33,   6,   6,  76,   4,  21,
  19,  74, -20,  19,  15,  32,
  81,  24,   6,  63,   8,  13,
  13,   3, -97, -65,-104, -85,
  28, 197, 175,  15, 144, 154,
 336,   0,   3,  -4,  24,  22,
   7,  36,  36,1043,-206, 124,
-1190, 174,-409, 112,  51,  41,
  10,   3,   4,-115,-304,-850,
-566,  -1,-1801, 180,-106, 183,
};
const int32_t kLateB0 = -6;
const int32_t kLateW0[84] = {
 156, 291, 337, 613,1117,-147,
-284,-340,-616,-1079, -67, -39,
  29, -19,  12,   0,  28, -21,
   6, -33, -31, -14,  -9,  48,
  77,  40, -42, -60,   1,   4,
  26,  93, -10,   2,  40,  53,
 -81,  12,  44, -13,  43,  29,
  -4, -74, -58, -81, -17, 131,
  68,  55,  92, 324, 706,   5,
 -17,   0,  -2,   8,  -6,   3,
  -7,  -5,  -3,  72,  80,  34,
-232, 376, 202,  54, -27,  11,
   3,   2,  66, 127, 108, -26,
-209, -24,-431, 138,-106, -29,
};
const int32_t kClippedB0 = -6;
const int32_t kClippedW0[84] = {
  21, 156, 160, 245, 566, -48,
-127,-135,-225,-498,  21,  -7,
   3,  25,   1,  -1, -14, -23,
  -5, -15,  -1,   2,   4,-519,
  37,  14,  33,  15,   2,  -1,
   2,   2,  -2,   4, -12,   2,
  61,   3,   2, -14,   2,  -5,
  -1,  -6,  23,  21,   0,  17,
  25, 153, 207, 314, 537,  18,
  95,   0,   8,  -1,   5,   5,
   7,   1,   1,  26, -29,   8,
 -88,  44,1032, -11,  -2,  -2,
  -3,  -2,   7,  -4, -37,-224,
-174,-110,-528,  69,  55,  16,
};
const int32_t kScaleB0 = 9;
const int32_t kScaleW0[84] = {
  14,  -8,  -8,   3,   6,   7,
  -8, -17,  -9, -39,  -7,  -1,
   2,   2,   1,  -1,  14,   8,
  -1,  -1,  -5,   0,   1, -79,
  -5,   0,   1,   2,   0,  -3,
   1,  -3,  -1,   2,   1,  -1,
   9,   6,   0,   3,   2,   0,
   2,  -5,   9,   8,  -2,  -5,
   6,   0,  11, -29, -99,   6,
   3,   0,  -1,   0,   1,   2,
  -1,   1,   0,   0,  20,   5,
 -31, -15, 103,   7,   0,   3,
   1,   1,   0,   3,  -5, -12,
 -10, -25, -51,   6,  -4,  -5,
};


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
  "OUR_PASSED_PAWNS",
  "THEIR_PASSED_PAWNS",
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
  "NUM_TARGET_SQUARES",
  "TIME",
  "PAWN_PM",
  "KNIGHT_PM",
  "BISHOP_PM",
  "ROOK_PM",
  "QUEEN_PM",
  "KING_PM",
  "KPVK_OPPOSITION",
  "KPVK_IN_FRONT_OF_PAWN",
  "KPVK_OFFENSIVE_KEY_SQUARES",
  "KPVK_DEFENSIVE_KEY_SQUARES",
  "SQUARE_RULE",
  "ADVANCED_PAWNS_1",
  "ADVANCED_PAWNS_2",
  "OPEN_ROOKS",
  "ROOKS_ON_THEIR_SIDE",
  "KING_CASTLED",
  "CASTLING_RIGHTS",
  "KING_IN_FRONT_OF_PASSED_PAWN",
  "KING_IN_FRONT_OF_PASSED_PAWN2",
  "PAWN_V_LONELY_KING",
  "KNIGHTS_V_LONELY_KING",
  "BISHOPS_V_LONELY_KING",
  "ROOK_V_LONELY_KING",
  "QUEEN_V_LONELY_KING",
  "OUR_MATERIAL_THREATS",
  "THEIR_MATERIAL_THREATS",
  "LONELY_KING_ON_EDGE",
};

struct Evaluator {
  Evaluator() {}

  template<Color US>
  Evaluation score(const Position& pos) {
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

    const Bitboard ourMen = pos.colorBitboards_[US];
    const Bitboard theirMen = pos.colorBitboards_[THEM];
    const bool isThreeManEndgame = std::popcount(ourMen | theirMen) == 3;
    bool isDraw = false;
    isDraw |= (ourMen == ourKings) && (theirMen == theirKings);
    isDraw |= ((ourMen | theirMen) == (ourKings | ourKnights | theirKings | theirKnights)) && isThreeManEndgame;
    isDraw |= ((ourMen | theirMen) == (ourKings | ourBishops | theirKings | theirBishops)) && isThreeManEndgame;
    if (isDraw) {
      return 0;
    }

    const Bitboard ourRoyalty = ourQueens | ourKings;
    const Bitboard theirRoyalty = theirQueens | theirKings;
    const Bitboard ourMajors = ourRooks | ourRoyalty;
    const Bitboard theirMajors = theirRooks | theirRoyalty;
    const Bitboard ourMinors = ourKnights | ourBishops;
    const Bitboard theirMinors = theirKnights | theirBishops;
    const Bitboard ourPieces = ourMajors | ourMinors;
    const Bitboard theirPieces = theirMajors | theirMinors;

    const Bitboard ourPawnTargets = compute_pawn_targets<US>(pos);
    const Bitboard theirPawnTargets = compute_pawn_targets<THEM>(pos);
    const Bitboard ourKnightTargets = compute_knight_targets<US>(pos);
    const Bitboard theirKnightTargets = compute_knight_targets<THEM>(pos);
    const Bitboard usBishopTargets = compute_bishoplike_targets<US>(pos, ourBishops);
    const Bitboard theirBishopTargets = compute_bishoplike_targets<THEM>(pos, theirBishops);
    const Bitboard ourRookTargets = compute_rooklike_targets<US>(pos, ourRooks);
    const Bitboard theirRookTargets = compute_rooklike_targets<THEM>(pos, theirRooks);
    const Bitboard usQueenTargets = compute_bishoplike_targets<US>(pos, ourQueens) | compute_rooklike_targets<US>(pos, ourQueens);
    const Bitboard theirQueenTargets = compute_bishoplike_targets<THEM>(pos, theirQueens) | compute_rooklike_targets<THEM>(pos, theirQueens);
    const Bitboard usKingTargets = compute_king_targets<US>(pos, ourKingSq);
    const Bitboard theirKingTargets = compute_king_targets<THEM>(pos, theirKingSq);

    constexpr Bitboard kOurSide = (US == Color::WHITE ? kWhiteSide : kBlackSide);
    constexpr Bitboard kTheirSide = (US == Color::WHITE ? kBlackSide : kWhiteSide);
    constexpr Direction kForward = (US == Color::WHITE ? Direction::NORTH : Direction::SOUTH);
    constexpr Direction kBackward = opposite_dir(kForward);
    constexpr Bitboard kOurBackRanks = (US == Color::WHITE ? kRanks[6] | kRanks[7] : kRanks[1] | kRanks[0]);
    constexpr Bitboard kTheirBackRanks = (US == Color::WHITE ? kRanks[1] | kRanks[0] : kRanks[6] | kRanks[7]);
    constexpr Bitboard kHappyKingSquares = bb(62) | bb(58) | bb(57) | bb(6) | bb(1) | bb(2);

    const Bitboard usTargets = ourPawnTargets | ourKnightTargets | usBishopTargets | ourRookTargets | usQueenTargets | usKingTargets;
    const Bitboard themTargets = theirPawnTargets | theirKnightTargets | theirBishopTargets | theirRookTargets | theirQueenTargets | theirKingTargets;

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
    features[EF::KING_ON_CENTER_FILE] = (ourKingSq % 8 == 3 || ourKingSq % 8 == 4) - (theirKingSq % 8 == 3 || theirKingSq % 8 == 4);
    features[EF::THREATS_NEAR_KING_2] = std::popcount(kNearby[2][ourKingSq] & themTargets & ~usTargets) - std::popcount(kNearby[2][theirKingSq] & usTargets & ~themTargets);
    features[EF::THREATS_NEAR_KING_3] = std::popcount(kNearby[3][ourKingSq] & themTargets & ~usTargets) - std::popcount(kNearby[2][theirKingSq] & usTargets & ~themTargets);

    // Pawns
    const Bitboard theirBlockadedPawns = shift<kForward>(ourPawns) & theirPawns;
    const Bitboard outBlockadedPawns = shift<kBackward>(theirPawns) & ourPawns;
    const Bitboard ourProtectedPawns = ourPawns & ourPawnTargets;
    const Bitboard theirProtectedPawns = theirPawns & theirPawnTargets;
    Bitboard ourPassedPawns, theirPassedPawns;
    Bitboard filesWithoutOurPawns, filesWithoutTheirPawns;
    {
      Bitboard ourFilled, theirFilled;
      Bitboard filesWithOurPawns, filesWithTheirPawns;
      if (US == Color::WHITE) {
        ourFilled = northFill(ourPawns);
        theirFilled = southFill(theirPawns);
        filesWithOurPawns = southFill(ourFilled);
        filesWithTheirPawns = northFill(theirFilled);
      } else {
        ourFilled = southFill(ourPawns);
        theirFilled = northFill(theirPawns);
        filesWithOurPawns = northFill(ourFilled);
        filesWithTheirPawns = southFill(theirFilled);
      }
      const Bitboard aheadOfOurPawnsFat = fatten(ourFilled);
      const Bitboard aheadOfTheirPawnsFat = fatten(theirFilled);
      filesWithoutOurPawns = ~filesWithOurPawns;
      filesWithoutTheirPawns = ~filesWithTheirPawns;
      ourPassedPawns = ourPawns & ~shift<kBackward>(fatten(theirFilled));
      theirPassedPawns = theirPawns & ~shift<kForward>(fatten(ourFilled));
      const Bitboard ourIsolatedPawns = ourPawns & ~shift<Direction::WEST>(filesWithOurPawns) & ~shift<Direction::EAST>(filesWithOurPawns);
      const Bitboard theirIsolatedPawns = theirPawns & ~shift<Direction::WEST>(filesWithTheirPawns) & ~shift<Direction::EAST>(filesWithTheirPawns);
      const Bitboard ourDoubledPawns = ourPawns & shift<kForward>(ourFilled);
      const Bitboard theirDoubledPawns = theirPawns & shift<kBackward>(theirFilled);

      constexpr Bitboard kRookPawns = kFiles[0] | kFiles[7];

      features[EF::PAWNS_CENTER_16] = std::popcount(ourPawns & kCenter16) - std::popcount(theirPawns & kCenter16);
      features[EF::PAWNS_CENTER_16] = std::popcount(ourPawns & kCenter16) - std::popcount(theirPawns & kCenter16);
      features[EF::PAWNS_CENTER_4] = std::popcount(ourPawns & kCenter4) - std::popcount(theirPawns & kCenter4);
      features[EF::OUR_PASSED_PAWNS] = std::popcount(ourPassedPawns);
      features[EF::THEIR_PASSED_PAWNS] = std::popcount(theirPassedPawns);
      features[EF::ISOLATED_PAWNS] = std::popcount(ourIsolatedPawns) - std::popcount(theirIsolatedPawns);
      features[EF::DOUBLED_PAWNS] = std::popcount(ourDoubledPawns) - std::popcount(theirDoubledPawns);
      features[EF::DOUBLE_ISOLATED_PAWNS] = std::popcount(ourDoubledPawns & ourIsolatedPawns) - std::popcount(theirDoubledPawns & theirIsolatedPawns);
      features[EF::ADVANCED_PAWNS_1] = std::popcount(ourPawns & kTheirBackRanks) - std::popcount(theirPawns & kOurBackRanks);
      features[EF::ADVANCED_PAWNS_2] = std::popcount(ourPawns & shift<kBackward>(kTheirBackRanks)) - std::popcount(theirPawns & shift<kForward>(kOurBackRanks));

      if (US == Color::WHITE) {
        features[EF::ADVANCED_PASSED_PAWNS_2] = std::popcount(ourPassedPawns & kRanks[1]) * 2 - std::popcount(theirPassedPawns & kRanks[6]);
        features[EF::ADVANCED_PASSED_PAWNS_3] = std::popcount(ourPassedPawns & kRanks[2]) * 2 - std::popcount(theirPassedPawns & kRanks[5]);
        features[EF::ADVANCED_PASSED_PAWNS_4] = std::popcount(ourPassedPawns & kRanks[3]) * 2 - std::popcount(theirPassedPawns & kRanks[4]);
      } else {
        features[EF::ADVANCED_PASSED_PAWNS_2] = std::popcount(ourPassedPawns & kRanks[6]) * 2 - std::popcount(theirPassedPawns & kRanks[1]);
        features[EF::ADVANCED_PASSED_PAWNS_3] = std::popcount(ourPassedPawns & kRanks[5]) * 2 - std::popcount(theirPassedPawns & kRanks[2]);
        features[EF::ADVANCED_PASSED_PAWNS_4] = std::popcount(ourPassedPawns & kRanks[4]) * 2 - std::popcount(theirPassedPawns & kRanks[3]);
      }

      features[EF::PAWN_MINOR_CAPTURES] = std::popcount(ourPawnTargets & theirMinors) * 3 - std::popcount(theirPawnTargets & ourMinors);
      features[EF::PAWN_MAJOR_CAPTURES] = std::popcount(ourPawnTargets & theirMajors) * 3 - std::popcount(theirPawnTargets & ourMajors);
      features[EF::PROTECTED_PAWNS] = std::popcount(ourPawns & ourPawnTargets) - std::popcount(theirPawns & theirPawnTargets);
      features[EF::PROTECTED_PASSED_PAWNS] = std::popcount(ourPassedPawns & ourPawnTargets) * 2 - std::popcount(theirPassedPawns & theirPawnTargets);
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
      features[EF::SCARY_BISHOPS] = std::popcount(ourBishopTargetsIgnoringNonBlockades & theirMajors) - std::popcount(theirBishopTargetsIgnoringNonBlockades & ourMajors);
      features[EF::SCARIER_BISHOPS] = std::popcount(ourBishopTargetsIgnoringNonBlockades & theirRoyalty) - std::popcount(theirBishopTargetsIgnoringNonBlockades & ourRoyalty);

      // const Bitboard ourPawnsOnUs = kUsSquares & ourPawns;
      // const Bitboard ourPawnsOnThem = kThemSquares & ourPawns;
      // const Bitboard theirPawnsOnUs = kUsSquares & theirPawns;
      // const Bitboard theirPawnsOnThem = kThemSquares & theirPawns;
    }

    {  // Rooks
      const Bitboard openFiles = filesWithoutOurPawns & filesWithoutTheirPawns;
      features[EF::BLOCKADED_ROOKS] = std::popcount(ourRooks & filesWithoutOurPawns) - std::popcount(theirRooks & filesWithoutTheirPawns);
      features[EF::SCARY_ROOKS] = std::popcount(ourRookTargets & theirRoyalty) - std::popcount(theirRookTargets & ourRoyalty);
      features[EF::INFILTRATING_ROOKS] = std::popcount(ourRooks & kTheirBackRanks) - std::popcount(theirRooks & kOurBackRanks);
      features[EF::OPEN_ROOKS] = std::popcount(ourRooks & openFiles) - std::popcount(theirRooks & openFiles);
      features[EF::ROOKS_ON_THEIR_SIDE] = std::popcount(ourRooks & kTheirSide) - std::popcount(theirRooks & kOurSide);
    }

    {  // Knights
      if (US == Color::WHITE) {
        features[EF::KNIGHTS_DEVELOPED] = std::popcount(theirKnights & (bb( 1) | bb( 6))) - std::popcount(ourKnights & (bb(57) | bb(62)));
      } else {
        features[EF::KNIGHTS_DEVELOPED] = std::popcount(theirKnights & (bb(57) | bb(62))) - std::popcount(ourKnights & (bb( 1) | bb( 6)));
      }
      features[EF::KNIGHT_MAJOR_CAPTURES] = std::popcount(ourKnightTargets & theirMajors) * 2 - std::popcount(theirKnightTargets & ourMajors);
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

      const int wx = ourKingSq % 8;
      const int wy = ourKingSq / 8;
      const int bx = theirKingSq % 8;
      const int by = theirKingSq / 8;
      const int kingsDist = std::max(std::abs(wx - bx), std::abs(wy - by));
      features[EF::LONELY_KING_IN_CENTER] = kDistToCorner[theirKingSq] * (std::popcount(pos.colorBitboards_[THEM]) == 1);
      features[EF::LONELY_KING_IN_CENTER] -= kDistToCorner[ourKingSq] * (std::popcount(pos.colorBitboards_[US]) == 1);

      features[EF::LONELY_KING_AWAY_FROM_ENEMY_KING] = (std::popcount(pos.colorBitboards_[THEM]) == 1) * kingsDist;
      features[EF::LONELY_KING_AWAY_FROM_ENEMY_KING] -= (std::popcount(pos.colorBitboards_[US]) == 1) * kingsDist;

      const CastlingRights cr = pos.currentState_.castlingRights;
      features[EF::CASTLING_RIGHTS] = ((cr & kCastlingRights_WhiteKing) > 0);
      features[EF::CASTLING_RIGHTS] += ((cr & kCastlingRights_WhiteQueen) > 0);
      features[EF::CASTLING_RIGHTS] -= ((cr & kCastlingRights_BlackKing) > 0);
      features[EF::CASTLING_RIGHTS] -= ((cr & kCastlingRights_BlackQueen) > 0);
      features[EF::KING_CASTLED] = std::popcount(ourKings & kHappyKingSquares) - std::popcount(theirKings & kHappyKingSquares);

      features[EF::NUM_TARGET_SQUARES] = std::popcount(usTargets) * 2 - std::popcount(themTargets);
    }

    {  // Piece map values.
      std::fill_n(&features[EF::PAWN_PM], 6, 0);
      for (ColoredPiece cp = ColoredPiece::WHITE_PAWN; cp < ColoredPiece::NUM_COLORED_PIECES; cp = ColoredPiece(cp + 1)) {
        Piece piece = cp2p(cp);
        EF feature = EF(EF::PAWN_PM + piece - 1);
        Bitboard bitmap = pos.pieceBitboards_[cp];
        while (bitmap) {
          int delta = kPieceMap[(cp - 1) * 64 + pop_lsb(bitmap)];
          features[feature] += delta;
        }
      }
      if (US == Color::BLACK) {
        features[EF::PAWN_PM] *= -1;
        features[EF::KNIGHT_PM] *= -1;
        features[EF::BISHOP_PM] *= -1;
        features[EF::ROOK_PM] *= -1;
        features[EF::QUEEN_PM] *= -1;
        features[EF::KING_PM] *= -1;
      }
    }

    const bool isKingPawnEndgame = (ourKings == ourPieces) && (theirKings == theirPieces);
    features[EF::KPVK_OPPOSITION] = -((shift<kForward>(shift<kForward>(ourKings)) & theirKings) > 0) * isKingPawnEndgame;
    features[EF::KPVK_IN_FRONT_OF_PAWN] = 0;
    features[EF::KPVK_OFFENSIVE_KEY_SQUARES] = 0;
    features[EF::KPVK_DEFENSIVE_KEY_SQUARES] = 0;

    features[EF::SQUARE_RULE] = 0;
    features[EF::SQUARE_RULE] += ((kSquareRuleTheirTurn[US][theirKingSq] & ourPassedPawns) > 0) * isKingPawnEndgame;
    features[EF::SQUARE_RULE] -= ((kSquareRuleYourTurn[THEM][ourKingSq] & theirPassedPawns) > 0) * isKingPawnEndgame;

    {  // If we have the pawn in a KPVK engame.
      bool isKPVK = isKingPawnEndgame && (std::popcount(ourPawns) == 1) && (theirPawns == 0);

      const Bitboard inFrontOfPawn = shift<kForward>(ourPawns);
      const Bitboard keySquares = fatten(shift<kForward>(inFrontOfPawn & ~kRookFiles));
      const Bitboard inFront = (US == Color::WHITE ? (ourPawns - 1) : ~(ourPawns - 1));
      const Square promoSq = Square(US == Color::WHITE ? lsb(ourPawns) % 8 : lsb(ourPawns) % 8 + 56);

      features[EF::KPVK_IN_FRONT_OF_PAWN] += ((ourKings & inFront) > 0) * isKPVK;
      features[EF::KPVK_OFFENSIVE_KEY_SQUARES] += ((ourKings & keySquares) > 0) * isKPVK;
      features[EF::KPVK_DEFENSIVE_KEY_SQUARES] += ((theirKings & inFrontOfPawn) > 0) * isKPVK;
    }
    {  // If they have the pawn in a KPVK engame. Note we add a '+1' penalty for square rule
      bool isKPVK = isKingPawnEndgame && (std::popcount(theirPawns) == 1) && (ourPawns == 0);

      const Bitboard inFrontOfPawn = shift<kBackward>(theirPawns);
      const Bitboard keySquares = fatten(shift<kBackward>(inFrontOfPawn & ~kRookFiles));
      const Bitboard inFront = (US == Color::WHITE ? ~(theirPawns - 1) : (theirPawns - 1));
      const Square promoSq = Square(US == Color::WHITE ? lsb(theirPawns) % 8 + 56 : lsb(theirPawns) % 8);

      features[EF::KPVK_IN_FRONT_OF_PAWN] -= ((theirKings & inFront) > 0) * isKPVK;
      features[EF::KPVK_OFFENSIVE_KEY_SQUARES] -= ((theirKings & keySquares) > 0) * isKPVK;
      features[EF::KPVK_DEFENSIVE_KEY_SQUARES] -= ((ourKings & inFrontOfPawn) > 0) * isKPVK;
    }

    Bitboard aheadOfOurPassedPawnsFat, aheadOfTheirPassedPawnsFat;
    if (US == Color::WHITE) {
      aheadOfOurPassedPawnsFat = fatten(northFill(ourPassedPawns));
      aheadOfTheirPassedPawnsFat = fatten(southFill(theirPassedPawns));
    } else {
      aheadOfOurPassedPawnsFat = fatten(southFill(ourPassedPawns));
      aheadOfTheirPassedPawnsFat = fatten(northFill(theirPassedPawns));
    }

    // We split these into two features, the idea being that being ahead of your pawns while your opponent's
    // queen is on the board is dangerous, but being ahead of your opponent's passed pawns is not.
    features[EF::KING_IN_FRONT_OF_PASSED_PAWN] = ((ourKings & aheadOfOurPassedPawnsFat) > 0 && theirQueens == 0);
    features[EF::KING_IN_FRONT_OF_PASSED_PAWN] -= ((theirKings & aheadOfTheirPassedPawnsFat) > 0 && ourQueens == 0);
    features[EF::KING_IN_FRONT_OF_PASSED_PAWN2] = (ourKings & aheadOfTheirPassedPawnsFat) > 0;
    features[EF::KING_IN_FRONT_OF_PASSED_PAWN2] -= (theirKings & aheadOfOurPassedPawnsFat) > 0;

    const bool isOurKingLonely = (ourMen == ourKings);
    const bool isTheirKingLonely = (theirMen == theirKings);

    // Bonus vs lonely king.
    features[EF::PAWN_V_LONELY_KING] = isTheirKingLonely * std::popcount(ourPawns);
    features[EF::PAWN_V_LONELY_KING] -= isOurKingLonely * std::popcount(theirPawns);
    features[EF::KNIGHTS_V_LONELY_KING] = isTheirKingLonely * std::popcount(ourKnights);
    features[EF::KNIGHTS_V_LONELY_KING] -= isOurKingLonely * std::popcount(theirKnights);
    features[EF::BISHOPS_V_LONELY_KING] = isTheirKingLonely * std::popcount(ourBishops);
    features[EF::BISHOPS_V_LONELY_KING] -= isOurKingLonely * std::popcount(theirBishops);
    features[EF::ROOK_V_LONELY_KING] = isTheirKingLonely * std::popcount(ourRooks);
    features[EF::ROOK_V_LONELY_KING] -= isOurKingLonely * std::popcount(theirRooks);
    features[EF::QUEEN_V_LONELY_KING] = isTheirKingLonely * std::popcount(ourQueens);
    features[EF::QUEEN_V_LONELY_KING] -= isOurKingLonely * std::popcount(theirQueens);

    features[EF::LONELY_KING_ON_EDGE] = kDistToEdge[theirKingSq] * isTheirKingLonely;
    features[EF::LONELY_KING_ON_EDGE] -= kDistToEdge[ourKingSq] * isOurKingLonely;

    {
      Bitboard ourMaterialThreats = 0;
      ourMaterialThreats |= ourPawnTargets & theirPieces;
      ourMaterialThreats |= ourKnightTargets & theirMajors;
      ourMaterialThreats |= usBishopTargets & theirMajors;
      ourMaterialThreats |= ourRookTargets & theirRoyalty;
      ourMaterialThreats |= usQueenTargets & theirKings;

      Bitboard theirMaterialThreats = 0;
      theirMaterialThreats |= theirPawnTargets & ourPieces;
      theirMaterialThreats |= theirKnightTargets & ourMajors;
      theirMaterialThreats |= theirBishopTargets & ourMajors;
      theirMaterialThreats |= theirRookTargets & ourRoyalty;
      theirMaterialThreats |= theirQueenTargets & ourKings;

      features[EF::OUR_MATERIAL_THREATS] = std::popcount(ourMaterialThreats);
      features[EF::THEIR_MATERIAL_THREATS] = std::popcount(theirMaterialThreats);
    }

    const int16_t ourPiecesRemaining = std::popcount(pos.colorBitboards_[US] & ~ourPawns) + std::popcount(ourQueens) * 2 - 1;
    const int16_t theirPiecesRemaining = std::popcount(pos.colorBitboards_[THEM] & ~theirPawns) + std::popcount(theirQueens) * 2 - 1;
    const int32_t time = 18 - (ourPiecesRemaining + theirPiecesRemaining);

    features[EF::TIME] = time;

    // Use larger integer to make arithmetic safe.
    const int32_t early = this->early<US>(pos);
    const int32_t late = this->late<US>(pos);
    const int32_t clipped = this->clipped<US>(pos);
    const float scaled = 0.2 + sigmoid(float(this->scaled<US>(pos)) / 100.0);

    const int32_t eval = (early * (18 - time) + late * time) / 18 + clipped;

    int32_t e = int32_t(std::round(eval * scaled));
    e = std::min(int32_t(-kLongestForcedMate), std::max(int32_t(kLongestForcedMate), e));

    return e;
  }

  template<Color US>
  Evaluation early(const Position& pos) const {
    int32_t r = kEarlyB0;
    for (size_t i = 0; i < EF::NUM_EVAL_FEATURES; ++i) {
      r += features[i] * kEarlyW0[i];
    }
    return r;
  }

  template<Color US>
  Evaluation late(const Position& pos) const {
    int32_t r = kLateB0;
    for (size_t i = 0; i < EF::NUM_EVAL_FEATURES; ++i) {
      r += features[i] * kLateW0[i];
    }
    return r;
  }

  template<Color US>
  Evaluation scaled(const Position& pos) const {
    int32_t r = kScaleB0;
    for (size_t i = 0; i < EF::NUM_EVAL_FEATURES; ++i) {
      r += features[i] * kScaleW0[i];
    }
    return r;
  }

  template<Color US>
  Evaluation clipped(const Position& pos) const {
    int32_t r = kClippedB0;
    for (size_t i = 0; i < EF::NUM_EVAL_FEATURES; ++i) {
      r += features[i] * kClippedW0[i];
    }
    return std::max(-100, std::min(100, r));
  }

  // template<Color US>
  // int32_t early(const Position& pos) const {
  //   float A[8];
  //   for (size_t i = 0; i < 8; ++i) {
  //     A[i] = kEarlyB0[i];
  //     for (size_t j = 0; j < EF::NUM_EVAL_FEATURES; ++j) {
  //       A[i] += features[j] * kEarlyW0[i * EF::NUM_EVAL_FEATURES + j];
  //     }
  //     A[i] = (std::max(A[i], 0.0f) + std::min(A[i], 0.0f) / 100) / 100;
  //   }
  //   float r = kEarlyB1;
  //   for (size_t i = 0; i < 8; ++i) {
  //     r += A[i] * kEarlyW1[i];
  //   }
  //   return r;
  // }

  // template<Color US>
  // int32_t late(const Position& pos) const {
  //   float A[8];
  //   for (size_t i = 0; i < 8; ++i) {
  //     A[i] = kLateB0[i];
  //     for (size_t j = 0; j < EF::NUM_EVAL_FEATURES; ++j) {
  //       A[i] += features[j] * kLateW0[i * EF::NUM_EVAL_FEATURES + j];
  //     }
  //     A[i] = (std::max(A[i], 0.0f) + std::min(A[i], 0.0f) / 100) / 100;
  //   }
  //   float r = kLateB1;
  //   for (size_t i = 0; i < 8; ++i) {
  //     r += A[i] * kLateW1[i];
  //   }
  //   return r;
  // }

  Evaluation features[NUM_EVAL_FEATURES];
};

}  // namespace ChessEngine

#endif  // EVALUATOR_H
