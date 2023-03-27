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

  NUM_EVAL_FEATURES,
};

const int32_t kEarlyB0 = 6;
const int32_t kEarlyW0[EF::NUM_EVAL_FEATURES] = {
  11, 218, 221, 226, 511, -38,
-198,-198,-208,-459,-346,  20,
 -36, -50, -19,  -3, -29,  25,
  -6,  -2,   1,   4,  16,  -4,
   0,   7,  66, 130,   2,   8,
  18,  29, -10,  10,   2,  20,
   7,  10,  13,  48,  13,   2,
  16,  13, -16,   5,  -1,  -5,
   2,  44,  21,   1,   6,-358,
 -88,   0,  11,4182,2743, 324,
3386,1967, -74,  52,  23,   9,
  11,   2, -53, -75,   0,   0,
   0,   0,   0,  20,   5,-204,
   5,   9,   9,   1,   2,   1,
  -1,   0,   0,  -1,  -1,   2,
  12,   0,   3, -11, -16, -17,
   8,   6,  67,  50,   2,   9,
};
const int32_t kLateB0 = 0;
const int32_t kLateW0[EF::NUM_EVAL_FEATURES] = {
  57, 111, 141, 310, 540, -66,
-110,-142,-308,-474,  27, -37,
  14,   9,   4,   0,  16, -22,
  -2, -13,  -6,  -9, -14,-119,
  52,  21,  39,  72,   3,   6,
  -8,  23,  -9,  33,   4,  13,
   1,  -5,  -5,  33, -22,   9,
  -3, -17, -22, -35, -16,  11,
  15,  55,  57,  25,  18,-192,
  64,   0,   5,  51,1545, 325,
 771,  28, 196,  10,  -7,  36,
   5,   0,  42,  60,   0,   0,
   0,   0,   0,  -1,  25, 208,
   5,   7,  -3,  10,   3,   2,
   1,   3,   0,   1,  -5,   5,
  -4,   6,  -8,  -6, -29,  18,
  -6,   2,  94,  60,  -2,  46,
};
const int32_t kClippedB0 = 7;
const int32_t kClippedW0[EF::NUM_EVAL_FEATURES] = {
  38, 111, 118, 192, 355, -25,
-113,-124,-197,-379,-799,  -8,
   4,  19,   0,   0,   4,   0,
  -3,   1,  -6,  -1,   2, 181,
   0,   6,   1, -44,   1,   3,
   3,   9,  -4,   8,  -7,   7,
   0,  10,   4,  38,  -2,   4,
   1, -10,  -5,  -7,  -2,   0,
   9, -19, -10,  -6,  -5,-282,
 -23,   0,  -2,1699,1913,-363,
 907, 262, -66,   9,   1,  10,
   4,  -2,   2,   0,   0,   0,
   0,   0,   0,  -5,  -4, 325,
   2,   1,   3,   3,   1,   0,
   1,   2,   2,   2,   0,   0,
   2,  -1,  -1,   4,   0,   0,
  -3,   6, -23, -15,  -8, -10,
};
const int32_t kLonelyKingB0 = -3;
const int32_t kLonelyKingW0[EF::NUM_EVAL_FEATURES] = {
  39,  68,  94, 222, 132, -40,
 -64, -92,-222, -77,-778,  32,
  -3,  -7,   0,   2,  -8,  -4,
  12, -20, -32, -12,   4, -75,
  25,   7,  76,  41, -11,  -9,
 -34, -66,  37,  63, -36,  15,
  44,  17, -56, 120,  11,  -2,
 -20, -43, -29,   7, -56, -29,
  36,  74, 104,  -5, 226, 200,
 -48,   0,   1, -93,-1692,  42,
-845,  87, 130,   6,   8,  31,
  -5, -18,   2,   8,   0,   0,
   0,   0,   0, -38,  52,-222,
 -28, -28,   9,   7,   5,  -4,
   4,   4, -10,  -8, -10,  -5,
 -11,-131, -12,  10,  35, 117,
 -58, -25,  -9,  21,-181,-496,
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

    const Bitboard ourRooklikePieces = ourRooks | ourQueens;
    const Bitboard theirRooklikePieces = theirRooks | theirQueens;
    const Bitboard ourBishoplikePieces = ourBishops | ourQueens;
    const Bitboard theirBishoplikePieces = theirBishops | theirQueens;

    const Bitboard ourPawnTargets = compute_pawn_targets<US>(pos);
    const Bitboard theirPawnTargets = compute_pawn_targets<THEM>(pos);
    const Bitboard ourKnightTargets = compute_knight_targets<US>(pos);
    const Bitboard theirKnightTargets = compute_knight_targets<THEM>(pos);
    // TODO: bishops can attack one square through our own pawns.
    const Bitboard ourBishopTargets = compute_bishoplike_targets<US>(pos, ourBishops, (ourPieces & ~ourBishoplikePieces) | theirPieces);
    const Bitboard theirBishopTargets = compute_bishoplike_targets<THEM>(pos, theirBishops, (theirPieces & ~theirBishoplikePieces) | ourPieces);
    const Bitboard ourRookTargets = compute_rooklike_targets<US>(pos, ourRooks, (ourPieces & ~ourRooklikePieces) | theirPieces);
    const Bitboard theirRookTargets = compute_rooklike_targets<THEM>(pos, theirRooks, ourPieces | (theirPieces & ~theirRooklikePieces));
    const Bitboard ourQueenTargets = compute_bishoplike_targets<US>(pos, ourQueens, (ourPieces & ~ourBishoplikePieces) | theirPieces)
    | compute_rooklike_targets<US>(pos, ourQueens, (ourPieces & ~ourRooklikePieces) | theirPieces);
    const Bitboard theirQueenTargets = compute_bishoplike_targets<THEM>(pos, theirQueens, (theirPieces & ~theirBishoplikePieces) | ourPieces)
    | compute_rooklike_targets<THEM>(pos, theirQueens, (theirPieces & ~theirRooklikePieces) | ourPieces);
    const Bitboard ourKingTargets = compute_king_targets<US>(pos, ourKingSq);
    const Bitboard theirKingTargets = compute_king_targets<THEM>(pos, theirKingSq);

    constexpr Bitboard kOurSide = (US == Color::WHITE ? kWhiteSide : kBlackSide);
    constexpr Bitboard kTheirSide = (US == Color::WHITE ? kBlackSide : kWhiteSide);
    constexpr Direction kForward = (US == Color::WHITE ? Direction::NORTH : Direction::SOUTH);
    constexpr Direction kBackward = opposite_dir(kForward);
    constexpr Bitboard kOurBackRanks = (US == Color::WHITE ? kRanks[6] | kRanks[7] : kRanks[1] | kRanks[0]);
    constexpr Bitboard kTheirBackRanks = (US == Color::WHITE ? kRanks[1] | kRanks[0] : kRanks[6] | kRanks[7]);
    constexpr Bitboard kHappyKingSquares = bb(62) | bb(58) | bb(57) | bb(6) | bb(1) | bb(2);

    Bitboard ourTargets = ourPawnTargets;
    Bitboard ourDoubleTargets = 0;
    Bitboard theirTargets = theirPawnTargets;
    Bitboard theirDoubleTargets = 0;
    { // Compute the above four variables.
      // Note: "ourDoubleTargets" and "theirDoubleTargets" are approximations, since
      // they ignore the possibility that two of the same piece can attack a square.
      ourDoubleTargets |= ourTargets & ourKnightTargets;
      ourTargets |= ourKnightTargets;
      theirDoubleTargets |= theirTargets & theirKnightTargets;
      theirTargets |= theirKnightTargets;

      ourDoubleTargets |= ourTargets & ourBishopTargets;
      ourTargets |= ourBishopTargets;
      theirDoubleTargets |= theirTargets & theirBishopTargets;
      theirTargets |= theirBishopTargets;

      ourDoubleTargets |= ourTargets & ourRookTargets;
      ourTargets |= ourRookTargets;
      theirDoubleTargets |= theirTargets & theirRookTargets;
      theirTargets |= theirRookTargets;

      ourDoubleTargets |= ourTargets & ourQueenTargets;
      ourTargets |= ourQueenTargets;
      theirDoubleTargets |= theirTargets & theirQueenTargets;
      theirTargets |= theirQueenTargets;

      ourDoubleTargets |= ourTargets & ourKingTargets;
      ourTargets |= ourKingTargets;
      theirDoubleTargets |= theirTargets & theirKingTargets;
      theirTargets |= theirKingTargets;
    }

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
    features[EF::THREATS_NEAR_KING_2] = std::popcount(kNearby[2][ourKingSq] & theirTargets & ~ourTargets) - std::popcount(kNearby[2][theirKingSq] & ourTargets & ~theirTargets);
    features[EF::THREATS_NEAR_KING_3] = std::popcount(kNearby[3][ourKingSq] & theirTargets & ~ourTargets) - std::popcount(kNearby[2][theirKingSq] & ourTargets & ~theirTargets);

    // Pawns
    const Bitboard ourBlockadedPawns = shift<kBackward>(theirPawns) & ourPawns;
    const Bitboard theirBlockadedPawns = shift<kForward>(ourPawns) & theirPawns;
    const Bitboard ourProtectedPawns = ourPawns & ourPawnTargets;
    const Bitboard theirProtectedPawns = theirPawns & theirPawnTargets;
    Bitboard ourPassedPawns, theirPassedPawns;
    Bitboard filesWithoutOurPawns, filesWithoutTheirPawns;
    Bitboard possibleOutpostsForUs, possibleOutpostsForThem;
    {
      Bitboard aheadOfOurPawns, aheadOfTheirPawns;
      Bitboard filesWithOurPawns, filesWithTheirPawns;
      if (US == Color::WHITE) {
        aheadOfOurPawns = northFill(ourPawns);
        aheadOfTheirPawns = southFill(theirPawns);
        filesWithOurPawns = southFill(aheadOfOurPawns);
        filesWithTheirPawns = northFill(aheadOfTheirPawns);
      } else {
        aheadOfOurPawns = southFill(ourPawns);
        aheadOfTheirPawns = northFill(theirPawns);
        filesWithOurPawns = northFill(aheadOfOurPawns);
        filesWithTheirPawns = southFill(aheadOfTheirPawns);
      }
      const Bitboard aheadOfOurPawnsFat = fatten(aheadOfOurPawns);
      const Bitboard aheadOfTheirPawnsFat = fatten(aheadOfTheirPawns);
      filesWithoutOurPawns = ~filesWithOurPawns;
      filesWithoutTheirPawns = ~filesWithTheirPawns;
      ourPassedPawns = ourPawns & ~shift<kBackward>(fatten(aheadOfTheirPawns));
      theirPassedPawns = theirPawns & ~shift<kForward>(fatten(aheadOfOurPawns));
      const Bitboard ourIsolatedPawns = ourPawns & ~shift<Direction::WEST>(filesWithOurPawns) & ~shift<Direction::EAST>(filesWithOurPawns);
      const Bitboard theirIsolatedPawns = theirPawns & ~shift<Direction::WEST>(filesWithTheirPawns) & ~shift<Direction::EAST>(filesWithTheirPawns);
      const Bitboard ourDoubledPawns = ourPawns & shift<kForward>(aheadOfOurPawns);
      const Bitboard theirDoubledPawns = theirPawns & shift<kBackward>(aheadOfTheirPawns);

      possibleOutpostsForUs = ~(
        shift<Direction::EAST>(shift<kBackward>(aheadOfTheirPawns))
        | shift<Direction::WEST>(shift<kBackward>(aheadOfTheirPawns))
      ) & ~kTheirBackRanks;
      possibleOutpostsForThem = ~(
        shift<Direction::EAST>(shift<kForward>(aheadOfOurPawns))
        | shift<Direction::WEST>(shift<kForward>(aheadOfOurPawns))
      ) & ~kOurBackRanks;

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

      features[EF::PAWN_MINOR_CAPTURES] = std::popcount(ourPawnTargets & theirMinors) - std::popcount(theirPawnTargets & ourMinors);
      features[EF::PAWN_MAJOR_CAPTURES] = std::popcount(ourPawnTargets & theirMajors) - std::popcount(theirPawnTargets & ourMajors);
      features[EF::PROTECTED_PAWNS] = std::popcount(ourPawns & ourPawnTargets) - std::popcount(theirPawns & theirPawnTargets);
      features[EF::PROTECTED_PASSED_PAWNS] = std::popcount(ourPassedPawns & ourPawnTargets) - std::popcount(theirPassedPawns & theirPawnTargets);
    }

    const Bitboard ourBishopTargetsIgnoringNonBlockades = compute_bishoplike_targets<US>(pos, ourBishops, ourBlockadedPawns);
    const Bitboard theirBishopTargetsIgnoringNonBlockades = compute_bishoplike_targets<THEM>(pos, theirBishops, theirBlockadedPawns);
    {  // Bishops
      if (US == Color::WHITE) {
        features[EF::BISHOPS_DEVELOPED] = std::popcount(theirBishops & (bb( 2) | bb( 5))) - std::popcount(ourBishops & (bb(58) | bb(61)));
      } else {
        features[EF::BISHOPS_DEVELOPED] = std::popcount(theirBishops & (bb(58) | bb(61))) - std::popcount(ourBishops & (bb( 2) | bb( 5)));
      }
      features[EF::BISHOP_PAIR] = (std::popcount(ourBishops) >= 2) - (std::popcount(theirBishops) >= 2);
      features[EF::BLOCKADED_BISHOPS] = std::popcount(ourBishopTargetsIgnoringNonBlockades & (ourBlockadedPawns | theirProtectedPawns)) - std::popcount(theirBishopTargetsIgnoringNonBlockades & (theirBlockadedPawns | ourProtectedPawns));
      features[EF::SCARY_BISHOPS] = std::popcount(ourBishopTargetsIgnoringNonBlockades & theirMajors) - std::popcount(theirBishopTargetsIgnoringNonBlockades & ourMajors);
      features[EF::SCARIER_BISHOPS] = std::popcount(ourBishopTargetsIgnoringNonBlockades & theirRoyalty) - std::popcount(theirBishopTargetsIgnoringNonBlockades & ourRoyalty);
      features[EF::OUTPOSTED_BISHOPS] = std::popcount(ourBishops & possibleOutpostsForUs) - std::popcount(theirBishops & possibleOutpostsForThem);

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
      features[EF::KNIGHT_MAJOR_CAPTURES] = std::popcount(ourKnightTargets & theirMajors) - std::popcount(theirKnightTargets & ourMajors);
      features[EF::KNIGHTS_CENTER_16] = std::popcount(ourKnights & kCenter16) - std::popcount(theirKnights & kCenter16);
      features[EF::KNIGHTS_CENTER_4] = std::popcount(ourKnights & kCenter4) - std::popcount(theirKnights & kCenter4);
      features[EF::KNIGHT_ON_ENEMY_SIDE] = std::popcount(ourKnights & kTheirSide) - std::popcount(theirKnights & kOurSide);
      features[EF::OUTPOSTED_KNIGHTS] = std::popcount(ourKnights & possibleOutpostsForUs) - std::popcount(theirKnights & possibleOutpostsForThem);

      features[EF::BISHOPS_BLOCKING_KNIGHTS] = std::popcount(shift<kForward>(shift<kForward>(shift<kForward>(theirKnights))) & ourBishops) - std::popcount(shift<kForward>(shift<kForward>(shift<kForward>(ourKnights))) & theirBishops);
    }

    const bool isOurKingLonely = (ourMen == ourKings);
    const bool isTheirKingLonely = (theirMen == theirKings);

    const CastlingRights cr = pos.currentState_.castlingRights;
    {
      // Hanging pieces are more valuable if it is your turn since they're literally free material,
      // as opposed to threats. Also is a very useful heuristic so that leaf nodes don't sack a rook
      // for a pawn.
      const Bitboard usHanging = theirTargets & ~ourTargets & pos.colorBitboards_[US];
      const Bitboard themHanging = ourTargets & ~theirTargets & pos.colorBitboards_[THEM];
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

      features[EF::LONELY_KING_AWAY_FROM_ENEMY_KING] = isTheirKingLonely * (8 - kingsDist);
      features[EF::LONELY_KING_AWAY_FROM_ENEMY_KING] -= isOurKingLonely * (8 - kingsDist);

      features[EF::CASTLING_RIGHTS] = ((cr & kCastlingRights_WhiteKing) > 0);
      features[EF::CASTLING_RIGHTS] += ((cr & kCastlingRights_WhiteQueen) > 0);
      features[EF::CASTLING_RIGHTS] -= ((cr & kCastlingRights_BlackKing) > 0);
      features[EF::CASTLING_RIGHTS] -= ((cr & kCastlingRights_BlackQueen) > 0);
      features[EF::KING_CASTLED] = std::popcount(ourKings & kHappyKingSquares) - std::popcount(theirKings & kHappyKingSquares);

      features[EF::NUM_TARGET_SQUARES] = std::popcount(ourTargets) - std::popcount(theirTargets);
    }

    {
      const Bitboard usHanging = theirDoubleTargets & ~ourDoubleTargets & pos.colorBitboards_[US];
      const Bitboard themHanging = ourDoubleTargets & ~theirDoubleTargets & pos.colorBitboards_[THEM];
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

    features[EF::LONELY_KING_ON_EDGE] -= (3 - kDistToEdge[theirKingSq]) * isTheirKingLonely;
    features[EF::LONELY_KING_ON_EDGE] = (3 - kDistToEdge[ourKingSq]) * isOurKingLonely;

    {
      Bitboard ourMaterialThreats = 0;
      ourMaterialThreats |= ourPawnTargets & theirPieces;
      ourMaterialThreats |= ourKnightTargets & theirMajors;
      ourMaterialThreats |= ourBishopTargets & theirMajors;
      ourMaterialThreats |= ourRookTargets & theirRoyalty;
      ourMaterialThreats |= ourQueenTargets & theirKings;

      Bitboard theirMaterialThreats = 0;
      theirMaterialThreats |= theirPawnTargets & ourPieces;
      theirMaterialThreats |= theirKnightTargets & ourMajors;
      theirMaterialThreats |= theirBishopTargets & ourMajors;
      theirMaterialThreats |= theirRookTargets & ourRoyalty;
      theirMaterialThreats |= theirQueenTargets & ourKings;

      features[EF::OUR_MATERIAL_THREATS] = std::popcount(ourMaterialThreats);
      features[EF::THEIR_MATERIAL_THREATS] = std::popcount(theirMaterialThreats);
    }

    features[EF::PAWN_MOVES] = std::popcount(ourPawnTargets) - std::popcount(theirPawnTargets);
    features[EF::KNIGHT_MOVES] = std::popcount(ourKnightTargets) - std::popcount(theirKnightTargets);
    features[EF::BISHOP_MOVES] = std::popcount(ourBishopTargetsIgnoringNonBlockades) - std::popcount(theirBishopTargetsIgnoringNonBlockades);
    features[EF::ROOK_MOVES] = std::popcount(ourRookTargets) - std::popcount(theirRookTargets);
    features[EF::QUEEN_MOVES] = std::popcount(ourQueenTargets) - std::popcount(theirQueenTargets);
    features[EF::PAWN_MOVES_ON_THEIR_SIDE] = std::popcount(ourPawnTargets & kTheirSide) - std::popcount(theirPawnTargets & kOurSide);
    features[EF::KNIGHT_MOVES_ON_THEIR_SIDE] = std::popcount(ourKnightTargets & kTheirSide) - std::popcount(theirKnightTargets & kOurSide);
    features[EF::BISHOP_MOVES_ON_THEIR_SIDE] = std::popcount(ourBishopTargetsIgnoringNonBlockades & kTheirSide) - std::popcount(theirBishopTargetsIgnoringNonBlockades & kOurSide);
    features[EF::ROOK_MOVES_ON_THEIR_SIDE] = std::popcount(ourRookTargets & kTheirSide) - std::popcount(theirRookTargets & kOurSide);
    features[EF::QUEEN_MOVES_ON_THEIR_SIDE] = std::popcount(ourQueenTargets & kTheirSide) - std::popcount(theirQueenTargets & kOurSide);

    // Bonus for king having pawns in front of him, or having pawns in front of him once he castles.
    features[EF::KING_HOME_QUALITY] = std::popcount(kKingHome[ourKingSq] & ourPawns) - std::popcount(kKingHome[theirKingSq] & theirPawns);
    const Evaluation whitePotentialHome = std::max(std::popcount(kKingHome[Square::G1] & ourPawns) * ((cr & kCastlingRights_WhiteKing) > 0), std::popcount(kKingHome[Square::B1] & ourPawns) * ((cr & kCastlingRights_WhiteQueen) > 0));
    const Evaluation blackPotentialHome = std::max(std::popcount(kKingHome[Square::G8] & ourPawns) * ((cr & kCastlingRights_BlackKing) > 0), std::popcount(kKingHome[Square::B8] & ourPawns) * ((cr & kCastlingRights_BlackQueen) > 0));
    if (US == Color::WHITE) {
      features[EF::KING_HOME_QUALITY] += whitePotentialHome - blackPotentialHome;
    } else {
      features[EF::KING_HOME_QUALITY] += blackPotentialHome - whitePotentialHome;
    }

    const int16_t ourPiecesRemaining = std::popcount(pos.colorBitboards_[US] & ~ourPawns) + std::popcount(ourQueens) * 2 - 1;
    const int16_t theirPiecesRemaining = std::popcount(pos.colorBitboards_[THEM] & ~theirPawns) + std::popcount(theirQueens) * 2 - 1;
    const int32_t time = 18 - (ourPiecesRemaining + theirPiecesRemaining);

    features[EF::TIME] = time;

    // Use larger integer to make arithmetic safe.
    const int32_t early = this->early<US>(pos);
    const int32_t late = this->late<US>(pos);
    const int32_t clipped = this->clipped<US>(pos);
    const int32_t lonely_king = this->lonely_king<US>(pos);

    int32_t pieceMap = (pos.earlyPieceMapScore_ * (18 - time) + pos.latePieceMapScore_ * time) / 18;
    if (US == Color::BLACK) {
      pieceMap *= -1;
    }

    int32_t eval = (early * (18 - time) + late * time) / 18 + clipped + lonely_king + pieceMap;

    return std::min(int32_t(-kLongestForcedMate), std::max(int32_t(kLongestForcedMate), eval));
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
  Evaluation clipped(const Position& pos) const {
    int32_t r = kClippedB0;
    for (size_t i = 0; i < EF::NUM_EVAL_FEATURES; ++i) {
      r += features[i] * kClippedW0[i];
    }
    return std::max(-100, std::min(100, r));
  }

  template<Color US>
  Evaluation lonely_king(const Position& pos) const {
    int32_t r = kLonelyKingB0;
    for (size_t i = 0; i < EF::NUM_EVAL_FEATURES; ++i) {
      r += features[i] * kLonelyKingW0[i];
    }

    const int32_t ourPieces = features[EF::OUR_KNIGHTS] + features[EF::OUR_BISHOPS] + features[EF::OUR_ROOKS] + features[EF::OUR_QUEENS];
    const int32_t theirPieces = features[EF::THEIR_KNIGHTS] + features[EF::THEIR_BISHOPS] + features[EF::THEIR_ROOKS] + features[EF::THEIR_QUEENS];

    return r * (1 - (ourPieces != 0) * (theirPieces != 0));
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
