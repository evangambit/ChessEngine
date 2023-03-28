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

  QUEEN_THREATS_NEAR_KING,
  MISSING_FIANCHETTO_BISHOP,
  BISHOP_PAWN_DISAGREEMENT,
  CLOSED_1,
  CLOSED_2,
  CLOSED_3,

  NUM_EVAL_FEATURES,
};

constexpr Bitboard kWhiteKingCorner = bb(Square::H1) | bb(Square::H2) | bb(Square::G1) | bb(Square::G2) | bb(Square::F1);
constexpr Bitboard kWhiteQueenCorner = bb(Square::A1) | bb(Square::A2) | bb(Square::B1) | bb(Square::B2) | bb(Square::C1);
constexpr Bitboard kBlackKingCorner = bb(Square::H8) | bb(Square::H7) | bb(Square::G8) | bb(Square::G7) | bb(Square::F8);
constexpr Bitboard kBlackQueenCorner = bb(Square::A8) | bb(Square::A7) | bb(Square::B8) | bb(Square::B7) | bb(Square::C8);

const int32_t kEarlyB0 = 4;
const int32_t kEarlyW0[EF::NUM_EVAL_FEATURES] = {
  17, 204, 202, 189, 552, -35,
-188,-181,-170,-499,-182,  14,
 -24,   3, -18,  -3, -27,  26,
  -3,   6,  -2,  -1,  12,  18,
 -11,   6, -16,  42,   3,   7,
  12,  33,  -7,   8,   1,  11,
  30,   3,  12,  19,  13,   4,
  17,   8, -25, -18, -42, -15,
   7,  24,  24, -70, 108,   0,
   0,   0,   4,  18,  10,  -1,
   3,  -3, -27,  50,  17,  12,
  17,   0,-143, -76,   0,   0,
   0,   0,   0,  68, -47,  -2,
   2,   1,   7,   0,   0,   3,
   2,   0,   1,   1,  -3,   2,
   9,  -2,   1, -21, -22,   4,
 -17,  17,   4, -10,   6, -43,
-134, -13,   0,   1,  -1,   1,
};
const int32_t kLateB0 = -7;
const int32_t kLateW0[EF::NUM_EVAL_FEATURES] = {
  55, 161, 148, 311, 779, -55,
-145,-137,-301,-708,  28, -39,
  15,  -4,   5,   1,  19, -21,
  -3, -13,  -2,  -7, -15,-111,
  43,  18, -24,  -9,   3,   6,
  -8,  30,  -4,  -4,  33,   8,
 -15,   2,  10,  -6,  10,  10,
  -6, -28, -28, -48,  -6,  17,
  17,  30,  38,  -6, 201, 629,
 114,   0,   0, 277,2355, 156,
 727, -66, 295,  18,  -3,   6,
   2,   0,  62,  49,   0,   0,
   0,   0,   0,  25, -11,-919,
   5,   5,  -3,   0,   1,  -1,
  -6,   4,   2,   4,   0,   3,
  -1,   5, -26, -30, -33,  -1,
 -10,  19,  15, -37,   4,-157,
 -32,   6,   0,   0,  -1,   2,
};
const int32_t kClippedB0 = -7;
const int32_t kClippedW0[EF::NUM_EVAL_FEATURES] = {
  46, 169, 165, 266, 102, -43,
-166,-166,-264, -98,-937,  -5,
   2,  24,  -3,  -1,   2,  -3,
  -3,  -4,  -8,  -1,   3,  65,
  13,  10,   9,  -1,   2,   3,
   7,  10,  -2,  11,  -8,   7,
  21,   2,  11,  10,   7,   3,
   4, -18, -16,  -9, -15,   7,
  23,  67,  67, 268,-138, 980,
 225,   0,   0,1634,1103, 781,
3459, 232,-114,   2,  -2,  10,
   5,  -1,  -1,  15,   0,   0,
   0,   0,   0,  48, -13,-1146,
   8,   5,   2,   0,   2,   3,
   2,   2,   0,   0,   0,   3,
   1,  -4,  -8,  -1,   0,  -6,
  26,  16, 144, 256, 212, 519,
  35,  -6,   0,   0,   0,   0,
};
const int32_t kLonelyKingB0 = -9;
const int32_t kLonelyKingW0[EF::NUM_EVAL_FEATURES] = {
  37,  59,  96, 134, 177, -41,
 -67, -90,-147,-125,-2599,  33,
  -4,   3,  -2,   3,  -9,  -3,
  13, -20, -31, -12,   5, -28,
  28,  10, 208, 125,  -8,  -8,
 -39,  13,  32,  92, -69, -11,
  77,   8,   6, 103,  26,  -4,
 -44, -28, -13,  29, -53,  98,
  18,  83,  89, 181, 102,-673,
-121,   0,   2,-326,-2319, -64,
-987, 179,  73,  -2,  10,  22,
  -4,-138, -19,   8,   0,   0,
   0,   0,   0,-197, 179, 970,
 -25, -27,   7,   0,   1,   0,
 -11,   1,  -4,  -7,  -6,   2,
 -12,-213,  -5, -68,  23,-778,
-267, -18,-140, -42, 391,  29,
  17,   1,   0,   0,  -1,   5,
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
  "QUEEN_THREATS_NEAR_KING",
  "MISSING_FIANCHETTO_BISHOP",
  "BISHOP_PAWN_DISAGREEMENT",
  "CLOSED_1",
  "CLOSED_2",
  "CLOSED_3",
};

// captures = difference in values divided by 2

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

    // TODO: penalty for double attacks near king

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
    const Bitboard ourBishopTargets = compute_bishoplike_targets<US>(pos, ourBishops, (ourMen & ~ourBishoplikePieces) | theirMen);
    const Bitboard theirBishopTargets = compute_bishoplike_targets<THEM>(pos, theirBishops, (theirMen & ~theirBishoplikePieces) | ourMen);
    const Bitboard ourRookTargets = compute_rooklike_targets<US>(pos, ourRooks, (ourMen & ~ourRooklikePieces) | theirMen);
    const Bitboard theirRookTargets = compute_rooklike_targets<THEM>(pos, theirRooks, ourMen | (theirMen & ~theirRooklikePieces));
    const Bitboard ourQueenTargets = compute_bishoplike_targets<US>(pos, ourQueens, (ourMen & ~ourBishoplikePieces) | theirMen)
    | compute_rooklike_targets<US>(pos, ourQueens, (ourMen & ~ourRooklikePieces) | theirMen);
    const Bitboard theirQueenTargets = compute_bishoplike_targets<THEM>(pos, theirQueens, (theirMen & ~theirBishoplikePieces) | ourMen)
    | compute_rooklike_targets<THEM>(pos, theirQueens, (theirMen & ~theirRooklikePieces) | ourMen);
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
    features[EF::THREATS_NEAR_KING_3] = std::popcount(kNearby[3][ourKingSq] & theirTargets & ~ourTargets) - std::popcount(kNearby[3][theirKingSq] & ourTargets & ~theirTargets);
    features[EF::QUEEN_THREATS_NEAR_KING] = std::popcount(kNearby[1][ourKingSq] & theirDoubleTargets & ~ourDoubleTargets & theirQueenTargets) - std::popcount(kNearby[1][theirKingSq] & ourDoubleTargets & ~theirDoubleTargets & ourQueenTargets);

    {  // Add penalty if the king is in a fianchettoed corner and his bishop is not on the main diagonal.
      // Note: the "color" of a corner is the color of its fianchettoed bishop.
      constexpr Bitboard kOurWhiteCorner = (US == Color::WHITE ? kWhiteKingCorner : kBlackQueenCorner);
      constexpr Bitboard kOurBlackCorner = (US == Color::BLACK ? kWhiteQueenCorner : kBlackKingCorner);
      constexpr Bitboard kTheirWhiteCorner = (US != Color::WHITE ? kWhiteKingCorner : kBlackQueenCorner);
      constexpr Bitboard kTheirBlackCorner = (US != Color::BLACK ? kWhiteQueenCorner : kBlackKingCorner);
      constexpr Bitboard ourWhiteFianchettoPawn = bb(US == Color::WHITE ? Square::G2 : Square::B7);
      constexpr Bitboard ourBlackFianchettoPawn = bb(US == Color::WHITE ? Square::B2 : Square::G7);
      constexpr Bitboard theirWhiteFianchettoPawn = bb(US != Color::WHITE ? Square::G2 : Square::B7);
      constexpr Bitboard theirBlackFianchettoPawn = bb(US != Color::WHITE ? Square::B2 : Square::G7);
      features[EF::MISSING_FIANCHETTO_BISHOP] = 0;
      features[EF::MISSING_FIANCHETTO_BISHOP] += ((ourKings & kOurWhiteCorner) > 0) && ((kMainWhiteDiagonal & ourBishops) == 0) && ((kWhiteSquares & theirBishops) > 0) && ((ourPawns & ourWhiteFianchettoPawn) == 0);
      features[EF::MISSING_FIANCHETTO_BISHOP] += ((ourKings & kOurBlackCorner) > 0) && ((kMainBlackDiagonal & ourBishops) == 0) && ((kBlackSquares & theirBishops) > 0) && ((ourPawns & ourBlackFianchettoPawn) == 0);
      features[EF::MISSING_FIANCHETTO_BISHOP] -= ((theirKings & kOurWhiteCorner) > 0) && ((kMainWhiteDiagonal & theirBishops) == 0) && ((kWhiteSquares & ourBishops) > 0) && ((theirPawns & theirWhiteFianchettoPawn) == 0);
      features[EF::MISSING_FIANCHETTO_BISHOP] -= ((theirKings & kOurBlackCorner) > 0) && ((kMainBlackDiagonal & theirBishops) == 0) && ((kBlackSquares & ourBishops) > 0) && ((theirPawns & theirBlackFianchettoPawn) == 0);
    }

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

      features[EF::BISHOP_PAWN_DISAGREEMENT] = std::popcount(ourBishops & ourPawns & kWhiteSquares) + std::popcount(ourBishops & ourPawns & kBlackSquares);
      features[EF::BISHOP_PAWN_DISAGREEMENT] -= std::popcount(theirBishops & theirPawns & kWhiteSquares) + std::popcount(theirBishops & theirPawns & kBlackSquares);

      features[EF::CLOSED_1] = std::popcount(ourBlockadedPawns | theirBlockadedPawns);
      features[EF::CLOSED_2] = std::popcount((ourBlockadedPawns | theirBlockadedPawns) & kCenter16);
      features[EF::CLOSED_3] = std::popcount(
        ((shift<Direction::NORTH_EAST>(ourBlockadedPawns) | shift<Direction::NORTH_WEST>(ourBlockadedPawns)) & ourBlockadedPawns)
        |
        ((shift<Direction::NORTH_EAST>(theirBlockadedPawns) | shift<Direction::NORTH_WEST>(theirBlockadedPawns)) & theirBlockadedPawns)
      );
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
      features[EF::LONELY_KING_IN_CENTER] = (3 - kDistToCorner[theirKingSq]) * (std::popcount(pos.colorBitboards_[THEM]) == 1);
      features[EF::LONELY_KING_IN_CENTER] -= (3 - kDistToCorner[ourKingSq]) * (std::popcount(pos.colorBitboards_[US]) == 1);

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

    // loser in center: 0
    // they're lonely on edge: 3
    //   we're lonely on edge: -3
    features[EF::LONELY_KING_ON_EDGE] = (3 - kDistToEdge[theirKingSq]) * isTheirKingLonely;
    features[EF::LONELY_KING_ON_EDGE] -= (3 - kDistToEdge[ourKingSq]) * isOurKingLonely;

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

    {
      const Bitboard ourGoodKnightTargets = ourKnightTargets & ~theirPawnTargets;
      const Bitboard theirGoodKnightTargets = ourKnightTargets & ~theirPawnTargets;
      const Bitboard ourGoodBishopTargets = ourBishopTargetsIgnoringNonBlockades & ~theirPawnTargets;
      const Bitboard theirGoodBishopTargets = theirBishopTargetsIgnoringNonBlockades & ~ourPawnTargets;
      const Bitboard ourGoodRookTargets = ourRookTargets & ~theirPawnTargets;
      const Bitboard theirGoodRookTargets = theirRookTargets & ~ourPawnTargets;
      const Bitboard ourGoodQueenTargets = ourQueenTargets & ~(theirPawnTargets | theirKnightTargets | theirBishopTargets | theirRookTargets);
      const Bitboard theirGoodQueenTargets = theirQueenTargets & ~(ourPawnTargets | ourKnightTargets | ourBishopTargets | ourRookTargets);
      features[EF::PAWN_MOVES] = std::popcount(ourPawnTargets) - std::popcount(theirPawnTargets);
      features[EF::KNIGHT_MOVES] = std::popcount(ourGoodKnightTargets) - std::popcount(theirGoodKnightTargets);
      features[EF::BISHOP_MOVES] = std::popcount(ourGoodBishopTargets) - std::popcount(theirGoodBishopTargets);
      features[EF::ROOK_MOVES] = std::popcount(ourGoodRookTargets) - std::popcount(theirGoodRookTargets);
      features[EF::QUEEN_MOVES] = std::popcount(ourGoodQueenTargets) - std::popcount(theirGoodQueenTargets);
      features[EF::PAWN_MOVES_ON_THEIR_SIDE] = std::popcount(ourPawnTargets & kTheirSide) - std::popcount(theirPawnTargets & kOurSide);
      features[EF::KNIGHT_MOVES_ON_THEIR_SIDE] = std::popcount(ourGoodKnightTargets & kTheirSide) - std::popcount(theirGoodKnightTargets & kOurSide);
      features[EF::BISHOP_MOVES_ON_THEIR_SIDE] = std::popcount(ourGoodBishopTargets & kTheirSide) - std::popcount(theirGoodBishopTargets & kOurSide);
      features[EF::ROOK_MOVES_ON_THEIR_SIDE] = std::popcount(ourGoodRookTargets & kTheirSide) - std::popcount(theirGoodRookTargets & kOurSide);
      features[EF::QUEEN_MOVES_ON_THEIR_SIDE] = std::popcount(ourGoodQueenTargets & kTheirSide) - std::popcount(theirGoodQueenTargets & kOurSide);
    }

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
