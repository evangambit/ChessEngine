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

  NUM_EVAL_FEATURES,
};

const int32_t kEarlyB0 = 1;
const int32_t kEarlyW0[EF::NUM_EVAL_FEATURES] = { 28,247,262,329,566,-28,-247,-262,-329,-566,-115,15,-22,-29,-18,1,-24,-13,-8,-19,6,11,3,53,57,24,42,39,5,20,19,44,-17,24,-15,-32,72,21,14,63,1,5,12,9,-23,-18,-47,-9,4,87,91,39,45,-242,-607,2,-1,1,1,1,0,1,1};
const int32_t kLateB0 = -45;
const int32_t kLateW0[EF::NUM_EVAL_FEATURES] = { 78,130,140,268,382,-78,-130,-140,-268,-382,-49,-46,3,32,1,1,3,4,-13,-19,-5,-8,3,68,42,21,18,-36,5,5,25,21,1,1,31,-1,-38,3,45,-41,9,8,3,-32,-25,-50,-32,-28,19,20,-24,40,-11,32,20,4,-5,0,0,0,0,0,0};

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
  "ADVANCED_PASSED_PAWNS_1",
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

  "NUM_EVAL_FEATURES",
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

    const Bitboard ourRoyalty = ourQueens | ourKings;
    const Bitboard theiryRoyalty = theirQueens | theirKings;
    const Bitboard ourMajors = ourRooks | ourRoyalty;
    const Bitboard theirMajors = theirRooks | theiryRoyalty;
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
    const Bitboard usRookTargets = compute_rooklike_targets<US>(pos, ourRooks);
    const Bitboard theirRookTargets = compute_rooklike_targets<THEM>(pos, theirRooks);
    const Bitboard usQueenTargets = compute_bishoplike_targets<US>(pos, ourQueens) | compute_rooklike_targets<US>(pos, ourQueens);
    const Bitboard theirQueenTargets = compute_bishoplike_targets<THEM>(pos, theirQueens) | compute_rooklike_targets<THEM>(pos, theirQueens);

    constexpr Bitboard kOurSide = (US == Color::WHITE ? kWhiteSide : kBlackSide);
    constexpr Bitboard kTheirSide = (US == Color::WHITE ? kBlackSide : kWhiteSide);
    constexpr Direction kForward = (US == Color::WHITE ? Direction::NORTH : Direction::SOUTH);
    constexpr Direction kBackward = opposite_dir(kForward);

    const Bitboard usTargets = ourPawnTargets | ourKnightTargets | usBishopTargets | usRookTargets | usQueenTargets;
    const Bitboard themTargets = theirPawnTargets | theirKnightTargets | theirBishopTargets | theirRookTargets | theirQueenTargets;

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
      const Bitboard fatUsPawns = fatten(ourFilled);
      const Bitboard fatThemPawns = fatten(theirFilled);
      const Bitboard filesWithoutUsPawns = ~filesWithOurPawns;
      const Bitboard filesWithoutThemPawns = ~filesWithTheirPawns;
      const Bitboard ourPassedPawns = ourPawns & ~shift<kBackward>(fatten(theirFilled));
      const Bitboard theirPassedPawns = theirPawns & ~shift<kForward>(fatten(ourFilled));
      const Bitboard ourIsolatedPawns = ourPawns & ~shift<Direction::WEST>(filesWithOurPawns) & ~shift<Direction::EAST>(filesWithOurPawns);
      const Bitboard theirIsolatedPawns = theirPawns & ~shift<Direction::WEST>(filesWithTheirPawns) & ~shift<Direction::EAST>(filesWithTheirPawns);
      const Bitboard ourDoubledPawns = ourPawns & shift<kForward>(ourFilled);
      const Bitboard theirDoubledPawns = theirPawns & shift<kBackward>(theirFilled);

      constexpr Bitboard kRookPawns = kFiles[0] | kFiles[7];

      features[EF::PAWNS_CENTER_16] = std::popcount(ourPawns & kCenter16) - std::popcount(theirPawns & kCenter16);
      features[EF::PAWNS_CENTER_16] = std::popcount(ourPawns & kCenter16) - std::popcount(theirPawns & kCenter16);
      features[EF::PAWNS_CENTER_4] = std::popcount(ourPawns & kCenter4) - std::popcount(theirPawns & kCenter4);
      features[EF::PASSED_PAWNS] = std::popcount(ourPassedPawns) * 2 - std::popcount(theirPassedPawns);
      features[EF::ISOLATED_PAWNS] = std::popcount(ourIsolatedPawns) - std::popcount(theirIsolatedPawns);
      features[EF::DOUBLED_PAWNS] = std::popcount(ourDoubledPawns) - std::popcount(theirDoubledPawns);
      features[EF::DOUBLE_ISOLATED_PAWNS] = std::popcount(ourDoubledPawns & ourIsolatedPawns) - std::popcount(theirDoubledPawns & theirIsolatedPawns);

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
      features[EF::SCARIER_BISHOPS] = std::popcount(ourBishopTargetsIgnoringNonBlockades & theiryRoyalty) - std::popcount(theirBishopTargetsIgnoringNonBlockades & ourRoyalty);

      // const Bitboard ourPawnsOnUs = kUsSquares & ourPawns;
      // const Bitboard ourPawnsOnThem = kThemSquares & ourPawns;
      // const Bitboard theirPawnsOnUs = kUsSquares & theirPawns;
      // const Bitboard theirPawnsOnThem = kThemSquares & theirPawns;
    }

    constexpr Bitboard kOurBackRanks = (US == Color::WHITE ? kRanks[6] | kRanks[7] : kRanks[1] | kRanks[0]);
    constexpr Bitboard kTheirBackRanks = (US == Color::WHITE ? kRanks[1] | kRanks[0] : kRanks[6] | kRanks[7]);

    {  // Rooks
      features[EF::BLOCKADED_ROOKS] = std::popcount(usRookTargets & ourPawns) - std::popcount(theirRookTargets & theirPawns);
      features[EF::SCARY_ROOKS] = std::popcount(usRookTargets & theiryRoyalty) - std::popcount(theirRookTargets & ourRoyalty);
      features[EF::INFILTRATING_ROOKS] = std::popcount(ourRooks & kTheirBackRanks) - std::popcount(theirRooks & kOurBackRanks);
    }

    {  // Knights
      if (US == Color::WHITE) {
        features[EF::KNIGHTS_DEVELOPED] = std::popcount(theirKnights & (bb( 1) | bb( 6))) - std::popcount(ourKnights & (bb(57) | bb(62)));
      } else {
        features[EF::KNIGHTS_DEVELOPED] = std::popcount(theirKnights & (bb(57) | bb(62))) - std::popcount(ourKnights & (bb( 1) | bb( 6)));
      }
      features[EF::KNIGHT_MAJOR_CAPTURES] = std::popcount(compute_knight_targets<US>(pos) & theirMajors) * 2 - std::popcount(compute_knight_targets<THEM>(pos) & ourMajors);
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

    const int16_t ourPiecesRemaining = std::popcount(pos.colorBitboards_[US] & ~ourPawns) + std::popcount(ourQueens) - 1;
    const int16_t theirPiecesRemaining = std::popcount(pos.colorBitboards_[THEM] & ~theirPawns) + std::popcount(theirQueens) - 1;
    const int32_t earliness = (ourPiecesRemaining + theirPiecesRemaining);
    features[EF::TIME] = earliness;

    // Use larger integer to make arithmetic safe.
    const int32_t early = this->early<US>(pos);
    const int32_t late = this->late<US>(pos);
    
    return (early * earliness + late * (16 - earliness)) / 16;
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
  Evaluation features[NUM_EVAL_FEATURES];
};

}  // namespace ChessEngine

#endif  // EVALUATOR_H