#ifndef PAWN_ANALYSIS_H
#define PAWN_ANALYSIS_H

#include "utils.h"
#include "geometry.h"
#include "Threats.h"

namespace ChessEngine {

template<Color US>
struct PawnAnalysis {
  Bitboard ourBlockadedPawns, theirBlockadedPawns;
  Bitboard ourProtectedPawns, theirProtectedPawns;
  Bitboard ourPassedPawns, theirPassedPawns;
  Bitboard ourDoubledPawns, theirDoubledPawns;
  Bitboard filesWithoutOurPawns, filesWithoutTheirPawns;
  Bitboard possibleOutpostsForUs, possibleOutpostsForThem;
  Bitboard piecesOurPawnsCanThreaten, piecesTheirPawnsCanThreaten;
  Bitboard ourIsolatedPawns, theirIsolatedPawns;
  PawnAnalysis(const Position& pos, const Threats<US>& threats) {
    constexpr Color THEM = opposite_color<US>();
    constexpr Direction kForward = (US == Color::WHITE ? Direction::NORTH : Direction::SOUTH);
    constexpr Direction kBackward = (US == Color::WHITE ? Direction::SOUTH : Direction::NORTH);

    const Bitboard ourPawns = pos.pieceBitboards_[coloredPiece<US, Piece::PAWN>()];
    const Bitboard theirPawns = pos.pieceBitboards_[coloredPiece<THEM, Piece::PAWN>()];
    const Bitboard ourPieces = pos.colorBitboards_[US] & ~ourPawns;
    const Bitboard theirPieces = pos.colorBitboards_[THEM] & ~theirPawns;
    const Bitboard everyone = pos.colorBitboards_[Color::WHITE] | pos.colorBitboards_[Color::BLACK];

    constexpr Bitboard kOurBackRanks = (US == Color::WHITE ? kRanks[6] | kRanks[7] : kRanks[1] | kRanks[0]);
    constexpr Bitboard kTheirBackRanks = (US == Color::WHITE ? kRanks[1] | kRanks[0] : kRanks[6] | kRanks[7]);

    this->ourBlockadedPawns = shift<kBackward>(theirPawns) & ourPawns;
    this->theirBlockadedPawns = shift<kForward>(ourPawns) & theirPawns;
    this->ourProtectedPawns = ourPawns & threats.ourPawnTargets;
    this->theirProtectedPawns = theirPawns & threats.theirPawnTargets;
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
      filesWithoutOurPawns = ~filesWithOurPawns;
      filesWithoutTheirPawns = ~filesWithTheirPawns;
      ourPassedPawns = ourPawns & ~shift<kBackward>(fatten(aheadOfTheirPawns));
      theirPassedPawns = theirPawns & ~shift<kForward>(fatten(aheadOfOurPawns));
      this->ourIsolatedPawns = ourPawns & ~shift<Direction::WEST>(filesWithOurPawns) & ~shift<Direction::EAST>(filesWithOurPawns);
      this->theirIsolatedPawns = theirPawns & ~shift<Direction::WEST>(filesWithTheirPawns) & ~shift<Direction::EAST>(filesWithTheirPawns);
      this->ourDoubledPawns = ourPawns & shift<kForward>(aheadOfOurPawns);
      this->theirDoubledPawns = theirPawns & shift<kBackward>(aheadOfTheirPawns);

      possibleOutpostsForUs = ~(
        shift<Direction::EAST>(shift<kBackward>(aheadOfTheirPawns))
        | shift<Direction::WEST>(shift<kBackward>(aheadOfTheirPawns))
      ) & ~kTheirBackRanks;
      possibleOutpostsForThem = ~(
        shift<Direction::EAST>(shift<kForward>(aheadOfOurPawns))
        | shift<Direction::WEST>(shift<kForward>(aheadOfOurPawns))
      ) & ~kOurBackRanks;

      const Bitboard aheadOfOurPawnHome = (US == Color::WHITE ? kRanks[5] : kRanks[2]);
      const Bitboard behindTheirPawnHome = (US == Color::WHITE ? kRanks[2] : kRanks[5]);

      Bitboard squaresWeCanAdvancePawnsTo = shift<kForward>(ourPawns) & ~everyone;
      squaresWeCanAdvancePawnsTo |= shift<kForward>(squaresWeCanAdvancePawnsTo & aheadOfOurPawnHome) & ~everyone;
      squaresWeCanAdvancePawnsTo &= ~threats.badForOur[Piece::PAWN];

      Bitboard squaresTheyCanAdvancePawnsTo = shift<kBackward>(theirPawns) & ~everyone;
      squaresTheyCanAdvancePawnsTo |= shift<kBackward>(squaresTheyCanAdvancePawnsTo & behindTheirPawnHome) & ~everyone;
      squaresTheyCanAdvancePawnsTo &= ~threats.badForTheir[Piece::PAWN];

      constexpr Direction kForwardLeft = (US == Color::WHITE ? Direction::NORTH_WEST : Direction::SOUTH_EAST);
      constexpr Direction kForwardRight = (US == Color::WHITE ? Direction::NORTH_EAST : Direction::SOUTH_WEST);
      constexpr Direction kBackwardLeft = opposite_dir<kForwardLeft>();
      constexpr Direction kBackwardRight = opposite_dir<kForwardRight>();

      this->piecesOurPawnsCanThreaten = (shift<kForwardLeft>(squaresWeCanAdvancePawnsTo) | shift<kForwardRight>(squaresWeCanAdvancePawnsTo)) & theirPieces;
      this->piecesTheirPawnsCanThreaten = (shift<kBackwardLeft>(squaresTheyCanAdvancePawnsTo) | shift<kBackwardRight>(squaresTheyCanAdvancePawnsTo)) & ourPieces;
    }
  }
};

}  // namespace ChessEngine

#endif  // PAWN_ANALYSIS_H