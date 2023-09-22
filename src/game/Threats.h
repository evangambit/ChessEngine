#ifndef THREATS_H
#define THREATS_H

#include "utils.h"
#include "geometry.h"
#include "Position.h"

namespace ChessEngine {

template<Color US>
struct Threats {
  Bitboard ourPawnTargets;
  Bitboard ourKnightTargets;
  Bitboard ourBishopTargets;
  Bitboard ourRookTargets;
  Bitboard ourQueenTargets;
  Bitboard ourKingTargets;

  Bitboard theirPawnTargets;
  Bitboard theirKnightTargets;
  Bitboard theirBishopTargets;
  Bitboard theirRookTargets;
  Bitboard theirQueenTargets;
  Bitboard theirKingTargets;

  Bitboard ourTargets;
  Bitboard ourDoubleTargets;
  Bitboard theirTargets;
  Bitboard theirDoubleTargets;

  // TODO: use these.
  Bitboard badForOur[7];
  Bitboard badForTheir[7];

  template<ColoredPiece cp>
  Bitboard targets() const {
    constexpr bool isOurColor = (cp2color(cp) == US);
    constexpr Piece piece = cp2p(cp);
    switch (piece) {
      case Piece::PAWN:
        return isOurColor ? ourPawnTargets : theirPawnTargets;
      case Piece::KNIGHT:
        return isOurColor ? ourKnightTargets : theirKnightTargets;
      case Piece::BISHOP:
        return isOurColor ? ourBishopTargets : theirBishopTargets;
      case Piece::ROOK:
        return isOurColor ? ourRookTargets : theirRookTargets;
      case Piece::QUEEN:
        return isOurColor ? ourQueenTargets : theirQueenTargets;
      case Piece::KING:
        return isOurColor ? ourKingTargets : theirKingTargets;
      case Piece::NO_PIECE:
        return kEmptyBitboard;
    }
  }

  template<ColoredPiece cp>
  Bitboard badFor() const {
    constexpr bool isOurColor = (cp2color(cp) == US);
    constexpr Piece piece = cp2p(cp);
    if (isOurColor) {
      return badForOur[piece];
    } else {
      return badForTheir[piece];
    }
  }

  // TODO: bishops can attack one square through our own pawns.
  Threats(const Position& pos) {
    constexpr Color THEM = opposite_color<US>();
    constexpr Direction kForward = (US == Color::WHITE ? Direction::NORTH : Direction::SOUTH);
    constexpr Direction kForwardRight = (kForward == Direction::NORTH ? Direction::NORTH_EAST : Direction::SOUTH_WEST);
    constexpr Direction kForwardLeft = (kForward == Direction::NORTH ? Direction::NORTH_WEST : Direction::SOUTH_EAST);
    constexpr Direction kBackwardRight = (kForward == Direction::NORTH ? Direction::SOUTH_WEST : Direction::NORTH_EAST);
    constexpr Direction kBackwardLeft = (kForward == Direction::NORTH ? Direction::SOUTH_EAST : Direction::NORTH_WEST);

    const Square ourKingSq = lsb(pos.pieceBitboards_[coloredPiece<US, Piece::KING>()]);
    const Square theirKingSq = lsb(pos.pieceBitboards_[coloredPiece<THEM, Piece::KING>()]);

    const Bitboard everyone = pos.colorBitboards_[Color::WHITE] | pos.colorBitboards_[Color::BLACK];

    const Bitboard ourRooklikePieces = pos.pieceBitboards_[coloredPiece<US, Piece::ROOK>()] | pos.pieceBitboards_[coloredPiece<US, Piece::QUEEN>()];
    const Bitboard theirRooklikePieces = pos.pieceBitboards_[coloredPiece<THEM, Piece::ROOK>()] | pos.pieceBitboards_[coloredPiece<THEM, Piece::QUEEN>()];
    const Bitboard ourBishoplikePieces = pos.pieceBitboards_[coloredPiece<US, Piece::BISHOP>()] | pos.pieceBitboards_[coloredPiece<US, Piece::QUEEN>()];
    const Bitboard theirBishoplikePieces = pos.pieceBitboards_[coloredPiece<THEM, Piece::BISHOP>()] | pos.pieceBitboards_[coloredPiece<THEM, Piece::QUEEN>()];

    Bitboard ourPawn1 = shift<kForwardRight>(pos.pieceBitboards_[coloredPiece<US, Piece::PAWN>()]);
    Bitboard ourPawn2 = shift<kForwardLeft>(pos.pieceBitboards_[coloredPiece<US, Piece::PAWN>()]);
    Bitboard theirPawn1 = shift<kBackwardRight>(pos.pieceBitboards_[coloredPiece<THEM, Piece::PAWN>()]);
    Bitboard theirPawn2 = shift<kBackwardLeft>(pos.pieceBitboards_[coloredPiece<THEM, Piece::PAWN>()]);

    this->ourPawnTargets = ourPawn1 | ourPawn2;
    this->theirPawnTargets = theirPawn1 | theirPawn2;
    this->ourKnightTargets = compute_knight_targets<US>(pos);
    this->theirKnightTargets = compute_knight_targets<THEM>(pos);
    this->ourBishopTargets = compute_bishoplike_targets<US>(pos, pos.pieceBitboards_[coloredPiece<US, Piece::BISHOP>()], everyone & ~pos.pieceBitboards_[coloredPiece<US, Piece::BISHOP>()]);
    this->theirBishopTargets = compute_bishoplike_targets<THEM>(pos, pos.pieceBitboards_[coloredPiece<THEM, Piece::BISHOP>()], everyone & ~pos.pieceBitboards_[coloredPiece<THEM, Piece::BISHOP>()]);
    this->ourRookTargets = compute_rooklike_targets(pos.pieceBitboards_[coloredPiece<US, Piece::ROOK>()], everyone & ~pos.pieceBitboards_[coloredPiece<US, Piece::ROOK>()]);
    this->theirRookTargets = compute_rooklike_targets(pos.pieceBitboards_[coloredPiece<THEM, Piece::ROOK>()], everyone & ~pos.pieceBitboards_[coloredPiece<THEM, Piece::ROOK>()]);
    this->ourQueenTargets = compute_bishoplike_targets<US>(pos, pos.pieceBitboards_[coloredPiece<US, Piece::QUEEN>()], everyone & ~ourBishoplikePieces) | compute_rooklike_targets(pos.pieceBitboards_[coloredPiece<US, Piece::QUEEN>()], everyone & ~ourRooklikePieces);
    this->theirQueenTargets = compute_bishoplike_targets<THEM>(pos, pos.pieceBitboards_[coloredPiece<THEM, Piece::QUEEN>()], everyone & ~theirBishoplikePieces) | compute_rooklike_targets(pos.pieceBitboards_[coloredPiece<THEM, Piece::QUEEN>()], everyone & ~theirRooklikePieces);
    this->ourKingTargets = compute_king_targets<US>(pos, ourKingSq);
    this->theirKingTargets = compute_king_targets<THEM>(pos, theirKingSq);

    this->ourTargets = ourPawn1 | ourPawn2;
    this->ourDoubleTargets = ourPawn1 & ourPawn2;
    this->theirTargets = theirPawn1 | theirPawn2;
    this->theirDoubleTargets = theirPawn1 & theirPawn2;
    { // Compute the above four variables.
      // Note: "ourDoubleTargets" and "theirDoubleTargets" are approximations, since
      // they ignore the possibility that two of the same piece can attack a square
      // (except for pawn double-attacks).
      this->ourDoubleTargets |= this->ourTargets & this->ourKnightTargets;
      this->ourTargets |= this->ourKnightTargets;
      this->theirDoubleTargets |= this->theirTargets & this->theirKnightTargets;
      this->theirTargets |= this->theirKnightTargets;

      this->ourDoubleTargets |= this->ourTargets & this->ourBishopTargets;
      this->ourTargets |= this->ourBishopTargets;
      this->theirDoubleTargets |= this->theirTargets & this->theirBishopTargets;
      this->theirTargets |= this->theirBishopTargets;

      this->ourDoubleTargets |= this->ourTargets & this->ourRookTargets;
      this->ourTargets |= this->ourRookTargets;
      this->theirDoubleTargets |= this->theirTargets & this->theirRookTargets;
      this->theirTargets |= this->theirRookTargets;

      this->ourDoubleTargets |= this->ourTargets & this->ourQueenTargets;
      this->ourTargets |= this->ourQueenTargets;
      this->theirDoubleTargets |= this->theirTargets & this->theirQueenTargets;
      this->theirTargets |= this->theirQueenTargets;

      this->ourDoubleTargets |= this->ourTargets & this->ourKingTargets;
      this->ourTargets |= this->ourKingTargets;
      this->theirDoubleTargets |= this->theirTargets & this->theirKingTargets;
      this->theirTargets |= this->theirKingTargets;
    }

    const Bitboard badForAllOfUs = this->theirTargets & ~this->ourTargets;
    const Bitboard badForAllOfThem = this->ourTargets & ~this->theirTargets;

    const Bitboard theirMinorTargets = this->theirKnightTargets | this->theirBishopTargets;
    const Bitboard ourMinorTargets = this->ourKnightTargets | this->ourBishopTargets;

    this->badForOur[Piece::PAWN]   = badForAllOfUs | (this->theirDoubleTargets & ~this->ourDoubleTargets);
    this->badForOur[Piece::KNIGHT] = badForAllOfUs | this->theirPawnTargets | (this->theirDoubleTargets & ~this->ourDoubleTargets);
    this->badForOur[Piece::BISHOP] = this->badForOur[Piece::KNIGHT];
    this->badForOur[Piece::ROOK]   = badForAllOfUs | this->theirPawnTargets | theirMinorTargets | (this->theirDoubleTargets & ~this->ourDoubleTargets);
    this->badForOur[Piece::QUEEN]  = badForAllOfUs | this->theirPawnTargets | theirMinorTargets | this->theirRookTargets | (this->theirDoubleTargets & ~this->ourDoubleTargets);
    this->badForOur[Piece::KING]   = this->theirTargets;

    this->badForTheir[Piece::PAWN]   = badForAllOfThem | (this->ourDoubleTargets & ~this->theirDoubleTargets);
    this->badForTheir[Piece::KNIGHT] = badForAllOfThem | this->ourPawnTargets | (this->ourDoubleTargets & ~this->theirDoubleTargets);
    this->badForTheir[Piece::BISHOP] = this->badForTheir[Piece::KNIGHT];
    this->badForTheir[Piece::ROOK]   = badForAllOfThem | this->ourPawnTargets | ourMinorTargets | (this->ourDoubleTargets & ~this->theirDoubleTargets & this->ourRookTargets);
    this->badForTheir[Piece::QUEEN]  = badForAllOfThem | this->ourPawnTargets | ourMinorTargets | this->ourRookTargets | (this->ourDoubleTargets & ~this->theirDoubleTargets);
    this->badForTheir[Piece::KING]   = this->ourTargets;
  }
};

}  // namespace ChessEngine

#endif  // THREATS_H