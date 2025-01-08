#ifndef THREATS_H
#define THREATS_H

#include "utils.h"
#include "geometry.h"
#include "Position.h"

#include "movegen/bishops.h"
#include "movegen/rooks.h"
#include "movegen/knights.h"
#include "movegen/kings.h"

namespace ChessEngine {

inline Bitboard at_least_two(Bitboard a, Bitboard b) {
  return a & b;
}

inline Bitboard at_least_two(Bitboard a, Bitboard b, Bitboard c) {
  return (a & b) | (a & c) | (b & c);
}

Bitboard at_least_two(Bitboard a, Bitboard b, Bitboard c, Bitboard d) {
  // return (a & b) | (a & c) | (a & d) | (b & c) | (b & d) | (c & d);
  // return (a & (b | c | d)) | (b & (c | d)) | (c & d);  // 8 ops
  // return ((a ^ b) & (c ^ d)) | (a & b) | (c & d);  // 7 ops
  return ((a | b) & (c | d)) | (a & b) | (c & d);  // 7 ops
}

Bitboard at_least_two(Bitboard a, Bitboard b, Bitboard c, Bitboard d, Bitboard e) {
  // return (a & b) | (a & c) | (a & d) | (a & e) | (b & c) | (b & d) | (b & e) | (c & d) | (c & e) | (d & e);
  // return (a & (b | c | d | e)) | (b & (c | d | e)) | (c & (d | e)) | (d & e);  // 13 ops
  return ((a | b) & (c | d | e)) | (a & b) | at_least_two(c, d, e);  // 12 ops
}

Bitboard at_least_two(Bitboard a, Bitboard b, Bitboard c, Bitboard d, Bitboard e, Bitboard f) {
  return ((a | b | c) & (d | e | f)) | at_least_two(a, b, c) | at_least_two(d, e, f);  // 21 ops
}

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

    const SafeSquare ourKingSq = lsb_i_promise_board_is_not_empty(pos.pieceBitboards_[coloredPiece<US, Piece::KING>()]);
    const SafeSquare theirKingSq = lsb_i_promise_board_is_not_empty(pos.pieceBitboards_[coloredPiece<THEM, Piece::KING>()]);

    const Bitboard everyone = pos.colorBitboards_[Color::WHITE] | pos.colorBitboards_[Color::BLACK];

    const Bitboard ourRooklikePieces = pos.pieceBitboards_[coloredPiece<US, Piece::ROOK>()] | pos.pieceBitboards_[coloredPiece<US, Piece::QUEEN>()];
    const Bitboard theirRooklikePieces = pos.pieceBitboards_[coloredPiece<THEM, Piece::ROOK>()] | pos.pieceBitboards_[coloredPiece<THEM, Piece::QUEEN>()];
    const Bitboard ourBishoplikePieces = pos.pieceBitboards_[coloredPiece<US, Piece::BISHOP>()] | pos.pieceBitboards_[coloredPiece<US, Piece::QUEEN>()];
    const Bitboard theirBishoplikePieces = pos.pieceBitboards_[coloredPiece<THEM, Piece::BISHOP>()] | pos.pieceBitboards_[coloredPiece<THEM, Piece::QUEEN>()];

    Bitboard ourPawn1 = shift<kForwardRight>(pos.pieceBitboards_[coloredPiece<US, Piece::PAWN>()]);
    Bitboard ourPawn2 = shift<kForwardLeft>(pos.pieceBitboards_[coloredPiece<US, Piece::PAWN>()]);
    Bitboard theirPawn1 = shift<kBackwardRight>(pos.pieceBitboards_[coloredPiece<THEM, Piece::PAWN>()]);
    Bitboard theirPawn2 = shift<kBackwardLeft>(pos.pieceBitboards_[coloredPiece<THEM, Piece::PAWN>()]);

    const Bitboard ourKnights = pos.pieceBitboards_[coloredPiece<US, Piece::KNIGHT>()];
    const Bitboard ourBishops = pos.pieceBitboards_[coloredPiece<US, Piece::BISHOP>()];
    const Bitboard ourRooks = pos.pieceBitboards_[coloredPiece<US, Piece::ROOK>()];
    const Bitboard ourQueens = pos.pieceBitboards_[coloredPiece<US, Piece::QUEEN>()];
    const Bitboard ourKings = pos.pieceBitboards_[coloredPiece<US, Piece::KING>()];

    const Bitboard theirKnights = pos.pieceBitboards_[coloredPiece<THEM, Piece::KNIGHT>()];
    const Bitboard theirBishops = pos.pieceBitboards_[coloredPiece<THEM, Piece::BISHOP>()];
    const Bitboard theirRooks = pos.pieceBitboards_[coloredPiece<THEM, Piece::ROOK>()];
    const Bitboard theirQueens = pos.pieceBitboards_[coloredPiece<THEM, Piece::QUEEN>()];
    const Bitboard theirKings = pos.pieceBitboards_[coloredPiece<THEM, Piece::KING>()];

    // In general we assume "sane" positions -- no more than 2 knights, 2 bishops, 2 rooks, 1 queen (on each side).

    Bitboard ourKnight1_moves = kKnightMoves[lsb_or_none(ourKnights)];
    Bitboard ourKnight2_moves = kKnightMoves[msb_or_none(ourKnights)] * (std::popcount(ourKnights) > 1);
    Bitboard theirKnight1_moves = kKnightMoves[lsb_or_none(theirKnights)];
    Bitboard theirKnight2_moves = kKnightMoves[msb_or_none(theirKnights)] * (std::popcount(theirKnights) > 1);

    // Hard choice: do we include ourQueens in occupied?
    // YES: X is not typically threatend by "2 pieces" in this scenario, since X is usually defended by something less valuable than a queen [B Q X]
    //  NO: The case where we *do* want to count this as "2 attackers" is in attacks on the king, which are very important!

    // TODO: get rid of branches for sliding pieces (due to "sliding_moves" in "sliding.h")

    Bitboard ourBishops1_moves = ourBishops == kEmptyBitboard ? kEmptyBitboard : compute_one_bishops_targets(lsb_i_promise_board_is_not_empty(ourBishops), everyone & ~ourBishops);
    Bitboard ourBishops2_moves = std::popcount(ourBishops) < 2 ? kEmptyBitboard : compute_one_bishops_targets(msb_i_promise_board_is_not_empty(ourBishops), everyone & ~ourBishops);
    Bitboard theirBishops1_moves = theirBishops == kEmptyBitboard ? kEmptyBitboard : compute_one_bishops_targets(lsb_i_promise_board_is_not_empty(theirBishops), everyone & ~theirBishops);
    Bitboard theirBishops2_moves = std::popcount(theirBishops) < 2 ? kEmptyBitboard : compute_one_bishops_targets(msb_i_promise_board_is_not_empty(theirBishops), everyone & ~theirBishops);

    Bitboard ourRooks1_moves = ourRooks == kEmptyBitboard ? kEmptyBitboard : compute_single_rook_moves(lsb_i_promise_board_is_not_empty(ourRooks), everyone & ~ourRooks);
    Bitboard ourRooks2_moves = std::popcount(ourRooks) < 2 ? kEmptyBitboard : compute_single_rook_moves(msb_i_promise_board_is_not_empty(ourRooks), everyone & ~ourRooks);
    Bitboard theirRooks1_moves = theirRooks == kEmptyBitboard ? kEmptyBitboard : compute_single_rook_moves(lsb_i_promise_board_is_not_empty(theirRooks), everyone & ~theirRooks);
    Bitboard theirRooks2_moves = std::popcount(theirRooks) < 2 ? kEmptyBitboard : compute_single_rook_moves(msb_i_promise_board_is_not_empty(theirRooks), everyone & ~theirRooks);

    Bitboard ourQueenTargets = kEmptyBitboard;
    if (ourQueens != kEmptyBitboard) {
      ourQueenTargets |= compute_one_bishops_targets(lsb_i_promise_board_is_not_empty(ourQueens), everyone & ~ourBishops);
      ourQueenTargets |= compute_single_rook_moves(lsb_i_promise_board_is_not_empty(ourQueens), everyone & ~ourRooks);
    }
    Bitboard theirQueenTargets = kEmptyBitboard;
    if (theirQueens != kEmptyBitboard) {
      theirQueenTargets |= compute_one_bishops_targets(lsb_i_promise_board_is_not_empty(theirQueens), everyone & ~theirBishops & ~ourBishops);
      theirQueenTargets |= compute_single_rook_moves(lsb_i_promise_board_is_not_empty(theirQueens), everyone & ~theirRooks & ~ourRooks);
    }

    Bitboard ourKingTargets = kKingMoves[ourKingSq];
    Bitboard theirKingTargets = kKingMoves[theirKingSq];

    this->ourPawnTargets = ourPawn1 | ourPawn2;
    this->theirPawnTargets = theirPawn1 | theirPawn2;
    this->ourKnightTargets = ourKnight1_moves | ourKnight2_moves;
    this->theirKnightTargets = theirKnight1_moves | theirKnight2_moves;
    this->ourBishopTargets = ourBishops1_moves | ourBishops2_moves;
    this->theirBishopTargets = theirBishops1_moves | theirBishops2_moves;
    this->ourRookTargets = ourRooks1_moves | ourRooks2_moves;
    this->theirRookTargets = theirRooks1_moves | theirRooks2_moves;
    this->ourQueenTargets = ourQueenTargets;
    this->theirQueenTargets = ourQueenTargets;
    this->ourKingTargets = ourKingTargets;
    this->theirKingTargets = theirKingTargets;

    this->ourTargets = this->ourPawnTargets;
    this->theirTargets = this->theirPawnTargets;
    this->ourDoubleTargets = ourPawn1 & ourPawn2;
    this->theirDoubleTargets = theirPawn1 & theirPawn2;

    this->ourDoubleTargets |= ourKnight1_moves & this->ourTargets;
    this->ourTargets |= ourKnight1_moves;
    this->ourDoubleTargets |= ourKnight2_moves & this->ourTargets;
    this->ourTargets |= ourKnight2_moves;

    this->theirDoubleTargets |= theirKnight1_moves & this->theirTargets;
    this->theirTargets |= theirKnight1_moves;
    this->theirDoubleTargets |= theirKnight2_moves & this->theirTargets;
    this->theirTargets |= theirKnight2_moves;

    // Can speed this up by assuming bishops are on opposite colors.
    this->ourDoubleTargets |= this->ourBishopTargets & this->ourTargets;
    this->ourTargets |= this->ourBishopTargets;
    this->theirDoubleTargets |= (theirBishops1_moves | theirBishops2_moves) & this->theirTargets;
    this->theirTargets |= theirBishops1_moves | theirBishops2_moves;

    this->ourDoubleTargets |= ourRooks1_moves & this->ourTargets;
    this->ourTargets |= ourRooks1_moves;
    this->ourDoubleTargets |= ourRooks2_moves & this->ourTargets;
    this->ourTargets |= ourRooks2_moves;

    this->theirDoubleTargets |= theirRooks1_moves & this->theirTargets;
    this->theirTargets |= theirRooks1_moves;
    this->theirDoubleTargets |= theirRooks2_moves & this->theirTargets;
    this->theirTargets |= theirRooks2_moves;

    this->ourDoubleTargets |= ourQueenTargets & this->ourTargets;
    this->ourTargets |= ourQueenTargets;
    this->theirDoubleTargets |= theirQueenTargets & this->theirTargets;
    this->theirTargets |= theirQueenTargets;

    this->ourDoubleTargets |= ourKingTargets & this->ourTargets;
    this->ourTargets |= ourKingTargets;
    this->theirDoubleTargets |= theirKingTargets & this->theirTargets;
    this->theirTargets |= theirKingTargets;

    const Bitboard badForAllOfUs = this->theirTargets & ~this->ourTargets;
    const Bitboard badForAllOfThem = this->ourTargets & ~this->theirTargets;

    this->badForOur[Piece::PAWN] = badForAllOfUs;
    // Not defended by a pawn and attacked more than once.
    this->badForOur[Piece::PAWN] |= (~(ourPawn1 | ourPawn2)) & (this->theirDoubleTargets & ~this->ourDoubleTargets);
    // Defended by one pawn and attacked by a pawn and a piece.
    this->badForOur[Piece::PAWN] |= (ourPawn1 ^ ourPawn2) & (theirPawn1 | theirPawn2) & (this->theirDoubleTargets & ~this->ourDoubleTargets);

    this->badForTheir[Piece::PAWN] = badForAllOfThem;
    this->badForTheir[Piece::PAWN] |= (~(theirPawn1 & theirPawn2)) & (this->ourDoubleTargets & ~this->theirDoubleTargets);
    this->badForTheir[Piece::PAWN] |= (theirPawn1 ^ theirPawn2) & (ourPawn1 | ourPawn2) & (this->ourDoubleTargets & ~this->theirDoubleTargets);

    this->badForOur[Piece::PAWN] &= ~pos.colorBitboards_[THEM];
    this->badForTheir[Piece::PAWN] &= ~pos.colorBitboards_[US];

    // We exclude the piece being considered as a valid defender, since "badForOur" is frequently used to answer
    // the question "is it okay to move here?", and you can't keep defending a square if you move to it!

    // When computing double attacks, pieces that are more valuable than the piece being considered are merged
    // into a single attack -- if your bishop is attacked by two knights, it doesn't really matter if it is
    // defended by a king, a queen, and a rook -- it's still hanging!

    Bitboard ourTwo_def, ourTwo_atk, theirTwo_def, theirTwo_atk;
    Bitboard ourOne_def, ourOne_atk, theirOne_def, theirOne_atk;

    const Bitboard ourMinorTargets = this->ourKnightTargets | this->ourBishopTargets;
    const Bitboard theirMinorTargets = this->theirKnightTargets | this->theirBishopTargets;
    const Bitboard ourRoyalTargets = ourQueenTargets | ourKingTargets;
    const Bitboard theirRoyalTargets = theirQueenTargets | theirKingTargets;

    ourOne_def = (ourKnight1_moves & ourKnight2_moves) | this->ourBishopTargets | this->ourRookTargets | ourRoyalTargets;
    ourOne_atk = ourMinorTargets | this->ourRookTargets | ourRoyalTargets;
    theirOne_def = (theirKnight1_moves & theirKnight2_moves) | this->theirBishopTargets | this->theirRookTargets | theirRoyalTargets;
    theirOne_atk = theirMinorTargets | this->theirRookTargets | theirRoyalTargets;
    // Minor optimization: we merge both bishops since their targets are mutually exclusive 99% of the time.
    ourTwo_def = at_least_two(
      ourKnight1_moves & ourKnight2_moves, this->ourBishopTargets, this->ourRookTargets | ourRoyalTargets
    );
    ourTwo_atk = at_least_two(
      ourKnight1_moves, ourKnight2_moves, this->ourBishopTargets, this->ourRookTargets | ourRoyalTargets
    );
    theirTwo_def = at_least_two(
      theirKnight1_moves & theirKnight2_moves, this->theirBishopTargets, this->theirRookTargets | theirRoyalTargets
    );
    theirTwo_atk = at_least_two(
      theirKnight1_moves, theirKnight2_moves, this->theirBishopTargets, this->theirRookTargets | theirRoyalTargets
    );
    this->badForOur[Piece::KNIGHT] = badForAllOfUs | this->theirPawnTargets | (theirTwo_atk & ~ourTwo_def) | (theirOne_atk & ~ourOne_def);
    this->badForOur[Piece::KNIGHT] &= ~(theirKings | theirQueens | theirRooks | theirKnights | theirBishops);
    this->badForTheir[Piece::KNIGHT] = badForAllOfThem | this->ourPawnTargets | (ourTwo_atk & ~theirTwo_def) | (ourOne_atk & ~theirOne_def);
    this->badForTheir[Piece::KNIGHT] &= ~(ourKings | ourQueens | ourRooks | ourKnights | ourBishops);

    ourOne_def = this->ourKnightTargets | (ourBishops1_moves & ourBishops2_moves) | this->ourRookTargets | ourRoyalTargets;
    ourOne_atk = ourMinorTargets | this->ourRookTargets | ourRoyalTargets;
    theirOne_def = theirKnight1_moves | theirKnight2_moves | (theirBishops1_moves & theirBishops2_moves) | this->theirRookTargets | theirRoyalTargets;
    theirOne_atk = theirMinorTargets | this->theirRookTargets | theirRoyalTargets;
    ourTwo_def = at_least_two(ourKnight1_moves, ourKnight2_moves, this->ourRookTargets | ourRoyalTargets);
    ourTwo_atk = at_least_two(ourKnight1_moves, ourKnight2_moves, this->ourBishopTargets, this->ourRookTargets | ourRoyalTargets);
    theirTwo_def = at_least_two(theirKnight1_moves, theirKnight2_moves, this->theirRookTargets | theirRoyalTargets);
    theirTwo_atk = at_least_two(theirKnight1_moves, theirKnight2_moves, this->theirBishopTargets, this->theirRookTargets | theirRoyalTargets);
    this->badForOur[Piece::BISHOP] = badForAllOfUs | this->theirPawnTargets | (theirTwo_atk & ~ourTwo_def) | (theirOne_atk & ~ourOne_def);
    this->badForOur[Piece::BISHOP] &= ~(theirKings | theirQueens | theirRooks | theirKnights | theirBishops);
    this->badForTheir[Piece::BISHOP] = badForAllOfThem | this->ourPawnTargets | (ourTwo_atk & ~theirTwo_def) | (ourOne_atk & ~theirOne_def);
    this->badForTheir[Piece::BISHOP] &= ~(ourKings | ourQueens | ourRooks | ourKnights | ourBishops);

    ourOne_def = ourMinorTargets | (ourRooks1_moves & ourRooks2_moves) | ourRoyalTargets;
    ourOne_atk = ourMinorTargets | this->ourRookTargets | ourRoyalTargets;
    theirOne_def = theirMinorTargets | (theirRooks1_moves & theirRooks2_moves) | theirRoyalTargets;
    theirOne_atk = theirMinorTargets | this->theirRookTargets | theirRoyalTargets;
    ourTwo_def = at_least_two(ourKnight1_moves, ourKnight2_moves, this->ourBishopTargets, ourRooks1_moves & ourRooks2_moves, ourRoyalTargets);
    ourTwo_atk = at_least_two(ourKnight1_moves, ourKnight2_moves, this->ourBishopTargets, ourRooks1_moves, ourRooks2_moves, ourRoyalTargets);
    theirTwo_def = at_least_two(theirKnight1_moves, theirKnight2_moves, this->theirBishopTargets, theirRooks1_moves & theirRooks2_moves, theirRoyalTargets);
    theirTwo_atk = at_least_two(theirKnight1_moves, theirKnight2_moves, this->theirBishopTargets, theirRooks1_moves, theirRooks2_moves, theirRoyalTargets);
    this->badForOur[Piece::ROOK] = badForAllOfUs | (this->theirPawnTargets | theirMinorTargets) | (theirTwo_atk & ~ourTwo_def) | (theirOne_atk & ~ourOne_def);
    this->badForOur[Piece::ROOK] &= ~(theirKings | theirQueens | theirRooks);
    this->badForTheir[Piece::ROOK] = badForAllOfThem | (this->ourPawnTargets | ourMinorTargets) | (ourTwo_atk & ~theirTwo_def) | (ourOne_atk & ~theirOne_def);
    this->badForTheir[Piece::ROOK] &= ~(ourKings | ourQueens | ourRooks);

    this->badForOur[Piece::QUEEN] = badForAllOfUs | this->theirPawnTargets | theirMinorTargets | this->theirRookTargets | ((theirRoyalTargets) & ~this->ourTargets);
    this->badForOur[Piece::QUEEN] &= ~(theirKings | theirQueens);
    this->badForTheir[Piece::QUEEN] = badForAllOfThem | this->ourPawnTargets | ourMinorTargets | this->ourRookTargets | ((ourRoyalTargets) & ~this->theirTargets);
    this->badForTheir[Piece::QUEEN] &= ~(ourKings | ourQueens);

    this->badForOur[Piece::KING] = this->theirTargets;
    this->badForTheir[Piece::KING] = this->ourTargets;


    // this->ourPawnTargets = ourPawn1 | ourPawn2;
    // this->theirPawnTargets = theirPawn1 | theirPawn2;
    // this->ourKnightTargets = compute_knight_targets<US>(pos);
    // this->theirKnightTargets = compute_knight_targets<THEM>(pos);
    // this->ourBishopTargets = compute_bishoplike_targets(pos.pieceBitboards_[coloredPiece<US, Piece::BISHOP>()], everyone & ~pos.pieceBitboards_[coloredPiece<US, Piece::BISHOP>()]);
    // this->theirBishopTargets = compute_bishoplike_targets(pos.pieceBitboards_[coloredPiece<THEM, Piece::BISHOP>()], everyone & ~pos.pieceBitboards_[coloredPiece<THEM, Piece::BISHOP>()]);
    // this->ourRookTargets = compute_rooklike_targets(pos.pieceBitboards_[coloredPiece<US, Piece::ROOK>()], everyone & ~pos.pieceBitboards_[coloredPiece<US, Piece::ROOK>()]);
    // this->theirRookTargets = compute_rooklike_targets(pos.pieceBitboards_[coloredPiece<THEM, Piece::ROOK>()], everyone & ~pos.pieceBitboards_[coloredPiece<THEM, Piece::ROOK>()]);
    // this->ourQueenTargets = compute_bishoplike_targets(pos.pieceBitboards_[coloredPiece<US, Piece::QUEEN>()], everyone & ~ourBishoplikePieces) | compute_rooklike_targets(pos.pieceBitboards_[coloredPiece<US, Piece::QUEEN>()], everyone & ~ourRooklikePieces);
    // this->theirQueenTargets = compute_bishoplike_targets(pos.pieceBitboards_[coloredPiece<THEM, Piece::QUEEN>()], everyone & ~theirBishoplikePieces) | compute_rooklike_targets(pos.pieceBitboards_[coloredPiece<THEM, Piece::QUEEN>()], everyone & ~theirRooklikePieces);
    // this->ourKingTargets = compute_king_targets<US>(pos, ourKingSq);
    // this->theirKingTargets = compute_king_targets<THEM>(pos, theirKingSq);

    // const Bitboard ourPieces = pos.colorBitboards_[US] & ~pos.pieceBitboards_[coloredPiece<US, Piece::PAWN>()];
    // const Bitboard theirPieces = pos.colorBitboards_[THEM] & ~pos.pieceBitboards_[coloredPiece<THEM, Piece::PAWN>()];
    // const Bitboard ourMajors = ourPieces & ~pos.pieceBitboards_[coloredPiece<US, Piece::KNIGHT>()] & ~pos.pieceBitboards_[coloredPiece<US, Piece::BISHOP>()];
    // const Bitboard theirMajors = theirPieces & ~pos.pieceBitboards_[coloredPiece<THEM, Piece::KNIGHT>()] & ~pos.pieceBitboards_[coloredPiece<THEM, Piece::BISHOP>()];

    // this->ourTargets = ourPawn1 | ourPawn2;
    // this->ourDoubleTargets = ourPawn1 & ourPawn2;
    // this->theirTargets = theirPawn1 | theirPawn2;
    // this->theirDoubleTargets = theirPawn1 & theirPawn2;
    // { // Compute the above four variables.
    //   // Note: "ourDoubleTargets" and "theirDoubleTargets" are approximations, since
    //   // they ignore the possibility that two of the same piece can attack a square
    //   // (except for pawn double-attacks).
    //   this->ourDoubleTargets |= this->ourTargets & this->ourKnightTargets;
    //   this->ourTargets |= this->ourKnightTargets;
    //   this->theirDoubleTargets |= this->theirTargets & this->theirKnightTargets;
    //   this->theirTargets |= this->theirKnightTargets;

    //   this->ourDoubleTargets |= this->ourTargets & this->ourBishopTargets;
    //   this->ourTargets |= this->ourBishopTargets;
    //   this->theirDoubleTargets |= this->theirTargets & this->theirBishopTargets;
    //   this->theirTargets |= this->theirBishopTargets;

    //   this->ourDoubleTargets |= this->ourTargets & this->ourRookTargets;
    //   this->ourTargets |= this->ourRookTargets;
    //   this->theirDoubleTargets |= this->theirTargets & this->theirRookTargets;
    //   this->theirTargets |= this->theirRookTargets;

    //   this->ourDoubleTargets |= this->ourTargets & this->ourQueenTargets;
    //   this->ourTargets |= this->ourQueenTargets;
    //   this->theirDoubleTargets |= this->theirTargets & this->theirQueenTargets;
    //   this->theirTargets |= this->theirQueenTargets;

    //   this->ourDoubleTargets |= this->ourTargets & this->ourKingTargets;
    //   this->ourTargets |= this->ourKingTargets;
    //   this->theirDoubleTargets |= this->theirTargets & this->theirKingTargets;
    //   this->theirTargets |= this->theirKingTargets;
    // }

    // const Bitboard badForAllOfUs = this->theirTargets & ~this->ourTargets;
    // const Bitboard badForAllOfThem = this->ourTargets & ~this->theirTargets;

    // const Bitboard theirMinorTargets = this->theirKnightTargets | this->theirBishopTargets;
    // const Bitboard ourMinorTargets = this->ourKnightTargets | this->ourBishopTargets;

    // /**
    //  * Big brain: when computing badForOur[Knights], we should *exclude* one of our knights,
    //  * since we're using "badFor" to check if the knight can move there, and it cannot be
    //  * covering the square if it's moving there.
    //  */

    // this->badForOur[Piece::PAWN]   = (badForAllOfUs | (this->theirDoubleTargets & ~this->ourDoubleTargets)) & ~pos.colorBitboards_[THEM];
    // this->badForOur[Piece::KNIGHT] = (badForAllOfUs | this->theirPawnTargets | (this->theirDoubleTargets & ~this->ourDoubleTargets)) & ~theirPieces;
    // this->badForOur[Piece::BISHOP] = this->badForOur[Piece::KNIGHT];
    // this->badForOur[Piece::ROOK]   = (badForAllOfUs | this->theirPawnTargets | theirMinorTargets | (this->theirDoubleTargets & ~this->ourDoubleTargets)) & ~theirMajors;
    // this->badForOur[Piece::QUEEN]  = (badForAllOfUs | this->theirPawnTargets | theirMinorTargets | this->theirRookTargets | (this->theirDoubleTargets & ~this->ourDoubleTargets)) & ~pos.pieceBitboards_[coloredPiece<THEM, Piece::QUEEN>()];
    // this->badForOur[Piece::KING]   = this->theirTargets;

    // this->badForTheir[Piece::PAWN]   = (badForAllOfThem | (this->ourDoubleTargets & ~this->theirDoubleTargets)) & ~pos.colorBitboards_[US];
    // this->badForTheir[Piece::KNIGHT] = (badForAllOfThem | this->ourPawnTargets | (this->ourDoubleTargets & ~this->theirDoubleTargets)) & ~ourPieces;
    // this->badForTheir[Piece::BISHOP] = this->badForTheir[Piece::KNIGHT];
    // this->badForTheir[Piece::ROOK]   = (badForAllOfThem | this->ourPawnTargets | ourMinorTargets | (this->ourDoubleTargets & ~this->theirDoubleTargets & this->ourRookTargets)) & ~ourMajors;
    // this->badForTheir[Piece::QUEEN]  = (badForAllOfThem | this->ourPawnTargets | ourMinorTargets | this->ourRookTargets | (this->ourDoubleTargets & ~this->theirDoubleTargets)) & ~pos.pieceBitboards_[coloredPiece<US, Piece::QUEEN>()];
    // this->badForTheir[Piece::KING]   = this->ourTargets;
  }
};

}  // namespace ChessEngine

#endif  // THREATS_H