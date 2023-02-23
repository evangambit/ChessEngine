#ifndef MOVEGEN_H
#define MOVEGEN_H

#include "Position.h"

#include "movegen/pawns.h"
#include "movegen/knights.h"
#include "movegen/kings.h"
#include "movegen/rooks.h"
#include "movegen/bishops.h"
#include "movegen/sliding.h"

#include <bit>

namespace ChessEngine {

/**
 * There's a cool idea in Stockfish to have a "target" bitboard.
 * Only moves that move to the target are returned. This way we
 * can ask only for moves that block a check and/or capture the checker.
 */

// Absolutely required for castling.
// Also helps generate real moves (as opposed to pseudo-moves).
template<Color US>
bool can_enemy_attack(const Position& pos, Square sq) {
  return compute_enemy_attackers<US>(pos, sq) > 0;
}

// 1 if any piece can get to a square.
// 0 if no piece can get to a square.
// Includes "self captures"
template<Color US>
Bitboard compute_my_targets(const Position& pos) {
  Bitboard r = compute_pawn_targets<US>(pos);
  r |= compute_knight_targets<US>(pos);
  const Bitboard bishopLikePieces = pos.pieceBitboards_[coloredPiece<US, Piece::BISHOP>()] | pos.pieceBitboards_[coloredPiece<US, Piece::QUEEN>()];
  r |= compute_bishoplike_targets<US>(pos, bishopLikePieces);
  const Bitboard rookLikePieces = pos.pieceBitboards_[coloredPiece<US, Piece::ROOK>()] | pos.pieceBitboards_[coloredPiece<US, Piece::QUEEN>()];
  r |= compute_rooklike_targets<US>(pos, rookLikePieces);
  const Square kingSq = lsb(pos.pieceBitboards_[coloredPiece<US, Piece::KING>()]);
  r |= compute_king_targets<US>(pos, kingSq);
  return r;
}

template<Color US>
Bitboard compute_enemy_attackers(const Position& pos, const Square sq) {
  constexpr Color enemyColor = opposite_color<US>();

  const Bitboard enemyRooks = pos.pieceBitboards_[coloredPiece<enemyColor, Piece::ROOK>()] | pos.pieceBitboards_[coloredPiece<enemyColor, Piece::QUEEN>()];
  const Bitboard enemyBishops = pos.pieceBitboards_[coloredPiece<enemyColor, Piece::BISHOP>()] | pos.pieceBitboards_[coloredPiece<enemyColor, Piece::QUEEN>()];

  const Location loc = square2location(sq);
  const Bitboard enemies = pos.colorBitboards_[opposite_color<US>()];
  const Bitboard friends = pos.colorBitboards_[US] & ~loc;

  const Bitboard file = kFiles[sq % 8];
  const Bitboard rank = kRanks[sq / 8];

  Bitboard attackers = kEmptyBitboard;
  attackers |= (kKnightMoves[sq] & pos.pieceBitboards_[coloredPiece<enemyColor, Piece::KNIGHT>()]);
  attackers |= (kKingMoves[sq] & pos.pieceBitboards_[coloredPiece<enemyColor, Piece::KING>()]);

  attackers |= compute_enemy_pawn_attackers<US>(pos, loc);

  {  // Compute east/west moves.
    const uint8_t y = sq / 8;
    const unsigned rankShift = y * 8;
    uint8_t fromByte = loc >> rankShift;
    uint8_t occ = (enemies | friends) >> rankShift;
    uint8_t toByte = sliding_moves(fromByte, occ);
    Bitboard to = Bitboard(toByte) << rankShift;
    attackers |= (to & enemyRooks);
  }

  {  // North/south attackers
    const uint8_t x = (sq % 8);
    const unsigned columnShift = 7 - x;
    uint8_t fromByte = (((loc << columnShift) & kFiles[7]) * kRookMagic) >> 56;
    uint8_t occ = (((((enemies | friends) & file) << columnShift) & kFiles[7]) * kRookMagic) >> 56;
    uint8_t toByte = sliding_moves(fromByte, occ);
    Bitboard to = (((Bitboard(toByte & 254) * kRookMagic) & kFiles[0]) | (toByte & 1)) << x;
    attackers |= (to & enemyRooks);
  }

  {  // Southeast/Northwest diagonal.
    uint8_t occ = diag::southeast_diag_to_byte(sq, enemies | friends);
    uint8_t fromByte = diag::southeast_diag_to_byte(sq, loc);
    Bitboard to = diag::byte_to_southeast_diag(sq, sliding_moves(fromByte, occ));
    attackers |= (to & enemyBishops);
  }
  {  // Southwest/Northeast diagonal.
    uint8_t occ = diag::southwest_diag_to_byte(sq, enemies | friends);
    uint8_t fromByte = diag::southwest_diag_to_byte(sq, loc);
    Bitboard to = diag::byte_to_southwest_diag(sq, sliding_moves(fromByte, occ));
    attackers |= (to & enemyBishops);
  }

  return attackers;
}

template<Color US>
PinMasks compute_pin_masks(const Position& pos) {
  constexpr Color THEM = opposite_color<US>();

  const Bitboard ourPieces = pos.colorBitboards_[US];
  const Bitboard occ = pos.colorBitboards_[US] | pos.colorBitboards_[THEM];
  const Bitboard ourKings = pos.pieceBitboards_[coloredPiece<US, Piece::KING>()];
  const Square ourKingSq = lsb(ourKings);
  const unsigned y = ourKingSq / 8;
  const unsigned x = ourKingSq % 8;

  const Bitboard enemyRooks = pos.pieceBitboards_[coloredPiece<THEM, Piece::ROOK>()] | pos.pieceBitboards_[coloredPiece<THEM, Piece::QUEEN>()];
  const Bitboard enemyBishops = pos.pieceBitboards_[coloredPiece<THEM, Piece::BISHOP>()] | pos.pieceBitboards_[coloredPiece<THEM, Piece::QUEEN>()];

  PinMasks r;

  {  // Compute east/west moves.
    const Bitboard rank = kRanks[y];
    const unsigned rankShift = y * 8;
    uint8_t kingByte = ourKings >> rankShift;
    uint8_t occByte = occ >> rankShift;
    uint8_t enemiesByte = enemyRooks >> rankShift;
    r.horizontal = Bitboard(sliding_pinmask(kingByte, occByte, enemiesByte)) << rankShift;
  }

  {  // Compute north/south moves.
    const Bitboard file = kFiles[x];
    const unsigned columnShift = 7 - x;
    uint8_t kingByte = (((ourKings << columnShift) & kFiles[7]) * kRookMagic) >> 56;
    uint8_t occByte = (((occ << columnShift) & kFiles[7]) * kRookMagic) >> 56;
    uint8_t enemiesByte = (((enemyRooks << columnShift) & kFiles[7]) * kRookMagic) >> 56;
    uint8_t toByte = sliding_pinmask(kingByte, occByte, enemiesByte);
    r.vertical = (((Bitboard(toByte & 254) * kRookMagic) & kFiles[0]) | (toByte & 1)) << x;
  }

  {  // Southeast/Northwest diagonal.
    uint8_t occByte = diag::southeast_diag_to_byte(ourKingSq, occ);
    uint8_t kingByte = diag::southeast_diag_to_byte(ourKingSq, ourKings);
    uint8_t enemiesByte = diag::southeast_diag_to_byte(ourKingSq, enemyBishops);
    r.northwest = diag::byte_to_southeast_diag(ourKingSq, sliding_pinmask(kingByte, occByte, enemiesByte));
  }
  {  // Southwest/Northeast diagonal.
    uint8_t occByte = diag::southwest_diag_to_byte(ourKingSq, occ);
    uint8_t kingByte = diag::southwest_diag_to_byte(ourKingSq, ourKings);
    uint8_t enemiesByte = diag::southwest_diag_to_byte(ourKingSq, enemyBishops);
    r.northeast = diag::byte_to_southwest_diag(ourKingSq, sliding_pinmask(kingByte, occByte, enemiesByte));
  }

  r.all = r.horizontal | r.vertical | r.northwest | r.northeast;

  return r;
}

// We take the liberty of ignoring MGT if you're in check.
template<Color US, MoveGenType MGT>
ExtMove* compute_moves(const Position& pos, ExtMove *moves) {
  constexpr Color enemyColor = opposite_color<US>();
  assert(US == pos.turn_);
  const Bitboard ourKings = pos.pieceBitboards_[coloredPiece<US, Piece::KING>()];
  const Bitboard theirKings = pos.pieceBitboards_[coloredPiece<enemyColor, Piece::KING>()];
  if (std::popcount(ourKings | theirKings) != 2) {
    // Game over, no legal moves.
    return moves;
  }
  const Square ourKing = lsb(ourKings);
  Bitboard checkers = compute_enemy_attackers<US>(pos, ourKing);
  const PinMasks pm = compute_pin_masks<US>(pos);

  const unsigned numCheckers = std::popcount(checkers);

  if (numCheckers > 1) {  // Double check; king must move.
    return compute_king_moves<US, MoveGenType::ALL_MOVES, true>(pos, moves, kUniverse);
  }

  Bitboard target = kUniverse;
  if (numCheckers == 1) {
    target = kSquaresBetween[ourKing][lsb(checkers)];
  }

  const Bitboard validKingSquares = ~compute_my_targets<opposite_color<US>()>(pos);

  if (numCheckers > 0) {
    moves = compute_pawn_moves<US, MoveGenType::ALL_MOVES>(pos, moves, target, pm);
    moves = compute_knight_moves<US, MoveGenType::ALL_MOVES>(pos, moves, target, pm);
    moves = compute_king_moves<US, MGT, true>(pos, moves, validKingSquares);
    moves = compute_bishop_like_moves<US, MoveGenType::ALL_MOVES>(pos, moves, target, pm);
    moves = compute_rook_like_moves<US, MoveGenType::ALL_MOVES>(pos, moves, target, pm);
  } else {
    moves = compute_pawn_moves<US, MGT>(pos, moves, target, pm);
    moves = compute_knight_moves<US, MGT>(pos, moves, target, pm);
    moves = compute_king_moves<US, MGT, false>(pos, moves, validKingSquares);
    moves = compute_bishop_like_moves<US, MGT>(pos, moves, target, pm);
    moves = compute_rook_like_moves<US, MGT>(pos, moves, target, pm);
  }
  return moves;
}

template<Color US>
ExtMove* compute_legal_moves(Position *pos, ExtMove *moves) {
  ExtMove pseudoMoves[kMaxNumMoves];
  ExtMove *end = compute_moves<US, MoveGenType::ALL_MOVES>(*pos, pseudoMoves);
  for (ExtMove *move = pseudoMoves; move < end; ++move) {
    make_move<US>(pos, move->move);
    Square sq = lsb(pos->pieceBitboards_[coloredPiece<US,Piece::KING>()]);
    if (can_enemy_attack<US>(*pos, sq) == 0) {
      (*moves++) = *move;
    // } else {
    //   undo<US>(pos);
    //   PinMasks pm = compute_pin_masks<US>(*pos);
    //   std::cout << bstr(pm.horizontal) << std::endl;
    //   std::cout << bstr(pm.vertical) << std::endl;
    //   std::cout << bstr(pm.northeast) << std::endl;
    //   std::cout << bstr(pm.northwest) << std::endl;
    //   std::cout << *pos << std::endl;
    //   std::cout << pos->fen() << std::endl;
    //   std::cout << *move << std::endl;
    //   throw std::runtime_error("");
    }
    undo<US>(pos);
  }
  return moves;
}

}  // namespace ChessEngine

#endif  // MOVEGEN_H