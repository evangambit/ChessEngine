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

void initialize_movegen() {
  initialize_sliding();
}

namespace StaticExchangeAnalysis {

template<Color US>
void simple_make_move(Position *pos, Square from, Square to) {
  const ColoredPiece movingCP = pos->tiles_[from];
  const ColoredPiece capturedPieceCP = pos->tiles_[to];
  pos->tiles_[to] = movingCP;
  pos->tiles_[from] = ColoredPiece::NO_COLORED_PIECE;
  pos->pieceBitboards_[movingCP] = pos->pieceBitboards_[movingCP] | bb(to) & ~bb(from);
  pos->pieceBitboards_[capturedPieceCP] &= ~bb(to);
}

template<Color US>
void simple_undo_move(Position *pos, Square from, Square to, ColoredPiece capturedPiece) {
  const ColoredPiece movingCP = pos->tiles_[to];
  pos->tiles_[to] = capturedPiece;
  pos->tiles_[from] = movingCP;
  pos->pieceBitboards_[movingCP] = pos->pieceBitboards_[movingCP] | bb(from) & ~bb(to);
  pos->pieceBitboards_[capturedPiece] |= bb(to);
}

template<Color US>
int static_exchange(Position *pos) {
  constexpr ColoredPiece ourPawnCP = coloredPiece<US, Piece::PAWN>();
  constexpr ColoredPiece ourKnightCP = coloredPiece<US, Piece::KNIGHT>();
  constexpr ColoredPiece ourBishopCP = coloredPiece<US, Piece::BISHOP>();
  constexpr ColoredPiece ourRookCP = coloredPiece<US, Piece::ROOK>();
  constexpr ColoredPiece ourQueenCP = coloredPiece<US, Piece::QUEEN>();

  constexpr Color THEM = opposite_color<US>();
  constexpr ColoredPiece theirQueenCP = coloredPiece<THEM, Piece::QUEEN>();
  constexpr ColoredPiece theirRookCP = coloredPiece<THEM, Piece::ROOK>();
  constexpr ColoredPiece theirBishopCP = coloredPiece<THEM, Piece::BISHOP>();
  constexpr ColoredPiece theirKnightCP = coloredPiece<THEM, Piece::KNIGHT>();

  constexpr Direction southeast = (US == Color::WHITE ? Direction::SOUTH_EAST : Direction::NORTH_WEST);
  constexpr Direction southwest = (US == Color::WHITE ? Direction::SOUTH_WEST : Direction::NORTH_EAST);

  constexpr int kPieceValues[7] = {0, 1, 3, 3, 5, 9, 99};

  if (compute_attackers<THEM>(*pos, lsb(pos->pieceBitboards_[coloredPiece<US, Piece::KING>()]))) {
    return kPieceValues[Piece::KING];
  }

  // Try all ways to capture enemy queen.
  if (pos->pieceBitboards_[theirQueenCP]) {
    Square queenSq = lsb(pos->pieceBitboards_[theirQueenCP]);
    Bitboard attackers = compute_attackers<THEM>(*pos, queenSq);
    for (Piece piece = Piece::PAWN; piece < Piece::QUEEN; piece = Piece(piece + 1)) {
      ColoredPiece cp = coloredPiece<US>(piece);
      if (attackers & pos->pieceBitboards_[cp]) {
        Square attackersSq = lsb(attackers & pos->pieceBitboards_[cp]);
        simple_make_move<US>(pos, attackersSq, queenSq);
        int r = (kPieceValues[Piece::QUEEN] - kPieceValues[Piece::PAWN]) - static_exchange<THEM>(pos);
        simple_undo_move<US>(pos, attackersSq, queenSq, theirQueenCP);
        return r;
      }
    }
  }

  // Try all ways to capture an enemy piece with a pawn.
  Bitboard pawnTargets = compute_pawn_targets<US>(*pos);
  for (Piece piece = Piece::ROOK; piece > Piece::PAWN; piece = Piece(piece - 1)) {
    Bitboard vulnerablePieces = pawnTargets & pos->pieceBitboards_[coloredPiece<THEM>(piece)];
    if (!vulnerablePieces) {
      continue;
    }
    Square targetSq = lsb(vulnerablePieces);
    Location targetLoc = bb(targetSq);
    Square attackersSq = lsb(pos->pieceBitboards_[ourPawnCP] & (shift<southeast>(targetLoc) | shift<southwest>(targetLoc)));
    simple_make_move<US>(pos, attackersSq, targetSq);
    int r = (kPieceValues[piece] - kPieceValues[Piece::PAWN]) - static_exchange<THEM>(pos);
    simple_undo_move<US>(pos, attackersSq, targetSq, theirRookCP);
    return r;
  }

  // TODO: minor piece capturing a rook.

  return 0;
}

}  // namespace StaticExchangeAnalysis

/**
 * There's a cool idea in Stockfish to have a "target" bitboard.
 * Only moves that move to the target are returned. This way we
 * can ask only for moves that block a check and/or capture the checker.
 */

// Absolutely required for castling.
// Also helps generate real moves (as opposed to pseudo-moves).
template<Color US>
bool can_enemy_attack(const Position& pos, Square sq) {
  return compute_attackers<opposite_color<US>()>(pos, sq) > 0;
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
Bitboard compute_my_targets_except_king(const Position& pos) {
  Bitboard r = compute_pawn_targets<US>(pos);
  r |= compute_knight_targets<US>(pos);
  const Bitboard bishopLikePieces = pos.pieceBitboards_[coloredPiece<US, Piece::BISHOP>()] | pos.pieceBitboards_[coloredPiece<US, Piece::QUEEN>()];
  r |= compute_bishoplike_targets<US>(pos, bishopLikePieces);
  const Bitboard rookLikePieces = pos.pieceBitboards_[coloredPiece<US, Piece::ROOK>()] | pos.pieceBitboards_[coloredPiece<US, Piece::QUEEN>()];
  r |= compute_rooklike_targets<US>(pos, rookLikePieces);
  return r;
}

template<Color US>
Bitboard compute_attackers(const Position& pos, const Square sq) {
  constexpr Color THEM = opposite_color<US>();

  const Bitboard ourRooks = pos.pieceBitboards_[coloredPiece<US, Piece::ROOK>()] | pos.pieceBitboards_[coloredPiece<US, Piece::QUEEN>()];
  const Bitboard ourBishops = pos.pieceBitboards_[coloredPiece<US, Piece::BISHOP>()] | pos.pieceBitboards_[coloredPiece<US, Piece::QUEEN>()];

  const Location loc = square2location(sq);
  const Bitboard enemies = pos.colorBitboards_[US];
  const Bitboard friends = pos.colorBitboards_[THEM] & ~loc;

  const Bitboard file = kFiles[sq % 8];
  const Bitboard rank = kRanks[sq / 8];

  Bitboard attackers = kEmptyBitboard;
  attackers |= (kKnightMoves[sq] & pos.pieceBitboards_[coloredPiece<US, Piece::KNIGHT>()]);
  attackers |= (kKingMoves[sq] & pos.pieceBitboards_[coloredPiece<US, Piece::KING>()]);

  attackers |= compute_pawn_attackers<US>(pos, loc);

  {  // Compute east/west moves.
    const uint8_t y = sq / 8;
    const unsigned rankShift = y * 8;
    uint8_t fromByte = loc >> rankShift;
    uint8_t occ = (enemies | friends) >> rankShift;
    uint8_t toByte = sliding_moves(fromByte, occ);
    Bitboard to = Bitboard(toByte) << rankShift;
    attackers |= (to & ourRooks);
  }

  {  // North/south attackers
    const uint8_t x = (sq % 8);
    const unsigned columnShift = 7 - x;
    uint8_t fromByte = (((loc << columnShift) & kFiles[7]) * kRookMagic) >> 56;
    uint8_t occ = (((((enemies | friends) & file) << columnShift) & kFiles[7]) * kRookMagic) >> 56;
    uint8_t toByte = sliding_moves(fromByte, occ);
    Bitboard to = (((Bitboard(toByte & 254) * kRookMagic) & kFiles[0]) | (toByte & 1)) << x;
    attackers |= (to & ourRooks);
  }

  {  // Southeast/Northwest diagonal.
    uint8_t occ = diag::southeast_diag_to_byte(sq, enemies | friends);
    uint8_t fromByte = diag::southeast_diag_to_byte(sq, loc);
    Bitboard to = diag::byte_to_southeast_diag(sq, sliding_moves(fromByte, occ));
    attackers |= (to & ourBishops);
  }
  {  // Southwest/Northeast diagonal.
    uint8_t occ = diag::southwest_diag_to_byte(sq, enemies | friends);
    uint8_t fromByte = diag::southwest_diag_to_byte(sq, loc);
    Bitboard to = diag::byte_to_southwest_diag(sq, sliding_moves(fromByte, occ));
    attackers |= (to & ourBishops);
  }

  return attackers;
}

struct CheckMap {
  Bitboard data[Piece::KING + 1];
};

template<Color US>
CheckMap compute_potential_attackers(const Position& pos, const Square sq) {
  constexpr Color THEM = opposite_color<US>();

  const Location loc = square2location(sq);

  const Bitboard file = kFiles[sq % 8];
  const Bitboard rank = kRanks[sq / 8];
  const Bitboard everyone = (pos.colorBitboards_[Color::WHITE] | pos.colorBitboards_[Color::BLACK]) & ~loc;

  CheckMap r;

  r.data[Piece::KNIGHT] = kKnightMoves[sq];
  r.data[Piece::KING] = kKingMoves[sq];

  // todo: pawns
  r.data[Piece::NO_PIECE] = 0;

  if (US == Color::WHITE) {
    r.data[Piece::PAWN] = shift<Direction::SOUTH_EAST>(loc) | shift<Direction::SOUTH_WEST>(loc);
  } else {
    r.data[Piece::PAWN] = shift<Direction::NORTH_EAST>(loc) | shift<Direction::NORTH_WEST>(loc);
  }

  r.data[Piece::ROOK] = 0;
  {
    const uint8_t y = sq / 8;
    const unsigned rankShift = y * 8;
    uint8_t fromByte = loc >> rankShift;
    uint8_t occ = everyone >> rankShift;
    uint8_t toByte = sliding_moves(fromByte, occ);
    r.data[Piece::ROOK] |= Bitboard(toByte) << rankShift;
  }
  {
    const uint8_t x = (sq % 8);
    const unsigned columnShift = 7 - x;
    uint8_t fromByte = (((loc << columnShift) & kFiles[7]) * kRookMagic) >> 56;
    uint8_t occ = ((((everyone & file) << columnShift) & kFiles[7]) * kRookMagic) >> 56;
    uint8_t toByte = sliding_moves(fromByte, occ);
    r.data[Piece::ROOK] |= (((Bitboard(toByte & 254) * kRookMagic) & kFiles[0]) | (toByte & 1)) << x;
  }

  r.data[Piece::BISHOP] = 0;
  {  // Southeast/Northwest diagonal.
    uint8_t occ = diag::southeast_diag_to_byte(sq, everyone);
    uint8_t fromByte = diag::southeast_diag_to_byte(sq, loc);
    r.data[Piece::BISHOP] |= diag::byte_to_southeast_diag(sq, sliding_moves(fromByte, occ));
  }
  {  // Southwest/Northeast diagonal.
    uint8_t occ = diag::southwest_diag_to_byte(sq, everyone);
    uint8_t fromByte = diag::southwest_diag_to_byte(sq, loc);
    r.data[Piece::BISHOP] |= diag::byte_to_southwest_diag(sq, sliding_moves(fromByte, occ));
  }

  r.data[Piece::QUEEN] = r.data[Piece::BISHOP] | r.data[Piece::ROOK];

  return r;
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
  Bitboard checkers = compute_attackers<enemyColor>(pos, ourKing);
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

  // TODO: if MGT == CHECKS_AND_CAPTURES we won't consider moves where the queen moves like
  // a bishop and checks like a rook (or vice versa).
  if (numCheckers > 0) {
    moves = compute_pawn_moves<US, MoveGenType::ALL_MOVES>(pos, moves, target, pm);
    moves = compute_knight_moves<US, MoveGenType::ALL_MOVES>(pos, moves, target, pm);
    moves = compute_king_moves<US, MoveGenType::ALL_MOVES, true>(pos, moves, validKingSquares);
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