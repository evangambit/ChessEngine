#ifndef POSITION_H
#define POSITION_H

#include <cassert>
#include <cstdint>

#include <string>
#include <vector>
#include <algorithm>

#include "geometry.h"
#include "utils.h"

namespace ChessEngine {

std::vector<std::string> split(const std::string& text, char delimiter);

std::string join(const std::vector<std::string>& A, const std::string& delimiter);

enum MoveType {
  NORMAL = 0,
  EN_PASSANT = 1,
  CASTLE = 2,
  PROMOTION = 3,
};

struct Move {
  Square from : 6;
  Square to : 6;
  unsigned promotion : 2;  // knight, bishop, rook, queen
  MoveType moveType : 2;

  std::string uci() const;

  bool operator==(const Move& a) const {
    return from == a.from && to == a.to && promotion == a.promotion && moveType == a.moveType;
  }
};

struct ExtMove {
  ExtMove() {}
  ExtMove(Piece piece, Move move) : piece(piece), capture(Piece::NO_PIECE), move(move) {}
  ExtMove(Piece piece, Piece capture, Move move) : piece(piece), capture(capture), move(move) {}

  std::string str() const;

  std::string uci() const;

  Piece piece : 4;
  Piece capture : 4;
  Move move;  // 16 bits
  Evaluation score;  // 16 bits
};

std::ostream& operator<<(std::ostream& stream, const Move move);
std::ostream& operator<<(std::ostream& stream, const ExtMove move);

const Move kNullMove = Move{Square(0), Square(0), 0, MoveType::NORMAL};

struct PositionState {
  CastlingRights castlingRights;
  uint8_t halfMoveCounter;
  Square epSquare;
  uint64_t hash;  // Required for 3-move draw.
};

extern uint64_t kZorbristNumbers[ColoredPiece::NUM_COLORED_PIECES][kNumSquares];
extern uint64_t kZorbristCastling[16];
extern uint64_t kZorbristEnpassant[8];
extern uint64_t kZorbristTurn;

void initialize_zorbrist();

class Position {
 public:
  Position() : turn_(Color::WHITE) {
    this->_empty_();
  };
  Position(const std::string& fen);

  static Position init();

  std::string fen() const;

  std::string san(Move move) const;

  ColoredPiece tiles_[kNumSquares];
  Bitboard pieceBitboards_[ColoredPiece::NUM_COLORED_PIECES];
  Bitboard colorBitboards_[Color::NUM_COLORS];

  std::vector<PositionState> states_;
  std::vector<ExtMove> history_;
  std::vector<uint64_t> hashes_;
  PositionState currentState_;

  uint64_t hash_;

  // Incremented after a black move.
  uint32_t wholeMoveCounter_;
  Color turn_;

  void place_piece_(ColoredPiece cp, Square square);

  void remove_piece_(Square square);

  void assert_valid_state() const;
  void assert_valid_state(const std::string& msg) const;

  void update_hash_on_state_change(PositionState a, PositionState b);

 private:
  void _empty_();
};

#ifndef NDEBUG
extern Position *gDebugPos;
#endif

std::ostream& operator<<(std::ostream& stream, const Position& pos);

namespace {

// Maps (0 -> 0), (7 -> 1), (56 -> 2), and (63 -> 3)
uint8_t four_corners_to_byte(Bitboard b) {
  constexpr Bitboard mask = bb(0) | bb(7) | bb(56) | bb(63);
  return ((b & mask) * 0x1040000000000041) >> 60;
}

// Maps (4 -> {0, 1}) and (60 -> {3, 4})
uint8_t king_starts_to_byte(Bitboard b) {
  constexpr Bitboard mask = bb(4) | bb(60);
  constexpr Bitboard magic = bb(6) | bb(7) | bb(60) | bb(61);
  return (((b & mask) >> 4) * magic) >> 60;
}

constexpr Bitboard kKingStartingPosition = bb(4) | bb(60);

}  // namespace

// TODO: don't need to unupdate hash; can just use pos->hashes.back()
template<Color MOVER_TURN>
void undo(Position *pos) {
  pos->assert_valid_state();
  assert(pos->history_.size() > 0);
  assert(pos->turn_ == opposite_color<MOVER_TURN>());
  pos->hashes_.pop_back();

  pos->update_hash_on_state_change(pos->currentState_, pos->states_.back());

  pos->turn_ = MOVER_TURN;
  pos->hash_ ^= kZorbristTurn;

  const ExtMove extMove = pos->history_.back();
  pos->currentState_ = pos->states_.back();
  pos->history_.pop_back();
  pos->states_.pop_back();
  if (MOVER_TURN == Color::BLACK) {
    pos->wholeMoveCounter_ -= 1;
  }

  const Move move = extMove.move;
  const ColoredPiece movingPiece = move.moveType == MoveType::PROMOTION ? coloredPiece<MOVER_TURN, Piece::PAWN>() : pos->tiles_[move.to];
  const ColoredPiece capturedPiece = coloredPiece<opposite_color<MOVER_TURN>()>(extMove.capture);
  const ColoredPiece promoPiece = move.moveType == MoveType::PROMOTION ? coloredPiece<MOVER_TURN>(Piece(move.promotion + 2)) : movingPiece;
  const Location f = square2location(move.from);
  const Location t = square2location(move.to);

  pos->pieceBitboards_[movingPiece] |= f;
  pos->pieceBitboards_[promoPiece] &= ~t;
  pos->colorBitboards_[MOVER_TURN] |= f;
  pos->colorBitboards_[MOVER_TURN] &= ~t;
  pos->tiles_[move.from] = movingPiece;
  pos->tiles_[move.to] = capturedPiece;
  pos->hash_ ^= kZorbristNumbers[movingPiece][move.from];
  pos->hash_ ^= kZorbristNumbers[promoPiece][move.to];

  const bool hasCapturedPiece = (capturedPiece != ColoredPiece::NO_COLORED_PIECE);
  pos->pieceBitboards_[capturedPiece] |= t;
  pos->colorBitboards_[opposite_color<MOVER_TURN>()] |= t * hasCapturedPiece;
  pos->hash_ ^= kZorbristNumbers[capturedPiece][move.to] * hasCapturedPiece;

  if (move.moveType == MoveType::CASTLE) {
    if (MOVER_TURN == Color::BLACK) {
      assert(move.from == 4);
      assert(move.to == 2 || move.to == 6);
    } else {
      assert(move.from == 60);
      assert(move.to == 62 || move.to == 58);
    }
    uint16_t rookDestination = (uint16_t(move.from) + uint16_t(move.to)) / 2;
    uint16_t rookOrigin = ((uint16_t(move.to) % 8) * 7 - 14) / 4 + (MOVER_TURN == Color::WHITE ? 56 : 0);

    Bitboard rookDestinationBB = bb(rookDestination);
    Bitboard rookOriginBB = bb(rookOrigin);

    const ColoredPiece myRookPiece = coloredPiece<MOVER_TURN, Piece::ROOK>();
    pos->pieceBitboards_[myRookPiece] |= rookOriginBB;
    pos->pieceBitboards_[myRookPiece] &= ~rookDestinationBB;
    pos->colorBitboards_[MOVER_TURN] |= rookOriginBB;
    pos->colorBitboards_[MOVER_TURN] &= ~rookDestinationBB;
    pos->tiles_[rookDestination] = ColoredPiece::NO_COLORED_PIECE;
    pos->tiles_[rookOrigin] = myRookPiece;
    pos->hash_ ^= kZorbristNumbers[myRookPiece][rookOrigin] * hasCapturedPiece;
    pos->hash_ ^= kZorbristNumbers[myRookPiece][rookDestination] * hasCapturedPiece;
  }

  if (move.moveType == MoveType::EN_PASSANT) {
    // TODO: get rid of if statement
    if (MOVER_TURN == Color::BLACK) {
      assert(move.from / 8 == 5);
      assert(move.to / 8 == 6);
    } else {
      assert(move.from / 8 == 3);
      assert(move.to / 8 == 2);
    }

    constexpr Color opposingColor = opposite_color<MOVER_TURN>();
    uint8_t enpassantLoc = (MOVER_TURN == Color::WHITE ? move.to + 8 : move.to - 8);
    Bitboard enpassantLocBB = bb(enpassantLoc);

    constexpr ColoredPiece opposingPawn = coloredPiece<opposingColor, Piece::PAWN>();

    assert(pos->tiles_[lsb(enpassantLocBB)] == opposingPawn);

    pos->pieceBitboards_[opposingPawn] |= enpassantLocBB;
    pos->colorBitboards_[opposingColor] |= enpassantLocBB;
    pos->hash_ ^= kZorbristNumbers[opposingPawn][enpassantLoc];
  }

  pos->assert_valid_state();
}

template<Color TURN>
void foo(Position *pos) {
  pos->assert_valid_state();
}

template<Color TURN>
void bar(Position *pos) {
  pos->assert_valid_state();
}

template<Color TURN>
void make_nullmove(Position *pos) {
  pos->hashes_.push_back(pos->hash_);

  pos->states_.push_back(pos->currentState_);

  pos->history_.push_back(ExtMove(Piece::NO_PIECE, Piece::NO_PIECE, kNullMove));

  pos->currentState_.epSquare = Square::NO_SQUARE;

  if (TURN == Color::BLACK) {
    pos->wholeMoveCounter_ += 1;
  }
  ++pos->currentState_.halfMoveCounter;
  pos->turn_ = opposite_color<TURN>();
  pos->hash_ ^= kZorbristTurn;

  pos->update_hash_on_state_change(pos->states_.back(), pos->currentState_);
}

template<Color MOVER_TURN>
void undo_nullmove(Position *pos) {
  pos->hash_ = pos->hashes_.back();
  pos->hashes_.pop_back();

  pos->currentState_ = pos->states_.back();
  pos->states_.pop_back();

  assert(pos->history_.back().move == kNullMove);
  pos->history_.pop_back();

  if (MOVER_TURN == Color::BLACK) {
    pos->wholeMoveCounter_ -= 1;
  }

  pos->turn_ = MOVER_TURN;
}

template<Color TURN>
void make_move(Position *pos, Move move) {
  foo<TURN>(pos);

  pos->hashes_.push_back(pos->hash_);

  const ColoredPiece movingPiece = pos->tiles_[move.from];
  const ColoredPiece promoPiece = move.moveType == MoveType::PROMOTION ? coloredPiece<TURN>(Piece(move.promotion + 2)) : movingPiece;
  constexpr Color opposingColor = opposite_color<TURN>();
  const ColoredPiece capturedPiece = pos->tiles_[move.to];

  #ifndef NDEBUG
  if (pos->turn_ != TURN) {
    gDebugPos = new Position(*pos);
    std::cout << "=== " << move.uci() << " ===" << std::endl;
    throw std::runtime_error("make_move a");
  }
  if (cp2color(promoPiece) != TURN) {
    gDebugPos = new Position(*pos);
    std::cout << "=== " << move.uci() << " ===" << std::endl;
    throw std::runtime_error("make_move b");
  }
  if (movingPiece == ColoredPiece::NO_COLORED_PIECE) {
    gDebugPos = new Position(*pos);
    std::cout << "=== " << move.uci() << " ===" << std::endl;
    throw std::runtime_error("make_move c");
  }
  if (cp2color(movingPiece) != TURN) {
    gDebugPos = new Position(*pos);
    std::cout << "=== " << move.uci() << " ===" << std::endl;
    throw std::runtime_error("make_move d");
  }
  if (cp2color(capturedPiece) == TURN) {
    gDebugPos = new Position(*pos);
    std::cout << "=== " << move.uci() << " ===" << std::endl;
    throw std::runtime_error("make_move e");
  }
  #endif

  const Location f = square2location(move.from);
  const Location t = square2location(move.to);

  pos->states_.push_back(pos->currentState_);
  pos->history_.push_back(ExtMove(cp2p(movingPiece), cp2p(capturedPiece), move));

  // Remove castling rights if a rook moves or is captured.
  pos->currentState_.castlingRights &= ~four_corners_to_byte(f);
  pos->currentState_.castlingRights &= ~four_corners_to_byte(t);

  // TODO: remove if statements
  if (TURN == Color::WHITE) {
    if (movingPiece == coloredPiece<TURN, Piece::PAWN>() && move.from - move.to == 16) {
      pos->currentState_.epSquare = Square(move.to + 8);
    } else {
      pos->currentState_.epSquare = Square::NO_SQUARE;
    }
  } else {
    if (movingPiece == coloredPiece<TURN, Piece::PAWN>() && move.to - move.from == 16) {
      pos->currentState_.epSquare = Square(move.to - 8);
    } else {
      pos->currentState_.epSquare = Square::NO_SQUARE;
    }
  }

  // Remove castling rights if a king moves. both lines are equivalent but the
  // second removes a multiplication.
  // currentState_.castlingRights &= ~king_starts_to_byte(from);
  pos->currentState_.castlingRights &= ~((
    ((f & kKingStartingPosition) > 0)
    |
    (((f & kKingStartingPosition) > 0) << 1)
  ) << ((2 - TURN) * 2));

  // Move the piece.
  pos->pieceBitboards_[movingPiece] &= ~f;
  pos->pieceBitboards_[promoPiece] |= t;
  pos->colorBitboards_[TURN] &= ~f;
  pos->colorBitboards_[TURN] |= t;
  pos->tiles_[move.to] = promoPiece;
  pos->tiles_[move.from] = ColoredPiece::NO_COLORED_PIECE;
  pos->hash_ ^= kZorbristNumbers[movingPiece][move.from];
  pos->hash_ ^= kZorbristNumbers[promoPiece][move.to];

  // Remove captured piece.
  pos->pieceBitboards_[capturedPiece] &= ~t;
  pos->colorBitboards_[opposingColor] &= ~t;
  const bool hasCapturedPiece = (capturedPiece != ColoredPiece::NO_COLORED_PIECE);
  pos->hash_ ^= kZorbristNumbers[capturedPiece][move.to] * hasCapturedPiece;

  if (move.moveType == MoveType::CASTLE) {
    // TODO: get rid of if statement
    if (TURN == Color::BLACK) {
      assert(move.from == 4);
      assert(move.to == 2 || move.to == 6);
    } else {
      assert(move.from == 60);
      assert(move.to == 62 || move.to == 58);
    }
    uint8_t rookDestination = (uint16_t(move.from) + uint16_t(move.to)) / 2;
    uint8_t rookOrigin = ((uint16_t(move.to) % 8) * 7 - 14) / 4 + (TURN == Color::WHITE ? 56 : 0);

    Bitboard rookDestinationBB = bb(rookDestination);
    Bitboard rookOriginBB = bb(rookOrigin);

    constexpr ColoredPiece myRookPiece = coloredPiece<TURN, Piece::ROOK>();
    pos->pieceBitboards_[myRookPiece] &= ~rookOriginBB;
    pos->pieceBitboards_[myRookPiece] |= rookDestinationBB;
    pos->colorBitboards_[TURN] &= ~rookOriginBB;
    pos->colorBitboards_[TURN] |= rookDestinationBB;
    pos->tiles_[rookOrigin] = ColoredPiece::NO_COLORED_PIECE;
    pos->tiles_[rookDestination] = myRookPiece;
    pos->hash_ ^= kZorbristNumbers[myRookPiece][rookOrigin] * hasCapturedPiece;
    pos->hash_ ^= kZorbristNumbers[myRookPiece][rookDestination] * hasCapturedPiece;
  }

  if (move.moveType == MoveType::EN_PASSANT) {
    // TODO: get rid of if statement
    if (TURN == Color::BLACK) {
      assert(move.from / 8 == 5);
      assert(move.to / 8 == 6);
    } else {
      assert(move.from / 8 == 3);
      assert(move.to / 8 == 2);
    }
    uint8_t enpassantLoc = (TURN == Color::WHITE ? move.to + 8 : move.to - 8);
    Bitboard enpassantLocBB = bb(enpassantLoc);

    constexpr ColoredPiece opposingPawn = coloredPiece<opposingColor, Piece::PAWN>();

    assert(pos->tiles_[lsb(enpassantLocBB)] == opposingPawn);

    pos->pieceBitboards_[opposingPawn] &= ~enpassantLocBB;
    pos->colorBitboards_[opposingColor] &= ~enpassantLocBB;
    pos->hash_ ^= kZorbristNumbers[opposingPawn][enpassantLoc];
  }

  if (TURN == Color::BLACK) {
    pos->wholeMoveCounter_ += 1;
  }
  ++pos->currentState_.halfMoveCounter;
  pos->currentState_.halfMoveCounter *= (movingPiece != coloredPiece<TURN, Piece::PAWN>());
  pos->turn_ = opposingColor;
  pos->hash_ ^= kZorbristTurn;

  pos->update_hash_on_state_change(pos->states_.back(), pos->currentState_);

  bar<TURN>(pos);

}

}  // namespace ChessEngine

#endif  // POSITION_H