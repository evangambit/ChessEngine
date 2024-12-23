#ifndef POSITION_H
#define POSITION_H

#include <cassert>
#include <cstdint>
#include <cstring>

#include <string>
#include <vector>
#include <algorithm>
#include <memory>

#include "geometry.h"
#include "piece_maps.h"
#include "utils.h"

#if NNUE_EVAL
#include "nnue.h"
#endif

namespace ChessEngine {

template<class T>
std::string join(const T& A, const std::string& delimiter) {
  std::string r = "";
  for (size_t i = 0; i < A.size(); ++i) {
    r += A[i];
    if (i != A.size() - 1) {
      r += delimiter;
    }
  }
  return r;
}

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
    return std::memcmp(this, &a, sizeof(Move)) == 0;
  }
};

struct ExtMove {
  ExtMove() {}
  ExtMove(Piece piece, Move move) : piece(piece), capture(ColoredPiece::NO_COLORED_PIECE), move(move) {}
  ExtMove(Piece piece, ColoredPiece capture2, Move move) : piece(piece), capture(capture2), move(move) {}

  std::string str() const;

  std::string uci() const;

  Piece piece : 4;
  ColoredPiece capture : 4;
  Move move;  // 16 bits
  Evaluation score;  // 16 bits
};

std::ostream& operator<<(std::ostream& stream, const Move move);
std::ostream& operator<<(std::ostream& stream, const ExtMove move);

const Move kNullMove = Move{Square(0), Square(0), 0, MoveType::NORMAL};
const ExtMove kNullExtMove = ExtMove(Piece::NO_PIECE, kNullMove);

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
  Position() : turn_(Color::WHITE), pieceMaps_(&kZeroPieceMap) {
    this->turn_ = Color::WHITE;
    this->_empty_();
    #if NNUE_EVAL
    this->network = std::make_shared<NnueNetworkInterface>();
    #endif
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

  #if NNUE_EVAL
  std::shared_ptr<NnueNetworkInterface> network;
  #endif

  void set_piece_maps(const PieceMaps& pieceMaps) {
    pieceMaps_ = &pieceMaps;
    for (int i = 0; i < PieceMapType::PieceMapTypeCount; ++i) {
      pieceMapScores[i] = 0;
    }
    for (size_t i = 0; i < kNumSquares; ++i) {
      this->increment_piece_map(tiles_[i], Square(i));
    }
  }
  PieceMaps const * pieceMaps_;

  #if NNUE_EVAL
  void set_network(std::shared_ptr<NnueNetworkInterface> network) {
    this->network = network;

    network->empty();
    for (int sq = 0; sq < kNumSquares; ++sq) {
      ColoredPiece cp = this->tiles_[sq];
      if (cp == ColoredPiece::NO_COLORED_PIECE) {
        continue;
      }
      network->set_piece(cp, Square(sq), 1);
    }

    network->set_index(NnueFeatures::NF_IS_WHITE_TURN, (this->turn_ == Color::WHITE));
    network->set_index(NnueFeatures::NF_WHITE_KING_CASTLING, ((this->currentState_.castlingRights & kCastlingRights_WhiteKing) > 0));
    network->set_index(NnueFeatures::NF_WHITE_QUEEN_CASTLING, ((this->currentState_.castlingRights & kCastlingRights_WhiteQueen) > 0));
    network->set_index(NnueFeatures::NF_BLACK_KING_CASTLING, ((this->currentState_.castlingRights & kCastlingRights_BlackKing) > 0));
    network->set_index(NnueFeatures::NF_BLACK_QUEEN_CASTLING, ((this->currentState_.castlingRights & kCastlingRights_BlackQueen) > 0));
  }
  #endif


  uint64_t hash_;

  // Incremented after a black move.
  uint32_t wholeMoveCounter_;
  Color turn_;

  // PieceMap scores from white's perspective.
  int32_t pieceMapScores[PieceMapType::PieceMapTypeCount];

  std::string history_str() const {
    std::string r = "";
    for (const auto& move : history_) {
      r += move.uci() + " ";
    }
    return r;
  }

  void place_piece_(ColoredPiece cp, Square square);

  void remove_piece_(Square square);

  bool is_material_draw() const {
    const Bitboard everyone = this->colorBitboards_[Color::WHITE] | this->colorBitboards_[Color::BLACK];
    const Bitboard everyoneButKings = everyone & ~(this->pieceBitboards_[ColoredPiece::WHITE_KING] | this->pieceBitboards_[ColoredPiece::BLACK_KING]);
    const bool isThreeManEndgame = std::popcount(everyone) == 3;
    bool isDraw = false;
    isDraw |= (everyoneButKings == 0);
    isDraw |= (everyoneButKings == (this->pieceBitboards_[ColoredPiece::WHITE_KNIGHT] | this->pieceBitboards_[ColoredPiece::BLACK_KNIGHT])) && isThreeManEndgame;
    isDraw |= (everyoneButKings == (this->pieceBitboards_[ColoredPiece::WHITE_BISHOP] | this->pieceBitboards_[ColoredPiece::BLACK_BISHOP])) && isThreeManEndgame;
    return isDraw;
  }

  inline void increment_piece_map(ColoredPiece cp, SafeSquare sq) {
    int32_t const *w = pieceMaps_->weights(cp, sq);
    for (int i = 0; i < PieceMapType::PieceMapTypeCount; ++i) {
      pieceMapScores[i] += w[i];
    }

    #if NNUE_EVAL
    if (cp != ColoredPiece::NO_COLORED_PIECE) {
      this->network->set_piece(cp, sq, 1);
    }
    #endif
  }
  inline void decrement_piece_map(ColoredPiece cp, SafeSquare sq) {
    int32_t const *w = pieceMaps_->weights(cp, sq);
    for (int i = 0; i < PieceMapType::PieceMapTypeCount; ++i) {
      pieceMapScores[i] -= w[i];
    }

    #if NNUE_EVAL
    if (cp != ColoredPiece::NO_COLORED_PIECE) {
      this->network->set_piece(cp, sq, 0);
    }
    #endif
  }

  inline void increment_piece_map(ColoredPiece cp, Square sq) {
    assert(sq < 64);
    increment_piece_map(cp, SafeSquare(sq));
  }
  inline void decrement_piece_map(ColoredPiece cp, Square sq) {
    assert(sq < 64);
    decrement_piece_map(cp, SafeSquare(sq));
  }

  void assert_valid_state() const;
  void assert_valid_state(const std::string& msg) const;

  // A checkmate on exactly the 100th half-move since a pawn move or capture will be considered drawn
  // here, so be careful about calling this in positions where there is a checkmate.
  bool is_draw_assuming_no_checkmate(unsigned plyFromRoot) const;
  bool is_draw_assuming_no_checkmate() const;

  bool is_3fold_repetition(unsigned plyFromRoot) const;
  bool is_fifty_move_rule() const;

  void update_hash_on_state_change(PositionState a, PositionState b);

  // This is used to avoid accidentally making a TT move in an invalid
  // position. It't unclear how necessary it is, since the only time we
  // blindly trust TT's bestMove is when checking for 3-fold draws (see
  // "search.h") and "illegal" moves are probably mostly okay here, since we
  // immediately undo them (e.g. capturing your own pieces or sliding over
  // pieces are fine).
  template<Color TURN>
  bool is_valid_move(Move move) const {
    // TODO: make more robust?
    return tiles_[move.from] != ColoredPiece::NO_COLORED_PIECE && cp2color(tiles_[move.to]) != TURN;
  }

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
  constexpr Bitboard mask = bb(SafeSquare::SA1) | bb(SafeSquare::SA8) | bb(SafeSquare::SH1) | bb(SafeSquare::SH8);
  return ((b & mask) * 0x1040000000000041) >> 60;
}

// Maps (4 -> {0, 1}) and (60 -> {3, 4})
uint8_t king_starts_to_byte(Bitboard b) {
  constexpr Bitboard mask = bb(SafeSquare::SE1) | bb(SafeSquare::SE8);
  constexpr Bitboard magic = bb(SafeSquare(6)) | bb(SafeSquare(7)) | bb(SafeSquare(60)) | bb(SafeSquare(61));
  return (((b & mask) >> 4) * magic) >> 60;
}

constexpr Bitboard kKingStartingPosition = bb(SafeSquare::SE1) | bb(SafeSquare::SE8);

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
  const ColoredPiece capturedPiece = extMove.capture;
  const ColoredPiece promoPiece = move.moveType == MoveType::PROMOTION ? coloredPiece<MOVER_TURN>(Piece(move.promotion + 2)) : movingPiece;
  const Location f = square2location(move.from);
  const Location t = square2location(move.to);
  const Square epSquare = pos->currentState_.epSquare;

  pos->pieceBitboards_[movingPiece] |= f;
  pos->pieceBitboards_[promoPiece] &= ~t;
  pos->colorBitboards_[MOVER_TURN] |= f;
  pos->colorBitboards_[MOVER_TURN] &= ~t;
  pos->tiles_[move.from] = movingPiece;
  pos->tiles_[move.to] = capturedPiece;
  pos->hash_ ^= kZorbristNumbers[movingPiece][move.from];
  pos->hash_ ^= kZorbristNumbers[promoPiece][move.to];

  pos->increment_piece_map(movingPiece, move.from);
  pos->decrement_piece_map(promoPiece, move.to);
  pos->increment_piece_map(capturedPiece, move.to);

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
    SafeSquare rookDestination = SafeSquare((uint16_t(move.from) + uint16_t(move.to)) / 2);
    SafeSquare rookOrigin = SafeSquare(((uint16_t(move.to) % 8) * 7 - 14) / 4 + (MOVER_TURN == Color::WHITE ? 56 : 0));

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

    pos->increment_piece_map(myRookPiece, rookOrigin);
    pos->decrement_piece_map(myRookPiece, rookDestination);
  }

  if (move.to == epSquare && movingPiece == coloredPiece<MOVER_TURN, Piece::PAWN>()) {
    // TODO: get rid of if statement
    if (MOVER_TURN == Color::BLACK) {
      assert(move.from / 8 == 4);
      assert(move.to / 8 == 5);
    } else {
      assert(move.from / 8 == 3);
      assert(move.to / 8 == 2);
    }

    constexpr Color opposingColor = opposite_color<MOVER_TURN>();
    Square enpassantSq = Square((MOVER_TURN == Color::WHITE ? move.to + 8 : move.to - 8));
    Bitboard enpassantLocBB = bb(enpassantSq);

    constexpr ColoredPiece opposingPawn = coloredPiece<opposingColor, Piece::PAWN>();

    pos->pieceBitboards_[opposingPawn] |= enpassantLocBB;
    pos->colorBitboards_[opposingColor] |= enpassantLocBB;
    assert(pos->tiles_[enpassantSq] == ColoredPiece::NO_COLORED_PIECE);
    pos->tiles_[enpassantSq] = opposingPawn;
    pos->hash_ ^= kZorbristNumbers[opposingPawn][enpassantSq];

    // TODO: tell network about en passant square
    pos->increment_piece_map(opposingPawn, enpassantSq);
  }

  #if NNUE_EVAL
  pos->network->set_index(NnueFeatures::NF_IS_WHITE_TURN, (MOVER_TURN == Color::WHITE));

  // TODO: only update if castling rights change.
  pos->network->set_index(NnueFeatures::NF_WHITE_KING_CASTLING, ((pos->currentState_.castlingRights & kCastlingRights_WhiteKing) > 0));
  pos->network->set_index(NnueFeatures::NF_WHITE_QUEEN_CASTLING, ((pos->currentState_.castlingRights & kCastlingRights_WhiteQueen) > 0));
  pos->network->set_index(NnueFeatures::NF_BLACK_KING_CASTLING, ((pos->currentState_.castlingRights & kCastlingRights_BlackKing) > 0));
  pos->network->set_index(NnueFeatures::NF_BLACK_QUEEN_CASTLING, ((pos->currentState_.castlingRights & kCastlingRights_BlackQueen) > 0));
  #endif

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

  pos->history_.push_back(ExtMove(Piece::NO_PIECE, ColoredPiece::NO_COLORED_PIECE, kNullMove));

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
  assert(move.to != Square::NO_SQUARE);
  assert(cp2color(pos->tiles_[move.to]) != TURN);

  pos->hashes_.push_back(pos->hash_);

  const ColoredPiece movingPiece = pos->tiles_[move.from];
  const ColoredPiece promoPiece = move.moveType == MoveType::PROMOTION ? coloredPiece<TURN>(Piece(move.promotion + 2)) : movingPiece;
  constexpr Color opposingColor = opposite_color<TURN>();
  const ColoredPiece capturedPiece = pos->tiles_[move.to];

  const Location f = square2location(move.from);
  const Location t = square2location(move.to);

  pos->states_.push_back(pos->currentState_);
  pos->history_.push_back(ExtMove(cp2p(movingPiece), capturedPiece, move));

  // Remove castling rights if a rook moves or is captured.
  pos->currentState_.castlingRights &= ~four_corners_to_byte(f);
  pos->currentState_.castlingRights &= ~four_corners_to_byte(t);

  // TODO: Set epSquare to NO_SQUARE if there is now way your opponent can play en passant next move.
  //       This will make it easier to count 3-fold draw.
  const Square epSquare = pos->currentState_.epSquare;
  if (TURN == Color::WHITE) {
    bool cond = (movingPiece == coloredPiece<TURN, Piece::PAWN>() && move.from - move.to == 16);
    pos->currentState_.epSquare = Square(cond * (move.to + 8) + (1 - cond) * Square::NO_SQUARE);
  } else {
    bool cond = (movingPiece == coloredPiece<TURN, Piece::PAWN>() && move.to - move.from == 16);
    pos->currentState_.epSquare = Square(cond * (move.to - 8) + (1 - cond) * Square::NO_SQUARE);
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

  pos->increment_piece_map(promoPiece, move.to);
  pos->decrement_piece_map(capturedPiece, move.to);
  pos->decrement_piece_map(movingPiece, move.from);

  if (move.moveType == MoveType::CASTLE) {
    // TODO: get rid of if statement
    if (TURN == Color::BLACK) {
      assert(move.from == 4);
      assert(move.to == 2 || move.to == 6);
    } else {
      assert(move.from == 60);
      assert(move.to == 62 || move.to == 58);
    }
    SafeSquare rookDestination = SafeSquare((uint16_t(move.from) + uint16_t(move.to)) / 2);
    SafeSquare rookOrigin = SafeSquare(((uint16_t(move.to) % 8) * 7 - 14) / 4 + (TURN == Color::WHITE ? 56 : 0));

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

    pos->increment_piece_map(myRookPiece, rookDestination);
    pos->decrement_piece_map(myRookPiece, rookOrigin);
  }

  if (move.to == epSquare && movingPiece == coloredPiece<TURN, Piece::PAWN>()) {
    // TODO: get rid of if statement
    if (TURN == Color::BLACK) {
      assert(move.from / 8 == 4);
      assert(move.to / 8 == 5);
    } else {
      assert(move.from / 8 == 3);
      assert(move.to / 8 == 2);
    }
    Square enpassantSq = Square(TURN == Color::WHITE ? move.to + 8 : move.to - 8);
    Bitboard enpassantLocBB = bb(enpassantSq);

    constexpr ColoredPiece opposingPawn = coloredPiece<opposingColor, Piece::PAWN>();

    assert(pos->tiles_[enpassantSq] == opposingPawn);

    pos->pieceBitboards_[opposingPawn] &= ~enpassantLocBB;
    pos->colorBitboards_[opposingColor] &= ~enpassantLocBB;
    pos->tiles_[enpassantSq] = ColoredPiece::NO_COLORED_PIECE;
    pos->hash_ ^= kZorbristNumbers[opposingPawn][enpassantSq];
    pos->decrement_piece_map(opposingPawn, enpassantSq);
  }

  if (TURN == Color::BLACK) {
    pos->wholeMoveCounter_ += 1;
  }
  ++pos->currentState_.halfMoveCounter;
  pos->currentState_.halfMoveCounter *= (movingPiece != coloredPiece<TURN, Piece::PAWN>() && capturedPiece == ColoredPiece::NO_COLORED_PIECE);
  pos->turn_ = opposingColor;
  pos->hash_ ^= kZorbristTurn;

  pos->update_hash_on_state_change(pos->states_.back(), pos->currentState_);

  #if NNUE_EVAL
  pos->network->set_index(NnueFeatures::NF_IS_WHITE_TURN, (opposingColor == Color::WHITE));

  // TODO: only update if castling rights change.
  pos->network->set_index(NnueFeatures::NF_WHITE_KING_CASTLING, ((pos->currentState_.castlingRights & kCastlingRights_WhiteKing) > 0));
  pos->network->set_index(NnueFeatures::NF_WHITE_QUEEN_CASTLING, ((pos->currentState_.castlingRights & kCastlingRights_WhiteQueen) > 0));
  pos->network->set_index(NnueFeatures::NF_BLACK_KING_CASTLING, ((pos->currentState_.castlingRights & kCastlingRights_BlackKing) > 0));
  pos->network->set_index(NnueFeatures::NF_BLACK_QUEEN_CASTLING, ((pos->currentState_.castlingRights & kCastlingRights_BlackQueen) > 0));
  #endif

  bar<TURN>(pos);

}

void ez_make_move(Position *position, Move move);

void ez_undo(Position *position);

}  // namespace ChessEngine

#endif  // POSITION_H