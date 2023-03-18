#include "Position.h"

#include "piece_maps.h"

#include <iostream>
#include <random>

namespace ChessEngine {

std::ostream& operator<<(std::ostream& stream, const Move move) {
  stream << square_to_string(move.from) << square_to_string(move.to);
  if (move.moveType == MoveType::PROMOTION) {
    stream << piece_to_char(Piece(move.promotion + 2));
  }
  return stream;
}

std::ostream& operator<<(std::ostream& stream, const ExtMove move) {
  return stream << move.str();
}

std::string Move::uci() const {
  if (*this == kNullMove) {
    return "NULL";
  }
  std::string r = "";
  r += square_to_string(from);
  r += square_to_string(to);
  if (moveType == MoveType::PROMOTION) {
    r += piece_to_char(Piece(promotion + 2));
  }
  return r;
}

std::string ExtMove::str() const {
  if (this->move == kNullMove) {
    return "NULL";
  }
  std::string r = "";
  r += char(piece_to_char(this->piece) + 'A' - 'a');
  r += square_to_string(this->move.from);
  if (this->capture != Piece::NO_PIECE) {
    r += "x";
  }
  r += square_to_string(this->move.to);
  if (this->move.moveType == MoveType::PROMOTION) {
    r += piece_to_char(Piece(this->move.promotion + 2));
  }
  return r;
}

std::string ExtMove::uci() const {
  return this->move.uci();
}

std::vector<std::string> split(const std::string& text, char delimiter) {
  std::vector<std::string> r;
  size_t pos = 0;
  size_t i = -1;
  while (++i < text.size()) {
    if (text[i] == delimiter) {
      r.push_back(text.substr(pos, i - pos));
      pos = ++i;
    }
  }
  r.push_back(text.substr(pos, text.size() - pos));
  return r;
}

std::string join(const std::vector<std::string>& A, const std::string& delimiter) {
  std::string r = "";
  for (size_t i = 0; i < A.size(); ++i) {
    r += A[i];
    if (i != A.size() - 1) {
      r += delimiter;
    }
  }
  return r;
}

std::ostream& operator<<(std::ostream& stream, const Position& pos) {
  pos.assert_valid_state();
  for (int y = 0; y < 8; ++y) {
    for (int x = 0; x < 8; ++x) {
      if (pos.tiles_[y*8+x] == ColoredPiece::NO_COLORED_PIECE) {
        stream << ".";
      } else {
        stream << colored_piece_to_char(pos.tiles_[y*8+x]);
      }
    }
    stream << std::endl;
  }
  return stream;
}

uint64_t kZorbristNumbers[ColoredPiece::NUM_COLORED_PIECES][kNumSquares];
uint64_t kZorbristCastling[16];
uint64_t kZorbristEnpassant[8];
uint64_t kZorbristTurn;

void initialize_zorbrist() {
  std::random_device rd;
  std::mt19937_64 e2(rd());
  std::uniform_int_distribution<long long int> dist(uint64_t(0), uint64_t(-1));
  for (ColoredPiece cp = ColoredPiece::NO_COLORED_PIECE; cp < ColoredPiece::NUM_COLORED_PIECES; cp = ColoredPiece(cp + 1)) {
    for (size_t i = 0; i < kNumSquares; ++i) {
      kZorbristNumbers[cp][i] = dist(e2);
    }
  }

  for (size_t i = 0; i < 16; ++i) {
    kZorbristCastling[i] = dist(e2);
  }

  for (size_t i = 0; i < 8; ++i) {
    kZorbristEnpassant[i] = dist(e2);
  }

  kZorbristTurn = dist(e2);
}

Position Position::init() {
  return Position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
}

void Position::_empty_() {
  std::fill_n(pieceBitboards_, ColoredPiece::NUM_COLORED_PIECES, kEmptyBitboard);
  std::fill_n(tiles_, kNumSquares, ColoredPiece::NO_COLORED_PIECE);
  colorBitboards_[Color::WHITE] = kEmptyBitboard;
  colorBitboards_[Color::BLACK] = kEmptyBitboard;
  currentState_.epSquare = Square::NO_SQUARE;
  earlyPieceMapScore_ = 0;
  latePieceMapScore_ = 0;
  hash_ = 0;
}

Position::Position(const std::string& fen) {
  std::vector<std::string> parts = split(fen, ' ');
  if (parts.size() != 6) {
    throw std::runtime_error("Position::Position error 1");
  }

  std::vector<std::string> rows = split(parts[0], '/');
  if (rows.size() != 8) {
    throw std::runtime_error("Position::Position error 2");
  }

  this->_empty_();

  for (size_t y = 0; y < 8; ++y) {
    size_t x = 0;
    for (size_t i = 0; i < rows[y].size(); ++i) {
      char c = rows[y][i];
      if (c >= '1' && c <= '8') {
        x += (c - '0');
        continue;
      }
      this->place_piece_(char_to_colored_piece(c), Square(8 * y + x));
      ++x;
    }
  }

  if (parts[1] != "w" && parts[1] != "b") {
    throw std::runtime_error("Position::Position error 3");
  }
  if (parts[1] == "w") {
    turn_ = Color::WHITE;
    hash_ ^= kZorbristTurn;
  } else {
    turn_ = Color::BLACK;
  }

  {  // Parse castling rights.
    const std::string& castlingPart = parts[2];
    currentState_.castlingRights = 0;
    if (castlingPart.find("K") != std::string::npos) {
      currentState_.castlingRights |= kCastlingRights_WhiteKing;
    }
    if (castlingPart.find("Q") != std::string::npos) {
      currentState_.castlingRights |= kCastlingRights_WhiteQueen;
    }
    if (castlingPart.find("k") != std::string::npos) {
      currentState_.castlingRights |= kCastlingRights_BlackKing;
    }
    if (castlingPart.find("q") != std::string::npos) {
      currentState_.castlingRights |= kCastlingRights_BlackQueen;
    }
  }
  assert(currentState_.castlingRights < 16);
  assert(currentState_.castlingRights >= 0);
  hash_ ^= kZorbristCastling[currentState_.castlingRights];

  currentState_.epSquare = string_to_square(parts[3]);
  hash_ ^= kZorbristEnpassant[currentState_.epSquare % 8 + 1] * (currentState_.epSquare != 0);

  currentState_.halfMoveCounter = std::stoi(parts[4]);

  wholeMoveCounter_ = std::stoi(parts[5]);
}

void Position::place_piece_(ColoredPiece cp, Square square) {
  assert_valid_square(square);
  const Location loc = square2location(square);
  assert(tiles_[square] == ColoredPiece::NO_COLORED_PIECE);
  tiles_[square] = cp;
  pieceBitboards_[cp] |= loc;
  colorBitboards_[cp2color(cp)] |= loc;
  hash_ ^= kZorbristNumbers[cp][square];
  earlyPieceMapScore_ += early_piece_map(cp, square);
  latePieceMapScore_ += late_piece_map(cp, square);
}

void Position::remove_piece_(Square square) {
  assert_valid_square(square);
  const Location loc = square2location(square);

  Location antiloc = ~loc;
  ColoredPiece cp = tiles_[square];

  assert(cp != ColoredPiece::NO_COLORED_PIECE);

  pieceBitboards_[cp] &= antiloc;
  colorBitboards_[cp2color(cp)] &= antiloc;
  hash_ ^= kZorbristNumbers[cp][square];
  earlyPieceMapScore_ -= early_piece_map(cp, square);
  latePieceMapScore_ -= late_piece_map(cp, square);
}

void Position::assert_valid_state() const {
  assert_valid_state("");
}

Position *gDebugPos = nullptr;

void Position::assert_valid_state(const std::string& msg) const {
  #ifndef NDEBUG
  assert_valid_color(turn_);

  // We cannot assume this bc we sometimes make illegal ("pseudo") moves.
  // assert(std::popcount(pieceBitboards_[ColoredPiece::WHITE_KING]) == 1);
  // assert(std::popcount(pieceBitboards_[ColoredPiece::BLACK_KING]) == 1);

  for (size_t i = 0; i < 64; ++i) {
    Color color = cp2color(tiles_[i]);
    if (color != Color::WHITE) {
      if ((colorBitboards_[Color::WHITE] & bb(i)) != 0) {
        gDebugPos = new Position(*this);
        throw std::runtime_error("assert_valid_state a " + std::to_string(i) + "; " + msg);
      }
    } else {
      if ((colorBitboards_[Color::WHITE] & bb(i)) == 0) {
        gDebugPos = new Position(*this);
        throw std::runtime_error("assert_valid_state b " + std::to_string(i) + "; " + msg);
      }
    }
    if (color != Color::BLACK) {
      if ((colorBitboards_[Color::BLACK] & bb(i)) != 0) {
        gDebugPos = new Position(*this);
        throw std::runtime_error("assert_valid_state c " + std::to_string(i) + "; " + msg);
      }
    } else {
      if ((colorBitboards_[Color::BLACK] & bb(i)) == 0) {
        gDebugPos = new Position(*this);
        throw std::runtime_error("assert_valid_state d " + std::to_string(i) + "; " + msg);
      }
    }
    for (ColoredPiece cp = ColoredPiece::WHITE_PAWN; cp <= ColoredPiece::BLACK_KING; cp = ColoredPiece(cp + 1)) {
      if (tiles_[i] != cp) {
        if ((pieceBitboards_[cp] & bb(i)) != 0) {
          gDebugPos = new Position(*this);
          throw std::runtime_error("assert_valid_state e " + std::to_string(i) + "; " + msg);
        }
      } else {
        if ((pieceBitboards_[cp] & bb(i)) == 0) {
          gDebugPos = new Position(*this);
          throw std::runtime_error("assert_valid_state f " + std::to_string(i) + "; " + msg);
        }
      }
    }
  }
  #endif
}

std::string Position::fen() const {
  this->assert_valid_state();

  std::string fen = "";
  for (size_t y = 0; y < 8; ++y) {
    size_t i = 0;
    for (size_t x = 0; x < 8; ++x) {
      ColoredPiece cp = tiles_[y * 8 + x];
      if (cp == ColoredPiece::NO_COLORED_PIECE) {
        ++i;
      } else {
        if (i > 0) {
          fen += std::to_string(i);
          i = 0;
        }
        fen += colored_piece_to_char(cp);
      }
    }
    if (i > 0) {
      fen += std::to_string(i);
    }
    if (y != 7) {
      fen += "/";
    }
  }

  if (turn_ == Color::WHITE) {
    fen += " w ";
  } else {
    fen += " b ";
  }

  if (currentState_.castlingRights & kCastlingRights_WhiteKing) {
    fen += "K";
  }
  if (currentState_.castlingRights & kCastlingRights_WhiteQueen) {
    fen += "Q";
  }
  if (currentState_.castlingRights & kCastlingRights_BlackKing) {
    fen += "k";
  }
  if (currentState_.castlingRights & kCastlingRights_BlackQueen) {
    fen += "q";
  }
  if (currentState_.castlingRights == 0) {
    fen += "-";
  }

  fen += " ";
  fen += square_to_string(currentState_.epSquare);
  fen += " ";
  fen += std::to_string(currentState_.halfMoveCounter);
  fen += " ";
  fen += std::to_string(wholeMoveCounter_);

  return fen;
}

std::string Position::san(Move move) const {
  ColoredPiece cp = tiles_[move.from];
  Piece p = cp2p(cp);
  std::string r;
  r += piece_to_char(p);
  return r + square_to_string(move.to) + "todo";
}

void Position::update_hash_on_state_change(PositionState a, PositionState b) {
  hash_ ^= kZorbristCastling[a.castlingRights];
  hash_ ^= kZorbristCastling[b.castlingRights];
  hash_ ^= kZorbristEnpassant[a.epSquare % 8 + 1] * (a.epSquare != 0);
  hash_ ^= kZorbristEnpassant[b.epSquare % 8 + 1] * (b.epSquare != 0);
}

}  // namespace ChessEngine