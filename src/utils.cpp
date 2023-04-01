#include <iostream>
#include <cassert>
#include <cstdint>

#include "utils.h"

namespace ChessEngine {

Color opposite_color(Color color) {
  return Color(3 - color);
}

void assert_valid_color(Color color) {
  assert(color == Color::WHITE || color == Color::BLACK);
}

void assert_valid_colored_piece(ColoredPiece cp) {
  assert(cp > 0 && cp < ColoredPiece::NUM_COLORED_PIECES);
}

ColoredPiece char_to_colored_piece(char c) {
  switch (c) {
    case 'P':
      return ColoredPiece::WHITE_PAWN;
    case 'N':
      return ColoredPiece::WHITE_KNIGHT;
    case 'B':
      return ColoredPiece::WHITE_BISHOP;
    case 'R':
      return ColoredPiece::WHITE_ROOK;
    case 'Q':
      return ColoredPiece::WHITE_QUEEN;
    case 'K':
      return ColoredPiece::WHITE_KING;
    case 'p': 
      return ColoredPiece::BLACK_PAWN;
    case 'n':
      return ColoredPiece::BLACK_KNIGHT;
    case 'b':
      return ColoredPiece::BLACK_BISHOP;
    case 'r':
      return ColoredPiece::BLACK_ROOK;
    case 'q':
      return ColoredPiece::BLACK_QUEEN;
    case 'k':
      return ColoredPiece::BLACK_KING;
  }
  throw std::runtime_error("Unrecognized character " + std::to_string(int32_t(c)));
  return ColoredPiece::NO_COLORED_PIECE;
}

char colored_piece_to_char(ColoredPiece cp) {
  char r = piece_to_char(cp2p(cp));
  if (r == '?') {
    return r;
  }
  if (cp2color(cp) == Color::WHITE) {
    r += 'A' - 'a';
  }
  return r;
}

char piece_to_char(Piece piece) {
  char r = 0;
  r += ('p' * (piece == Piece::PAWN));
  r += ('n' * (piece == Piece::KNIGHT));
  r += ('b' * (piece == Piece::BISHOP));
  r += ('r' * (piece == Piece::ROOK));
  r += ('q' * (piece == Piece::QUEEN));
  r += ('k' * (piece == Piece::KING));
  return r == 0 ? '?' : r;
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

}  // namespace ChessEngine
