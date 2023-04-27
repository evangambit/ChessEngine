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

std::string process_with_file_line(const std::string& line) {
  std::string r = "";
  size_t i = 0;
  while (i < line.size() && line[i] == ' ') {
    ++i;
  }
  for (; i < line.size(); ++i) {
    const bool hasNextChar = (i + 1 < line.size());
    if (line[i] == ' ' && hasNextChar && line[i + 1] == ' ') {
      continue;
    }
    if (line[i] == '/' && hasNextChar && line[i + 1] == '/') {
      break;
    }
    r += line[i];
  }
  return r;
}

std::string lpad(int32_t x) {
  std::string r = std::to_string(x);
  while (r.size() < 6) {
    r = " " + r;
  }
  return r;
}

std::string colored_piece_to_string(ColoredPiece cp) {
  switch (cp) {
    case ColoredPiece::WHITE_PAWN:
      return "WHITE PAWN";
    case ColoredPiece::WHITE_KNIGHT:
      return "WHITE KNIGHT";
    case ColoredPiece::WHITE_BISHOP:
      return "WHITE BISHOP";
    case ColoredPiece::WHITE_ROOK:
      return "WHITE ROOK";
    case ColoredPiece::WHITE_QUEEN:
      return "WHITE QUEEN";
    case ColoredPiece::WHITE_KING:
      return "WHITE KING";
    case ColoredPiece::BLACK_PAWN:
      return "BLACK PAWN";
    case ColoredPiece::BLACK_KNIGHT:
      return "BLACK KNIGHT";
    case ColoredPiece::BLACK_BISHOP:
      return "BLACK BISHOP";
    case ColoredPiece::BLACK_ROOK:
      return "BLACK ROOK";
    case ColoredPiece::BLACK_QUEEN:
      return "BLACK QUEEN";
    case ColoredPiece::BLACK_KING:
      return "BLACK KING";
    default:
      return "????";
  }
}

}  // namespace ChessEngine
