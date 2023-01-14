#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <cassert>
#include <cstdint>


namespace ChessEngine {

enum MoveGenType {
  CAPTURES = 1,
  QUIET_MOVES = 2,
  ALL_MOVES = 3,
};

typedef uint8_t CastlingRights;
typedef int16_t Evaluation;

constexpr Evaluation kMinEval = -32767;
constexpr Evaluation kMaxEval = 32767;

// Current record is 218 but we're conservative
// https://chess.stackexchange.com/questions/4490/maximum-possible-movement-in-a-turn
constexpr int kMaxNumMoves = 256;

enum Color {
  NO_COLOR = 0,
  WHITE = 1,
  BLACK = 2,
  NUM_COLORS = 3,
};

template<Color COLOR>
constexpr Color opposite_color();

template<>
constexpr Color opposite_color<Color::WHITE>() {
  return Color::BLACK;
}

template<>
constexpr Color opposite_color<Color::BLACK>() {
  return Color::WHITE;
}

Color opposite_color(Color color);

void assert_valid_color(Color color);

// Important for these values to line up with "four_corners_to_byte"
constexpr CastlingRights kCastlingRights_WhiteKing = 8;
constexpr CastlingRights kCastlingRights_WhiteQueen = 4;
constexpr CastlingRights kCastlingRights_BlackKing = 2;
constexpr CastlingRights kCastlingRights_BlackQueen = 1;

constexpr Evaluation kWhiteWins = 32767;
constexpr Evaluation kBlackWins = -32767;

enum Piece {
  NO_PIECE = 0,
  PAWN = 1,
  KNIGHT = 2,
  BISHOP = 3,
  ROOK = 4,
  QUEEN = 5,
  KING = 6,
};

enum ColoredPiece : int {
  NO_COLORED_PIECE = 0,
  WHITE_PAWN = 1,
  WHITE_KNIGHT = 2,
  WHITE_BISHOP = 3,
  WHITE_ROOK = 4,
  WHITE_QUEEN = 5,
  WHITE_KING = 6,
  BLACK_PAWN = 7,
  BLACK_KNIGHT = 8,
  BLACK_BISHOP = 9,
  BLACK_ROOK = 10,
  BLACK_QUEEN = 11,
  BLACK_KING = 12,
  NUM_COLORED_PIECES = 13,
};

void assert_valid_colored_piece(ColoredPiece cp);

template<Color color, Piece piece>
constexpr ColoredPiece coloredPiece() {
  if (piece == Piece::NO_PIECE) {
    return ColoredPiece::NO_COLORED_PIECE;
  }
  return ColoredPiece((color - 1) * 6 + piece);
}

template<Color color>
constexpr ColoredPiece coloredPiece(Piece piece) {
  if (piece == Piece::NO_PIECE) {
    // TODO: make this faster.
    return ColoredPiece::NO_COLORED_PIECE;
  }
  return ColoredPiece((color - 1) * 6 + piece);
}

constexpr ColoredPiece compute_colored_piece(Piece piece, Color color) {
  assert((color != Color::BLACK) || (piece != Piece::NO_PIECE));
  return ColoredPiece((color - 1) * 6 + piece);
}

constexpr Color cp2color(ColoredPiece cp) {
  return Color((cp + 5) / 6);
}

constexpr Piece cp2p(ColoredPiece cp) {
  return Piece((cp - 1) % 6 + 1);
}

ColoredPiece char_to_colored_piece(char c);

char colored_piece_to_char(ColoredPiece cp);

char piece_to_char(Piece piece);

}  // namespace ChessEngine

#endif  // UTILS_H