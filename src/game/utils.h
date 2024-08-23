#ifndef UTILS_H
#define UTILS_H

#include <cassert>
#include <cstdint>

#include <deque>
#include <iostream>
#include <vector>
#include <unordered_set>

namespace ChessEngine {

enum MoveGenType {
  CAPTURES,
  CHECKS_AND_CAPTURES,
  ALL_MOVES,
};

typedef uint8_t CastlingRights;
typedef int16_t Evaluation;

constexpr Evaluation kMinEval = -32767;
constexpr Evaluation kMaxEval = 32767;

constexpr Evaluation kMissingKing = kMinEval + 1;  // -32766

constexpr Evaluation kCheckmate = kMissingKing + 1;  // -32765
constexpr Evaluation kLongestForcedMate = kCheckmate + 100;    // -32665

constexpr Evaluation kQMissingKing = kLongestForcedMate + 1;  // -32664
constexpr Evaluation kQCheckmate = kQMissingKing + 1;  // -32663
constexpr Evaluation kQLongestForcedMate = kQCheckmate + 100;  // -32563

Evaluation child_eval_to_parent_eval(Evaluation eval);
Evaluation parent_eval_to_child_eval(Evaluation eval);

std::string eval2str(Evaluation eval);

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

// Branchless "condition ? a : b"
template<class T>
inline T select(bool condition, T a, T b) {
  assert(condition == 0 || condition == 1);
  T tmp = T(condition - 1);
  return T((a & ~tmp) | (b & tmp));
}

// Branchless "condition ? value : 0"
template<class T>
inline T value_or_zero(bool condition, T value) {
  assert(condition == 0 || condition == 1);
  return (value & ~(condition - 1));
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
  NUM_PIECES = 7,
};

enum ColoredPiece : unsigned {
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
  return ColoredPiece(((color - 1) * 6 + piece) * (piece != Piece::NO_PIECE));
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

std::vector<std::string> split(const std::string& text, char delimiter);

template<class T>
std::ostream& operator<<(std::ostream& stream, const std::vector<T>& vec) {
  if (vec.size() == 0) {
    return stream << "[]";
  }
  stream << "[" << vec[0];
  for (size_t i = 1; i < vec.size(); ++i) {
    stream << ", " << vec[i];
  }
  return stream << "]";
}

template<class T>
std::ostream& operator<<(std::ostream& stream, const std::unordered_set<T>& set) {
  if (set.size() == 0) {
    return stream << "{}";
  }
  stream << "{";
  bool isFirst = true;
  for (const auto& val : set) {
    if (!isFirst) {
      std::cout << ", ";
    }
    stream << val;
    isFirst = true;
  }
  return stream << "}";
}

template<class T>
std::ostream& operator<<(std::ostream& stream, const std::deque<T>& vec) {
  if (vec.size() == 0) {
    return stream << "{}";
  }
  stream << "{" << vec[0];
  for (size_t i = 1; i < vec.size(); ++i) {
    stream << ", " << vec[i];
  }
  return stream << "}";
}


template<class A, class B>
std::ostream& operator<<(std::ostream& stream, const std::pair<A, B>& pair) {
  return stream << "(" << pair.first << ", " << pair.second << ")";
}

template<class A>
std::ostream& operator<<(std::ostream& stream, const std::pair<A, Evaluation>& pair) {
  return stream << "(" << pair.first << ", " << eval2str(pair.second) << ")";
}

std::string process_with_file_line(const std::string& line);

std::string lpad(int32_t x);

std::string piece_to_string(Piece piece);

std::string colored_piece_to_string(ColoredPiece cp);

}  // namespace ChessEngine

#endif  // UTILS_H