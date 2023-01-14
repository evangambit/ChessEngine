#include <cassert>
#include <cstdint>

#include "geometry.h"

namespace ChessEngine {

std::string bstr(Bitboard b) {
  std::string r;
  for (size_t y = 0; y < 8; ++y) {
    for (size_t x = 0; x < 8; ++x) {
      if (b & ((Location(1) << (y * 8 + x)))) {
        r += "x";
      } else {
        r += ".";
      }
    }
    r += "\n";
  }
  return r;
}

std::string bstr(uint8_t b) {
  std::string r = "[";
  for (size_t i = 0; i < 8; ++i) {
    if (b & (1 << i)) {
      r += "x";
    } else {
      r += ".";
    }
  }
  r += "]";
  return r;
}

Bitboard kKingDist[8][64];

void initialize_geometry() {
  for (int dist = 0; dist < 8; ++dist) {
    for (int i = 0; i < 64; ++i) {
      Bitboard r = 0;
      for (int j = 0; j < 64; ++j) {
        int dx = abs(i % 8 - j % 8);
        int dy = abs(i / 8 - j / 8);
        if (dx <= dist && dy <= dist) {
          r |= bb(j);
        }
      }
      kKingDist[dist][i] = r;
    }
  }
}

Location square2location(Square sq) {
  assert(sq < 65);  // sq is valid
  if (sq == 64) return 0;
  return Location(1) << sq;
}

void assert_valid_square(Square sq) {
  assert(sq >= 0 && sq < kNumSquares);
}

void assert_valid_location(Location loc) {
  assert((loc & (loc - 1)) == 0);
}

Square string_to_square(const std::string& string) {
  if (string == "-") {
    return Square::NO_SQUARE;
  }
  if (string.size() != 2) {
    throw std::runtime_error("");
  }
  if (string[0] < 'a' || string[0] > 'h') {
    throw std::runtime_error("");
  }
  if (string[1] < '1' || string[1] > '8') {
    throw std::runtime_error("");
  }
  Square sq = Square((7 - (string[1] - '1')) * 8 + (string[0] - 'a'));
  assert(sq >= 0 && sq < kNumSquares);
  return sq;
}

std::string square_to_string(Square sq) {
  if (sq == Square::NO_SQUARE) {
    return "-";
  }
  assert_valid_square(sq);
  std::string r = "..";
  r[0] = 'a' + (sq % 8);
  r[1] = '8' - (sq / 8);
  return r;
}

Bitboard southFill(Bitboard b) {
   b |= (b <<  8);
   b |= (b << 16);
   b |= (b << 32);
   return b;
}

Bitboard northFill(Bitboard b) {
   b |= (b >>  8);
   b |= (b >> 16);
   b |= (b >> 32);
   return b;
}

}  // namespace ChessEngine
