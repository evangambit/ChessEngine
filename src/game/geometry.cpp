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

int8_t king_dist(SafeSquare sq1, SafeSquare sq2) {
  assert(sq1 != Square::NO_SQUARE);
  assert(sq2 != Square::NO_SQUARE);
  int8_t a = sq1;
  int8_t b = sq2;
  return std::max(std::abs(a % 8 - b % 8), std::abs(a / 8 - b / 8));
}

Bitboard kKingDist[8][64];
Bitboard kManhattanDist[15][64];
Bitboard kNearby[7][64];
Bitboard kKingHome[64];
Bitboard kSquaresBetween[64][64];
Bitboard kSquareRuleYourTurn[Color::NUM_COLORS][64];
Bitboard kSquareRuleTheirTurn[Color::NUM_COLORS][64];

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

  for (int dist = 0; dist < 15; ++dist) {
    for (int i = 0; i < 64; ++i) {
      Bitboard r = 0;
      for (int j = 0; j < 64; ++j) {
        int dx = abs(i % 8 - j % 8);
        int dy = abs(i / 8 - j / 8);
        if (dx + dy <= dist) {
          r |= bb(j);
        }
      }
      kManhattanDist[dist][i] = r;
    }
  }

  for (int dist = 0; dist < 7; ++dist) {
    for (int i = 0; i < 64; ++i) {
      int kx = i % 8;
      int ky = i / 8;
      kNearby[dist][i] = 0;
      for (int dx = -dist; dx <= dist; ++dx) {
        for (int dy = -dist; dy <= dist; ++dy) {
          if (kx + dx < 0) continue;
          if (kx + dx > 7) continue;
          if (ky + dy < 0) continue;
          if (ky + dy > 7) continue;
          kNearby[dist][i] |= bb((ky + dy) * 8 + (kx + dx));
        }
      }
    }
  }

  for (int a = 0; a < 64; ++a) {
    const int ax = a % 8;
    const int ay = a / 8;
    for (int b = 0; b < 64; ++b) {
      kSquaresBetween[a][b] = bb(a) | bb(b);
      const int bx = b % 8;
      const int by = b / 8;
      if (ax == bx) {
        for (int y = std::min(ay, by) + 1; y < std::max(ay, by); ++y) {
          kSquaresBetween[a][b] |= bb(y * 8 + ax);
        }
      } else if (ay == by) {
        for (int x = std::min(ax, bx) + 1; x < std::max(ax, bx); ++x) {
          kSquaresBetween[a][b] |= bb(ay * 8 + x);
        }
      } else if ((ax - ay) == (bx - by)) {
        // South-east diagonal
        for (int x = std::min(ax, bx) + 1; x < std::max(ax, bx); ++x) {
          kSquaresBetween[a][b] |= bb((ay - ax + x) * 8 + x);
        }
      } else if ((ax + ay) == (bx + by)) {
        // South-west diagonal
        for (int x = std::min(ax, bx) + 1; x < std::max(ax, bx); ++x) {
          kSquaresBetween[a][b] |= bb((ay + ax - x) * 8 + x);
        }
      } else if ((std::abs(ax - bx) == 1 && std::abs(ay - by) == 2) || (std::abs(ax - bx) == 2 && std::abs(ay - by) == 1)) {
        // Knight move

      }
    }
  }

  for (Color color = Color::WHITE; color <= Color::BLACK; color = Color(color + 1)) {
    for (int i = 0; i < 64; ++i) {
      kSquareRuleYourTurn[color][i] = kEmptyBitboard;
      for (int j = 8; j < 56; ++j) {
        const SafeSquare kingSq = SafeSquare(i);
        const SafeSquare pawnSq = SafeSquare(j);
        const SafeSquare promoSq = SafeSquare(color == Color::WHITE ? pawnSq % 8 : pawnSq % 8 + 56);
        if (king_dist(pawnSq, promoSq) < king_dist(kingSq, promoSq)) {
          kSquareRuleTheirTurn[color][i] |= bb(j);
        }
        if (king_dist(pawnSq, promoSq) < king_dist(kingSq, promoSq) - 1) {
          kSquareRuleYourTurn[color][i] |= bb(j);
        }
      }
    }
  }

  std::fill_n(&kKingHome[0], kNumSquares, kEmptyBitboard);
  kKingHome[SafeSquare::SA1] = bb(SafeSquare::SA2) | bb(SafeSquare::SA3)
    | bb(SafeSquare::SB2) | bb(SafeSquare::SB3);
  kKingHome[SafeSquare::SB1] = bb(SafeSquare::SA2) | bb(SafeSquare::SA3)
    | bb(SafeSquare::SB2) | bb(SafeSquare::SB3) | bb(SafeSquare::SC2);
  kKingHome[SafeSquare::SC1] = bb(SafeSquare::SA2) | bb(SafeSquare::SA3)
    | bb(SafeSquare::SB2) | bb(SafeSquare::SB3) | bb(SafeSquare::SC2);
  kKingHome[SafeSquare::SG1] = bb(SafeSquare::SF2) | bb(SafeSquare::SF3)
    | bb(SafeSquare::SG2) | bb(SafeSquare::SG3)
    | bb(SafeSquare::SF2);
  kKingHome[SafeSquare::SH1] = bb(SafeSquare::SG2) | bb(SafeSquare::SG3)
    | bb(SafeSquare::SH2) | bb(SafeSquare::SH3);

  kKingHome[SafeSquare::SA8] = bb(SafeSquare::SA7) | bb(SafeSquare::SA6)
    | bb(SafeSquare::SB7) | bb(SafeSquare::SB6);
  kKingHome[SafeSquare::SB8] = bb(SafeSquare::SA7) | bb(SafeSquare::SA6)
    | bb(SafeSquare::SB7) | bb(SafeSquare::SB6) | bb(SafeSquare::SC7);
  kKingHome[SafeSquare::SC8] = bb(SafeSquare::SA7) | bb(SafeSquare::SA6)
    | bb(SafeSquare::SB7) | bb(SafeSquare::SB6) | bb(SafeSquare::SC7);
  kKingHome[SafeSquare::SG8] = bb(SafeSquare::SF7) | bb(SafeSquare::SF6)
    | bb(SafeSquare::SG7) | bb(SafeSquare::SG6)
    | bb(SafeSquare::SF7);
  kKingHome[SafeSquare::SH8] = bb(SafeSquare::SG7) | bb(SafeSquare::SG6)
    | bb(SafeSquare::SH7) | bb(SafeSquare::SH6);
}

/*
  0  1  2  3  4  5  6  7
  8  9 10 11 12 13 14 15
 16 17 18 19 20 21 22 23
 24 25 26 27 28 29 30 31
 32 33 34 35 36 37 38 39
 40 41 42 43 44 45 46 47
 48 49 50 51 52 53 54 55
 56 57 58 59 60 61 62 63
*/

Location square2location(Square sq) {
  assert(sq < 65);  // sq is valid
  if (sq == 64) return 0;
  return Location(1) << sq;
}

Location square2location(SafeSquare sq) {
  assert(sq < 64);  // sq is valid
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
    throw std::runtime_error("string_to_square error 1");
  }
  if (string[0] < 'a' || string[0] > 'h') {
    throw std::runtime_error("string_to_square error 2");
  }
  if (string[1] < '1' || string[1] > '8') {
    throw std::runtime_error("string_to_square error 3");
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

Bitboard eastFill(Bitboard b) {
  b |= (b & ~kFiles[7]) << 1;
  b |= (b & ~(kFiles[7] | kFiles[6])) << 2;
  b |= (b & ~(kFiles[7] | kFiles[6] | kFiles[5] | kFiles[4])) << 4;
  return b;
}

Bitboard westFill(Bitboard b) {
  b |= (b & ~kFiles[0]) >> 1;
  b |= (b & ~(kFiles[0] | kFiles[1])) >> 2;
  b |= (b & ~(kFiles[0] | kFiles[1] | kFiles[2] | kFiles[3])) >> 4;
  return b;
}


const int8_t kDistToEdge[64] = {
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 1, 1, 1, 1, 1, 1, 0,
  0, 1, 2, 2, 2, 2, 1, 0,
  0, 1, 2, 3, 3, 2, 1, 0,
  0, 1, 2, 3, 3, 2, 1, 0,
  0, 1, 2, 2, 2, 2, 1, 0,
  0, 1, 1, 1, 1, 1, 1, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
};

const int8_t kDistToCorner[64] = {
  0, 1, 2, 3, 3, 2, 1, 0,
  1, 2, 3, 4, 4, 3, 2, 1,
  2, 3, 4, 5, 5, 4, 3, 2,
  3, 4, 5, 6, 6, 5, 4, 3,
  3, 4, 5, 6, 6, 5, 4, 3,
  2, 3, 4, 5, 5, 4, 3, 2,
  1, 2, 3, 4, 4, 3, 2, 1,
  0, 1, 2, 3, 3, 2, 1, 0,
};


}  // namespace ChessEngine
