#ifndef GEOMETRY_H
#define GEOMETRY_H

#include <cassert>
#include <iostream>
#include <cstdint>

#include <string>

#include "utils.h"

namespace ChessEngine {

typedef uint64_t Bitboard;
typedef uint64_t Location;

struct PinMasks {
  Bitboard horizontal;
  Bitboard vertical;
  Bitboard northeast;
  Bitboard northwest;
  Bitboard all;
};

std::string bstr(Bitboard b);

std::string bstr(uint8_t b);

constexpr Bitboard kEmptyBitboard = 0;
constexpr Bitboard kUniverse = ~Bitboard(0);
constexpr Bitboard kFiles[8] = {
  0x0101010101010101ULL,
  0x0202020202020202ULL,
  0x0404040404040404ULL,
  0x0808080808080808ULL,
  0x1010101010101010ULL,
  0x2020202020202020ULL,
  0x4040404040404040ULL,
  0x8080808080808080ULL,
};

constexpr Bitboard kRanks[8] = {
  0x00000000000000ffULL,
  0x000000000000ff00ULL,
  0x0000000000ff0000ULL,
  0x00000000ff000000ULL,
  0x000000ff00000000ULL,
  0x0000ff0000000000ULL,
  0x00ff000000000000ULL,
  0xff00000000000000ULL,
};

constexpr Bitboard kCenter16 = (kFiles[2] | kFiles[3] | kFiles[4] | kFiles[5]) & (kRanks[2] | kRanks[3] | kRanks[4] | kRanks[5]);
constexpr Bitboard kCenter4 = (kFiles[3] | kFiles[4]) & (kRanks[3] | kRanks[4]);

const Bitboard kWhiteSquares = 0xaa55aa55aa55aa55;
const Bitboard kBlackSquares = 0x55aa55aa55aa55aa;
const Bitboard kWhiteSide = 0xffffffff00000000;
const Bitboard kBlackSide = 0x00000000ffffffff;

constexpr Bitboard kOuterRing = kFiles[0] | kFiles[7] | kRanks[0] | kRanks[7];

constexpr int kNumSquares = 64;
enum Square : uint8_t {
  A8, B8, C8, D8, E8, F8, G8, H8,
  A7, B7, C7, D7, E7, F7, G7, H7,
  A6, B6, C6, D6, E6, F6, G6, H6,
  A5, B5, C5, D5, E5, F5, G5, H5,
  A4, B4, C4, D4, E4, F4, G4, H4,
  A3, B3, C3, D3, E3, F3, G3, H3,
  A2, B2, C2, D2, E2, F2, G2, H2,
  A1, B1, C1, D1, E1, F1, G1, H1,
  NO_SQUARE,
};

constexpr Bitboard kMainWhiteDiagonal = 0x8040201008040201;
constexpr Bitboard kMainBlackDiagonal = 0x10204081020408;

enum Direction : int8_t {
  SOUTH =  8,
  WEST  = -1,
  EAST  =  1,
  NORTH = -8,

  NORTH_WEST = -9,
  NORTH_EAST = -7,
  SOUTH_WEST = 7,
  SOUTH_EAST = 9,

  SOUTHx2 =  16,
  WESTx2  = -2,
  EASTx2  =  2,
  NORTHx2 = -16,
};

constexpr Direction opposite_dir(Direction dir) {
  switch (dir) {
    case Direction::SOUTH:
      return Direction::NORTH;
    case Direction::NORTH:
      return Direction::SOUTH;
    case Direction::EAST:
      return Direction::WEST;
    case Direction::WEST:
      return Direction::EAST;
    case Direction::SOUTH_EAST:
      return Direction::NORTH_WEST;
    case Direction::NORTH_WEST:
      return Direction::SOUTH_EAST;
    case Direction::SOUTH_WEST:
      return Direction::NORTH_EAST;
    case Direction::NORTH_EAST:
      return Direction::SOUTH_WEST;
    case Direction::SOUTHx2:
      return Direction::NORTHx2;
    case Direction::NORTHx2:
      return Direction::SOUTHx2;
    case Direction::EASTx2:
      return Direction::WESTx2;
    case Direction::WESTx2:
      return Direction::EASTx2;
  }
}

Location square2location(Square sq);

int8_t king_dist(Square sq1, Square sq2);

constexpr Location bb(unsigned sq) {
  return Location(1) << sq;
}

constexpr Bitboard kRookFiles = kFiles[0] | kFiles[7];

extern Bitboard kKingDist[8][64];
extern Bitboard kNearby[7][64];

// Used to figure out which squares can be moved to to stop a check.
extern Bitboard kSquaresBetween[64][64];

extern Bitboard kSquareRuleYourTurn[Color::NUM_COLORS][64];
extern Bitboard kSquareRuleTheirTurn[Color::NUM_COLORS][64];

extern Bitboard kKingHome[64];

void initialize_geometry();

void assert_valid_square(Square sq);

void assert_valid_location(Location loc);

Square string_to_square(const std::string& string);

std::string square_to_string(Square sq);

template<Direction dir>
Bitboard shift(Bitboard b) {
  b = (dir > 0) ? (b << dir) : (b >> (-dir));
  if ((dir + 16) % 8 == 1) {
    b &= ~kFiles[0];
  }
  if ((dir + 16) % 8 == 2) {
    b &= ~(kFiles[0] | kFiles[1]);
  }
  if ((dir + 16) % 8 == 7) {
    b &= ~kFiles[7];
  }
  if ((dir + 16) % 8 == 6) {
    b &= ~(kFiles[7] | kFiles[6]);
  }
  return b;
}

Bitboard northFill(Bitboard b);

Bitboard southFill(Bitboard b);

extern const int8_t kDistToEdge[64];
extern const int8_t kDistToCorner[64];

constexpr Square operator+(Square s, Direction d) { return Square(int(s) + int(d)); }
constexpr Square operator-(Square s, Direction d) { return Square(int(s) - int(d)); }

inline Square& operator+=(Square& s, Direction d) { return s = s + d; }
inline Square& operator-=(Square& s, Direction d) { return s = s - d; }

inline Square lsb(Bitboard b) {
  assert(b != 0);
  return Square(__builtin_ctzll(b));
}

inline Square msb(Bitboard b) {
  assert(b != 0);
  return Square(63 ^ __builtin_clzll(b));
}

inline Square pop_lsb(Bitboard& b) {
  assert(b != 0);
  Square s = lsb(b);
  b &= b - 1;
  return s;
}

inline Bitboard fatten(Bitboard b) {
  return shift<Direction::WEST>(b) | b | shift<Direction::EAST>(b);
}

}  // namespace ChessEngine

#endif  // GEOMETRY_H