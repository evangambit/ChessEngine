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

std::string bstr(Bitboard b);

std::string bstr(uint8_t b);

struct PinMasks {
  Bitboard horizontal;
  Bitboard vertical;
  Bitboard northeast;
  Bitboard northwest;
  Bitboard all;
  friend std::ostream& operator<<(std::ostream& out, PinMasks pm) {
    out << bstr(pm.horizontal) << std::endl;
    out << bstr(pm.vertical) << std::endl;
    out << bstr(pm.northeast) << std::endl;
    out << bstr(pm.northwest) << std::endl;
    return out;
  }
  PinMasks& operator|=(const PinMasks& that) {
    this->horizontal |= that.horizontal;
    this->vertical |= that.vertical;
    this->northeast |= that.northeast;
    this->northwest |= that.northwest;
    this->all |= that.all;
    return *this;
  }
};


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

constexpr Bitboard kRanks[9] = {
  0x00000000000000ffULL,
  0x000000000000ff00ULL,
  0x0000000000ff0000ULL,
  0x00000000ff000000ULL,
  0x000000ff00000000ULL,
  0x0000ff0000000000ULL,
  0x00ff000000000000ULL,
  0xff00000000000000ULL,
  0x0000000000000000ULL,  // Maps NO_SQUARE / 8 maps to 0.
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

enum SafeSquare : uint8_t {
  SA8, SB8, SC8, SD8, SE8, SF8, SG8, SH8,
  SA7, SB7, SC7, SD7, SE7, SF7, SG7, SH7,
  SA6, SB6, SC6, SD6, SE6, SF6, SG6, SH6,
  SA5, SB5, SC5, SD5, SE5, SF5, SG5, SH5,
  SA4, SB4, SC4, SD4, SE4, SF4, SG4, SH4,
  SA3, SB3, SC3, SD3, SE3, SF3, SG3, SH3,
  SA2, SB2, SC2, SD2, SE2, SF2, SG2, SH2,
  SA1, SB1, SC1, SD1, SE1, SF1, SG1, SH1,
};

enum UnsafeSquare : uint8_t {
  UA8, UB8, UC8, UD8, UE8, UF8, UG8, UH8,
  UA7, UB7, UC7, UD7, UE7, UF7, UG7, UH7,
  UA6, UB6, UC6, UD6, UE6, UF6, UG6, UH6,
  UA5, UB5, UC5, UD5, UE5, UF5, UG5, UH5,
  UA4, UB4, UC4, UD4, UE4, UF4, UG4, UH4,
  UA3, UB3, UC3, UD3, UE3, UF3, UG3, UH3,
  UA2, UB2, UC2, UD2, UE2, UF2, UG2, UH2,
  UA1, UB1, UC1, UD1, UE1, UF1, UG1, UH1,
  UNO_SQUARE,
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

template<Direction dir>
constexpr Direction opposite_dir() {
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

Location square2location(SafeSquare sq);

int8_t king_dist(SafeSquare sq1, SafeSquare sq2);

constexpr Location bb(SafeSquare sq) {
  return Location(1) << sq;
}

// Deprecated.
constexpr Location bb(unsigned sq) {
  return Location(1) << sq;
}

constexpr Bitboard kRookFiles = kFiles[0] | kFiles[7];

extern Bitboard kKingDist[8][64];
extern Bitboard kManhattanDist[15][64];
extern Bitboard kNearby[7][64];

// Used to figure out which squares can be moved to to stop a check.
extern Bitboard kSquaresBetween[64][64];

extern Bitboard kSquareRuleYourTurn[Color::NUM_COLORS][64];
extern Bitboard kSquareRuleTheirTurn[Color::NUM_COLORS][64];

extern Bitboard kKingHome[64];

void initialize_geometry();

void assert_valid_square(unsigned sq);

void assert_valid_location(Location loc);

UnsafeSquare string_to_square(const std::string& string);

std::string square_to_string(UnsafeSquare sq);

std::string square_to_string(SafeSquare sq);

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

Bitboard eastFill(Bitboard b);

Bitboard westFill(Bitboard b);

inline uint8_t eastmost_file_to_byte(Bitboard board) {
  constexpr Bitboard magic = bb(SafeSquare(49)) | bb(SafeSquare(42)) | bb(SafeSquare(35)) | bb(SafeSquare(28)) | bb(SafeSquare(21)) | bb(SafeSquare(14)) | bb(SafeSquare(7)) | bb(SafeSquare(0));
  return (board * magic) >> 56;
}

extern const int8_t kDistToEdge[64];
extern const int8_t kDistToCorner[64];

// Strictly speaking, this is not safe. We rely on the caller to ensure that the result is a valid square.
constexpr SafeSquare operator+(SafeSquare s, Direction d) { return SafeSquare(int(s) + int(d)); }
constexpr SafeSquare operator-(SafeSquare s, Direction d) { return SafeSquare(int(s) - int(d)); }

inline UnsafeSquare lsb(Bitboard b) {
  assert(b != 0);
  return UnsafeSquare(__builtin_ctzll(b));
}

inline SafeSquare lsb_i_promise_board_is_not_empty(Bitboard b) {
  assert(b != 0);
  return SafeSquare(__builtin_ctzll(b));
}

inline SafeSquare safe_msb(Bitboard b) {
  assert(b != 0);
  return SafeSquare(63 ^ __builtin_clzll(b));
}

inline UnsafeSquare lsb_or_none(Bitboard b) {
  return select<UnsafeSquare>(b != 0, UnsafeSquare(__builtin_ctzll(b)), UnsafeSquare::UNO_SQUARE);
}

inline SafeSquare lsb_or(Bitboard b, SafeSquare defaultValue) {
  return select<SafeSquare>(b != 0, SafeSquare(__builtin_ctzll(b)), defaultValue);
}

inline Square msb(Bitboard b) {
  assert(b != 0);
  return Square(63 ^ __builtin_clzll(b));
}

inline UnsafeSquare msb_or_none(Bitboard b) {
  return select<UnsafeSquare>(b != 0, UnsafeSquare(63 ^ __builtin_clzll(b)), UnsafeSquare::UNO_SQUARE);
}

inline UnsafeSquare msb_or(Bitboard b, UnsafeSquare defaultValue) {
  return select<UnsafeSquare>(b != 0, UnsafeSquare(63 ^ __builtin_clzll(b)), defaultValue);
}

inline UnsafeSquare pop_lsb(Bitboard& b) {
  assert(b != 0);
  UnsafeSquare s = lsb(b);
  b &= b - 1;
  return s;
}

inline Bitboard fatten(Bitboard b) {
  return shift<Direction::WEST>(b) | b | shift<Direction::EAST>(b);
}

}  // namespace ChessEngine

#endif  // GEOMETRY_H