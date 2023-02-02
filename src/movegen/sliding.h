#ifndef MOVEGEN_SLIDING_H
#define MOVEGEN_SLIDING_H

#include "../Position.h"
#include "../utils.h"

#include <bit>

namespace ChessEngine {

namespace {

// Code for generating:
// for (int i = 0; i < 8; ++i) {
//   const uint8_t piece = 1 << i;
//   for (int j = 0; j < 256; ++j) {
//     const uint8_t obstacles = j;
//     if (piece & obstacles) {
//       // Invalid state.
//       std::cout << "0,";
//     } else {
//       uint8_t r = 0;
//       for (int k = i + 1; k < 8; ++k) {
//         if (obstacles & (1 << k)) {
//           break;
//         }
//         r |= 1 << k;
//       }
//       for (int k = i - 1; k >= 0; --k) {
//         if (obstacles & (1 << k)) {
//           break;
//         }
//         r |= 1 << k;
//       }
//       std::cout << unsigned(r) << ",";
//     }
//     if (j % 8 == 7) {
//       std::cout << std::endl;
//     } else {
//       std::cout << " ";
//     }
//   }
// }

uint8_t kSlideLookup[2048] = {
  254, 0, 0, 0, 2, 0, 0, 0,
  6, 0, 0, 0, 2, 0, 0, 0,
  14, 0, 0, 0, 2, 0, 0, 0,
  6, 0, 0, 0, 2, 0, 0, 0,
  30, 0, 0, 0, 2, 0, 0, 0,
  6, 0, 0, 0, 2, 0, 0, 0,
  14, 0, 0, 0, 2, 0, 0, 0,
  6, 0, 0, 0, 2, 0, 0, 0,
  62, 0, 0, 0, 2, 0, 0, 0,
  6, 0, 0, 0, 2, 0, 0, 0,
  14, 0, 0, 0, 2, 0, 0, 0,
  6, 0, 0, 0, 2, 0, 0, 0,
  30, 0, 0, 0, 2, 0, 0, 0,
  6, 0, 0, 0, 2, 0, 0, 0,
  14, 0, 0, 0, 2, 0, 0, 0,
  6, 0, 0, 0, 2, 0, 0, 0,
  126, 0, 0, 0, 2, 0, 0, 0,
  6, 0, 0, 0, 2, 0, 0, 0,
  14, 0, 0, 0, 2, 0, 0, 0,
  6, 0, 0, 0, 2, 0, 0, 0,
  30, 0, 0, 0, 2, 0, 0, 0,
  6, 0, 0, 0, 2, 0, 0, 0,
  14, 0, 0, 0, 2, 0, 0, 0,
  6, 0, 0, 0, 2, 0, 0, 0,
  62, 0, 0, 0, 2, 0, 0, 0,
  6, 0, 0, 0, 2, 0, 0, 0,
  14, 0, 0, 0, 2, 0, 0, 0,
  6, 0, 0, 0, 2, 0, 0, 0,
  30, 0, 0, 0, 2, 0, 0, 0,
  6, 0, 0, 0, 2, 0, 0, 0,
  14, 0, 0, 0, 2, 0, 0, 0,
  6, 0, 0, 0, 2, 0, 0, 0,
  253, 252, 0, 0, 1, 0, 0, 0,
  5, 4, 0, 0, 1, 0, 0, 0,
  13, 12, 0, 0, 1, 0, 0, 0,
  5, 4, 0, 0, 1, 0, 0, 0,
  29, 28, 0, 0, 1, 0, 0, 0,
  5, 4, 0, 0, 1, 0, 0, 0,
  13, 12, 0, 0, 1, 0, 0, 0,
  5, 4, 0, 0, 1, 0, 0, 0,
  61, 60, 0, 0, 1, 0, 0, 0,
  5, 4, 0, 0, 1, 0, 0, 0,
  13, 12, 0, 0, 1, 0, 0, 0,
  5, 4, 0, 0, 1, 0, 0, 0,
  29, 28, 0, 0, 1, 0, 0, 0,
  5, 4, 0, 0, 1, 0, 0, 0,
  13, 12, 0, 0, 1, 0, 0, 0,
  5, 4, 0, 0, 1, 0, 0, 0,
  125, 124, 0, 0, 1, 0, 0, 0,
  5, 4, 0, 0, 1, 0, 0, 0,
  13, 12, 0, 0, 1, 0, 0, 0,
  5, 4, 0, 0, 1, 0, 0, 0,
  29, 28, 0, 0, 1, 0, 0, 0,
  5, 4, 0, 0, 1, 0, 0, 0,
  13, 12, 0, 0, 1, 0, 0, 0,
  5, 4, 0, 0, 1, 0, 0, 0,
  61, 60, 0, 0, 1, 0, 0, 0,
  5, 4, 0, 0, 1, 0, 0, 0,
  13, 12, 0, 0, 1, 0, 0, 0,
  5, 4, 0, 0, 1, 0, 0, 0,
  29, 28, 0, 0, 1, 0, 0, 0,
  5, 4, 0, 0, 1, 0, 0, 0,
  13, 12, 0, 0, 1, 0, 0, 0,
  5, 4, 0, 0, 1, 0, 0, 0,
  251, 250, 248, 248, 0, 0, 0, 0,
  3, 2, 0, 0, 0, 0, 0, 0,
  11, 10, 8, 8, 0, 0, 0, 0,
  3, 2, 0, 0, 0, 0, 0, 0,
  27, 26, 24, 24, 0, 0, 0, 0,
  3, 2, 0, 0, 0, 0, 0, 0,
  11, 10, 8, 8, 0, 0, 0, 0,
  3, 2, 0, 0, 0, 0, 0, 0,
  59, 58, 56, 56, 0, 0, 0, 0,
  3, 2, 0, 0, 0, 0, 0, 0,
  11, 10, 8, 8, 0, 0, 0, 0,
  3, 2, 0, 0, 0, 0, 0, 0,
  27, 26, 24, 24, 0, 0, 0, 0,
  3, 2, 0, 0, 0, 0, 0, 0,
  11, 10, 8, 8, 0, 0, 0, 0,
  3, 2, 0, 0, 0, 0, 0, 0,
  123, 122, 120, 120, 0, 0, 0, 0,
  3, 2, 0, 0, 0, 0, 0, 0,
  11, 10, 8, 8, 0, 0, 0, 0,
  3, 2, 0, 0, 0, 0, 0, 0,
  27, 26, 24, 24, 0, 0, 0, 0,
  3, 2, 0, 0, 0, 0, 0, 0,
  11, 10, 8, 8, 0, 0, 0, 0,
  3, 2, 0, 0, 0, 0, 0, 0,
  59, 58, 56, 56, 0, 0, 0, 0,
  3, 2, 0, 0, 0, 0, 0, 0,
  11, 10, 8, 8, 0, 0, 0, 0,
  3, 2, 0, 0, 0, 0, 0, 0,
  27, 26, 24, 24, 0, 0, 0, 0,
  3, 2, 0, 0, 0, 0, 0, 0,
  11, 10, 8, 8, 0, 0, 0, 0,
  3, 2, 0, 0, 0, 0, 0, 0,
  247, 246, 244, 244, 240, 240, 240, 240,
  0, 0, 0, 0, 0, 0, 0, 0,
  7, 6, 4, 4, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  23, 22, 20, 20, 16, 16, 16, 16,
  0, 0, 0, 0, 0, 0, 0, 0,
  7, 6, 4, 4, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  55, 54, 52, 52, 48, 48, 48, 48,
  0, 0, 0, 0, 0, 0, 0, 0,
  7, 6, 4, 4, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  23, 22, 20, 20, 16, 16, 16, 16,
  0, 0, 0, 0, 0, 0, 0, 0,
  7, 6, 4, 4, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  119, 118, 116, 116, 112, 112, 112, 112,
  0, 0, 0, 0, 0, 0, 0, 0,
  7, 6, 4, 4, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  23, 22, 20, 20, 16, 16, 16, 16,
  0, 0, 0, 0, 0, 0, 0, 0,
  7, 6, 4, 4, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  55, 54, 52, 52, 48, 48, 48, 48,
  0, 0, 0, 0, 0, 0, 0, 0,
  7, 6, 4, 4, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  23, 22, 20, 20, 16, 16, 16, 16,
  0, 0, 0, 0, 0, 0, 0, 0,
  7, 6, 4, 4, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  239, 238, 236, 236, 232, 232, 232, 232,
  224, 224, 224, 224, 224, 224, 224, 224,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  15, 14, 12, 12, 8, 8, 8, 8,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  47, 46, 44, 44, 40, 40, 40, 40,
  32, 32, 32, 32, 32, 32, 32, 32,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  15, 14, 12, 12, 8, 8, 8, 8,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  111, 110, 108, 108, 104, 104, 104, 104,
  96, 96, 96, 96, 96, 96, 96, 96,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  15, 14, 12, 12, 8, 8, 8, 8,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  47, 46, 44, 44, 40, 40, 40, 40,
  32, 32, 32, 32, 32, 32, 32, 32,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  15, 14, 12, 12, 8, 8, 8, 8,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  223, 222, 220, 220, 216, 216, 216, 216,
  208, 208, 208, 208, 208, 208, 208, 208,
  192, 192, 192, 192, 192, 192, 192, 192,
  192, 192, 192, 192, 192, 192, 192, 192,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  31, 30, 28, 28, 24, 24, 24, 24,
  16, 16, 16, 16, 16, 16, 16, 16,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  95, 94, 92, 92, 88, 88, 88, 88,
  80, 80, 80, 80, 80, 80, 80, 80,
  64, 64, 64, 64, 64, 64, 64, 64,
  64, 64, 64, 64, 64, 64, 64, 64,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  31, 30, 28, 28, 24, 24, 24, 24,
  16, 16, 16, 16, 16, 16, 16, 16,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  191, 190, 188, 188, 184, 184, 184, 184,
  176, 176, 176, 176, 176, 176, 176, 176,
  160, 160, 160, 160, 160, 160, 160, 160,
  160, 160, 160, 160, 160, 160, 160, 160,
  128, 128, 128, 128, 128, 128, 128, 128,
  128, 128, 128, 128, 128, 128, 128, 128,
  128, 128, 128, 128, 128, 128, 128, 128,
  128, 128, 128, 128, 128, 128, 128, 128,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  63, 62, 60, 60, 56, 56, 56, 56,
  48, 48, 48, 48, 48, 48, 48, 48,
  32, 32, 32, 32, 32, 32, 32, 32,
  32, 32, 32, 32, 32, 32, 32, 32,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  127, 126, 124, 124, 120, 120, 120, 120,
  112, 112, 112, 112, 112, 112, 112, 112,
  96, 96, 96, 96, 96, 96, 96, 96,
  96, 96, 96, 96, 96, 96, 96, 96,
  64, 64, 64, 64, 64, 64, 64, 64,
  64, 64, 64, 64, 64, 64, 64, 64,
  64, 64, 64, 64, 64, 64, 64, 64,
  64, 64, 64, 64, 64, 64, 64, 64,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
};

// 65536*kingLoc+256*occ+pinners
// Given a king, some occupied squares, and some pinners, returns all squares
// between a pinner (inclusive) and the king (exclusive). These are squares
// that it is legal for the pinned piece to move to.
// NOTE: pinners and king *also* count as occupied squares.
uint8_t kPinLookup[8*256*256];

}  // namespace

uint8_t sliding_moves(uint8_t loc, uint8_t occ) {
  assert(std::popcount(loc) == 1);
  return kSlideLookup[256 * lsb(loc) + occ];
}

void initialize_sliding() {
  for (int i = 0; i < 8; ++i) {
    const uint8_t piece = 1 << i;
    for (int j = 0; j < 256; ++j) {
      const uint8_t obstacles = j;
      const int idx = i * 256 + j;
      kSlideLookup[idx] = 0;
      if (piece & obstacles) {
        // Invalid state.
        continue;
      } else {
        uint8_t r = 0;
        for (int k = i + 1; k < 8; ++k) {
          r |= 1 << k;
          if (obstacles & (1 << k)) {
            break;
          }
        }
        for (int k = i - 1; k >= 0; --k) {
          r |= 1 << k;
          if (obstacles & (1 << k)) {
            break;
          }
        }
        kSlideLookup[idx] = r;
      }
    }
  }

  for (unsigned i = 0; i < 8; ++i) {
    for (unsigned occ = 0; occ < 256; ++occ) {
      for (unsigned pinners = 0; pinners < 256; ++pinners) {

        int idx = 65536*i+256*occ+pinners;
        uint8_t king = 1 << i;
        kPinLookup[idx] = 0;

        // Ignore positions with no pinners or no occupied squares (besides the king).
        if (pinners == 0) continue;
        if ((occ & ~(king | pinners)) == 0) continue;

        // Ignore positions where the king or pinners are not occupied squares.
        if ((occ & king) != king) continue;
        if ((occ & pinners) != pinners) continue;

        // This should never happen.
        if (king & pinners) continue;

        uint8_t leftPinners = pinners & (king - 1);
        if (leftPinners) {
          // There is a pinner to our left.
          uint8_t between = (king - 1) & ~(uint8_t(1 << msb(leftPinners)) - 1);
          if (std::popcount(between & occ) == 2) {
            kPinLookup[idx] |= between;
          }
        }

        uint8_t rightPinners = pinners & ~(king * 2 - 1);
        if (rightPinners) {
          // There is a pinner to our right.
          uint8_t between = ((1 << lsb(rightPinners)) * 2 - 1) & ~(king * 2 - 1);
          if (std::popcount(between & occ) == 2) {
            kPinLookup[idx] |= between;
          }
        }

      }
    }
  }
}

inline uint8_t sliding_pinmask(uint8_t loc, uint8_t occ, uint8_t pinners) {
  assert(std::popcount(loc) == 1);
  return kPinLookup[65536*lsb(loc)+256*occ+pinners];
}

}  // namespace ChessEngine

#endif  // MOVEGEN_SLIDING_H
