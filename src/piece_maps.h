#include <cstdint>

const int32_t kPieceMap[12*64] = {
    // WPawn
    -3,  -3,  -3,  -3,  -3,  -3,  -3,  -3,
     5,   6,   5,   4,   5,   3,   2,   1,
     1,   2,   2,   2,   2,   2,   2,   1,
     0,   0,   0,   0,   1,   0,   1,   0,
     0,   0,   0,   0,   0,   0,   0,  -1,
    -1,   0,  -1,  -1,   0,   0,   0,  -1,
     0,   0,   0,  -1,   0,   0,   0,  -1,
    -3,  -3,  -3,  -3,  -3,  -3,  -3,  -3,
    // WKnight
     0,  -1,   1,   1,   3,  -1,   3,  -1,
    -1,   0,   2,   1,   1,   2,   0,   3,
     0,   1,   1,   2,   3,   3,   0,   1,
     0,   1,   1,   2,   1,   1,   1,   0,
    -1,   0,   1,   0,   1,   0,   1,  -1,
    -2,  -1,   0,   0,   0,   0,  -1,  -1,
    -2,  -2,  -1,  -1,  -1,  -1,  -2,  -1,
    -4,  -2,  -2,  -2,  -1,  -2,  -2,  -4,
    // WBishop
     1,  -2,   1,   1,   0,  -2,  -2,   0,
     0,   0,   1,   0,   0,   1,   1,   0,
     0,   1,   0,   2,   2,   0,   1,   1,
     0,   0,   1,   1,   1,   1,   0,   1,
    -1,  -1,   0,   1,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,
    -1,   0,   0,   0,   0,   0,   0,  -1,
    -1,  -1,  -1,  -2,  -1,  -1,  -1,  -2,
    // WRook
     0,   0,   1,   0,   2,   1,   1,   1,
     2,   1,   2,   3,   2,   2,   3,   3,
     1,   1,   2,   1,   2,   3,   3,   0,
     0,   0,   0,   0,   0,   0,  -1,   0,
    -1,  -1,   0,   0,  -1,  -1,  -2,   0,
    -1,  -1,  -1,  -2,  -2,  -1,  -1,  -1,
    -2,  -2,  -1,  -1,  -1,  -1,  -2,  -2,
    -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
    // WQueen
    -1,   2,   1,  -1,   3,   3,   4,   2,
    -1,   0,   1,   1,   2,   3,   1,   2,
    -1,  -1,   1,   1,   2,   3,   3,   1,
    -1,  -1,   0,   0,   2,   1,   0,   1,
    -1,  -1,  -1,   0,   0,  -1,   0,   0,
    -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
    -1,  -1,  -1,  -1,  -1,  -1,  -2,   1,
    -2,  -2,  -2,  -1,  -2,  -3,  -2,  -1,
    // WKing
    -5,   0,   9,  -1,   4,   7,   7, -11,
     1,   8,  -2,   4,  -1,  -1,   3,  -4,
     1,   4,   4,   2,  -3,   5,   2,  -3,
    -4,  -1,  -2,   2,  -1,   0,  -2,   0,
    -2,   1,   0,  -2,  -1,  -1,  -1,  -3,
    -2,   0,  -1,  -1,  -1,  -1,  -1,  -2,
    -1,  -1,   0,  -1,  -1,   0,   0,   0,
    -2,   1,   1,  -2,   0,  -1,   1,   0,
    // BPawn
     2,   2,   2,   2,   2,   2,   2,   2,
     0,   0,   0,   0,   0,   0,  -1,   0,
     1,   0,   1,   0,   0,   0,   0,   0,
     1,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,  -1,  -1,   0,   0,
    -1,  -1,  -2,  -2,  -1,  -2,  -3,  -1,
    -4,  -4,  -4,  -6,  -4,  -2,  -1,  -2,
     2,   2,   2,   2,   2,   2,   2,   2,
    // BKnight
     2,   2,   3,   2,   2,   2,   2,   3,
     3,   2,   1,   1,   1,   1,   2,   2,
     2,   1,   0,   0,   0,   0,   0,   1,
     1,   0,   0,   0,   0,  -1,   0,   0,
     0,   0,  -1,  -1,  -1,  -1,  -1,  -1,
     0,  -1,  -1,  -2,  -2,  -3,  -1,  -1,
     0,  -1,  -1,  -1,  -3,  -3,   0,  -1,
     1,  -1,   1,  -2,  -1,  -1,  -4,  -1,
    // BBishop
     1,   1,   1,   2,   1,   1,   1,   2,
     0,   0,   0,   1,   0,   0,   0,   1,
     0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,  -1,  -1,   0,   0,   0,
     1,   0,  -1,  -2,  -1,  -1,   0,   0,
     0,   0,   0,  -1,  -1,   1,  -1,  -1,
     1,  -1,  -1,  -1,   1,  -1,  -2,   0,
     0,   1,   1,   1,   1,   2,  -2,  -2,
    // BRook
     1,   1,   1,   1,   1,   1,   1,   1,
     2,   2,   1,   2,   2,   1,   1,   2,
     2,   2,   2,   1,   1,   1,   1,   1,
     1,   1,   1,   1,   1,   2,   0,   1,
     0,   1,   0,  -1,  -1,   0,   0,  -1,
    -1,  -1,  -2,  -1,  -1,  -3,  -2,  -2,
    -1,  -1,  -1,  -2,  -3,  -4,  -2,  -2,
    -1,  -2,   0,   0,   1,  -4,  -1,   0,
    // BQueen
     2,   2,   2,   1,   2,   3,   2,   1,
     2,   1,   1,   1,   1,   0,   1,   0,
     1,   1,   1,   1,   1,   1,   0,   2,
     1,   1,   1,   0,   0,   0,  -1,   1,
     1,   1,   0,   0,  -1,  -1,   0,   0,
     1,   1,   0,  -1,  -2,  -3,  -2,  -2,
     1,   1,  -1,   1,  -1,  -3,  -1,  -2,
     0,  -2,  -4,   1,  -5,  -3,  -5,  -3,
    // BKing
     1,  -1,  -1,   2,   0,   1,  -1,   0,
     1,   0,   0,   1,   1,   0,   0,   1,
    -2,   0,   1,   0,   0,   1,   1,   2,
     4,   1,   0,   0,   0,   1,   2,   4,
     1,   0,  -1,   1,   2,   2,  -1,   3,
    -5,   3,  -5,   0,  -3,  -2,  -1,   3,
     8,  -5,  -1,  -2,   6,   0,  -6,   0,
     3,  -9,  -4,   5,   3,  -8,  -5,  -4,
};