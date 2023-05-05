// Production:
// g++ src/main.cpp src/game/*.cpp -std=c++20 -O3 -DNDEBUG -o main
// 
// Debug:
// g++ src/main.cpp src/game/*.cpp -std=c++20 -rdynamic -g1
//
// To Generate train.txt
// ./a.out mode printvec fens eval.txt > ./train.txt
//
// To evaluate changes
// ./a.out mode evaluate fens eval.txt depth 2

#include <cassert>
#include <cstdint>
#include <ctime>
#include <cstdlib>

#include <string>
#include <vector>
#include <iostream>
#include <algorithm>
#include <random>
#include <fstream>
#include <deque>

#include "game/geometry.h"
#include "game/utils.h"
#include "game/Position.h"
#include "game/movegen.h"
#include "game/movegen/sliding.h"
#include "game/Evaluator.h"
#include "game/search.h"

using namespace ChessEngine;

int main(int argc, char *argv[]) {
  initialize_geometry();
  initialize_zorbrist();
  initialize_movegen();

  // TODO: load analyzed positions

  return 0;
}
