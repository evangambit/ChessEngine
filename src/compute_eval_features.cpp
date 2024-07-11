#include "game/geometry.h"
#include "game/utils.h"
#include "game/Position.h"
#include "game/movegen.h"
#include "game/movegen/sliding.h"
#include "game/Evaluator.h"
#include "game/SquareEmbeddings.h"

using namespace ChessEngine;

Evaluator gEvaluator;

int main(int argc, char *argv[]) {
  initialize_geometry();
  initialize_zorbrist();
  initialize_movegen();

  if (argc != 4) {
    std::cerr << "Usage: " << argv[0] << " <input_file> <features_file> <evaluations_file>" << std::endl;
    return 1;
  }

  std::ifstream infile(argv[1]);
  std::ofstream featuresFile(argv[2], std::ios::binary);
  std::ofstream evaluationsFile(argv[3], std::ios::binary);

  if (!infile.is_open()) {
    std::cerr << "Could not open file: " << argv[1] << std::endl;
    return 1;
  }
  if (!featuresFile.is_open()) {
    std::cerr << "Could not open file: " << argv[2] << std::endl;
    return 1;
  }
  if (!evaluationsFile.is_open()) {
    std::cerr << "Could not open file: " << argv[2] << std::endl;
    return 1;
  }

  std::string line;
  size_t count = 0;
  while (std::getline(infile, line)) {
    // fen, wins, draws, losses
    std::vector<std::string> parts = split(line, '|');
    if (line == "") {
      continue;
    }
    if (parts.size() != 4) {
      std::cerr << "Invalid line: " << line << std::endl;
      return 1;
    }
    Position pos(parts[0]);
    if (pos.turn_ == Color::WHITE) {
      gEvaluator.score<Color::WHITE>(pos);
    } else {
      gEvaluator.score<Color::BLACK>(pos);
    }
    // write out int array of features
    featuresFile.write(reinterpret_cast<const char*>(gEvaluator.features), sizeof(gEvaluator.features));

    int16_t eval = std::stoi(parts[1]) + std::stoi(parts[2]) / 2;

    // Features are from the mover's perspecitve, so scores need to be from their perspective too.
    if (pos.turn_ == Color::BLACK) {
      eval = 1000 - eval;
    }
    evaluationsFile.write(reinterpret_cast<const char*>(&eval), sizeof(eval));

    if (++count % 100000 == 0) {
      std::cout << count / 1000 << std::endl;
    }
  }

  return 0;
}
