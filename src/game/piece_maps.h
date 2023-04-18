#ifndef PIECE_MAPS_h
#define PIECE_MAPS_h

#include "geometry.h"
#include "utils.h"

#include <cstdint>

#include <fstream>

namespace ChessEngine {

struct PieceMaps {
  PieceMaps() {
    std::fill_n(&earlyPieceMap[0], 13 * 64, 0);
    std::fill_n(&latePieceMap[0], 13 * 64, 0);
  }
  int32_t early_piece_map(ColoredPiece cp, Square sq) const;
  int32_t late_piece_map(ColoredPiece cp, Square sq) const;

  void save_weights_to_file(std::ofstream& myfile) {
    myfile << earlyPieceMap[0];
    for (size_t i = 1; i < 13 * 64; ++i) {
      myfile << " " << earlyPieceMap[i];
    }
    myfile << std::endl;

    myfile << latePieceMap[0];
    for (size_t i = 1; i < 13 * 64; ++i) {
      myfile << " " << latePieceMap[i];
    }
    myfile << std::endl;

    myfile.close();
  }

  void load_weights_from_file(std::ifstream &myfile) {
    std::string line;
    std::vector<std::string> params;

    getline(myfile, line);
    params = split(line, ' ');
    if (params.size() != 13 * 64) {
      throw std::runtime_error("");
    }
    for (size_t i = 0; i < 13 * 64; ++i) {
      earlyPieceMap[i] = stoi(params[i]);
    }

    getline(myfile, line);
    params = split(line, ' ');
    if (params.size() != 13 * 64) {
      throw std::runtime_error("");
    }
    for (size_t i = 0; i < 13 * 64; ++i) {
      latePieceMap[i] = stoi(params[i]);
    }

    myfile.close();
  }

 private:
  int32_t earlyPieceMap[13*64];
  int32_t latePieceMap[13*64];
};

const PieceMaps kZeroPieceMap = PieceMaps();

}  // namespace ChessEngine

#endif  // PIECE_MAPS_h
