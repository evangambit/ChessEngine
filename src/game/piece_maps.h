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
    for (size_t i = 0; i < ColoredPiece::NUM_COLORED_PIECES; ++i) {
      myfile << "// " << colored_piece_to_string(ColoredPiece(i)) << std::endl;
      for (size_t j = 0; j < 64; ++j) {
        myfile << lpad(earlyPieceMap[i * 64 + j]);
        if (j % 8 == 7) {
          myfile << std::endl;
        }
      }
    }

    for (size_t i = 0; i < ColoredPiece::NUM_COLORED_PIECES; ++i) {
      myfile << "// " << colored_piece_to_string(ColoredPiece(i)) << std::endl;
      for (size_t j = 0; j < 64; ++j) {
        myfile << lpad(latePieceMap[i * 64 + j]);
        if (j % 8 == 7) {
          myfile << std::endl;
        }
      }
    }
  }

  void load_weights_from_file(std::ifstream &myfile) {
    std::string line;
    std::vector<std::string> params;

    for (size_t i = 0; i < ColoredPiece::NUM_COLORED_PIECES; ++i) {
      getline(myfile, line);
      if (line.substr(0, 3) != "// ") {
        throw std::runtime_error("Unexpected weight format; expected \"// \" but got \"" + line.substr(0, 3) + "\"");
      }
      for (size_t y = 0; y < 8; ++y) {
        getline(myfile, line);
        line = process_with_file_line(line);
        std::vector<std::string> parts = split(line, ' ');
        if (parts.size() != 8) {
          throw std::runtime_error("Expected 8 weights in piece-map row but got " + std::to_string(parts.size()));
        }
        for (size_t x = 0; x < 8; ++x) {
          earlyPieceMap[i * 64 + y * 8 + x] = stoi(parts[x]);
        }
      }
    }

    for (size_t i = 0; i < ColoredPiece::NUM_COLORED_PIECES; ++i) {
      getline(myfile, line);
      if (line.substr(0, 3) != "// ") {
        throw std::runtime_error("Unexpected weight format; expected \"// \" but got \"" + line.substr(0, 3) + "\"");
      }
      for (size_t y = 0; y < 8; ++y) {
        getline(myfile, line);
        line = process_with_file_line(line);
        std::vector<std::string> parts = split(line, ' ');
        if (parts.size() != 8) {
          throw std::runtime_error("Expected 8 weights in piece-map row but got " + std::to_string(parts.size()));
        }
        for (size_t x = 0; x < 8; ++x) {
          latePieceMap[i * 64 + y * 8 + x] = stoi(parts[x]);
        }
      }
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
