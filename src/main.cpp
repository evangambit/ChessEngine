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

#include <algorithm>
#include <deque>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <stdio.h>
#include <execinfo.h>
#include <signal.h>
#include <stdlib.h>
#include <unistd.h>
#include <signal.h>    
#include <stdio.h>
#include <stdlib.h>
#include <execinfo.h>
#include <unistd.h>

#include "game/geometry.h"
#include "game/utils.h"
#include "game/Position.h"
#include "game/movegen.h"
#include "game/movegen/sliding.h"
#include "game/Evaluator.h"
#include "game/search.h"

using namespace ChessEngine;

Thinker gThinker;

std::string repeat(const std::string text, size_t n) {
  std::string r = "";
  for (size_t i = 0; i < n; ++i) {
    r += text;
  }
  return r;
}

void test_moves() {
  ExtMove moves[kMaxNumMoves];
  ExtMove *end;

  std::ifstream infile("move_test_data.txt");
  std::string line;
  size_t counter = 0;
  while (std::getline(infile, line)) {
    ++counter;
    // std::cout << counter << std::endl;
    // if (counter != 19) {
    //   continue;
    // }
    std::vector<std::string> parts = split(line, ':');
    if (parts.size() != 2) {
      throw std::runtime_error("parts.size() != 2");
    }
    Position pos(parts[0]);
    std::vector<std::string> expected = split(parts[1], ' ');

    const size_t h0 = pos.hash_;

    if (pos.turn_ == Color::WHITE) {
      end = compute_legal_moves<Color::WHITE>(&pos, moves);
    } else {
      end = compute_legal_moves<Color::BLACK>(&pos, moves);
    }

    const size_t h1 = pos.hash_;

    if (h0 != h1) {
      throw std::runtime_error("h0 != h1");
    }

    const size_t n = end - moves;
    std::sort(moves, end, [](ExtMove a, ExtMove b) {
      return a.uci() < b.uci();
    });
    std::sort(expected.begin(), expected.end());

    if (n != expected.size()) {

      std::cout << "counter: " << counter << std::endl;
      std::cout << pos << std::endl;
      std::cout << parts[0] << std::endl;
      for (size_t i = 0; i < std::max(expected.size(), n); ++i) {
        if (i < expected.size()) {
          std::cout << expected[i];
        } else {
          std::cout << "    ";
        }
        std::cout << " : ";
        if (i < n) {
          std::cout << moves[i].uci();
        } else {
          std::cout << "    ";
        }
        if (i < expected.size() && i < n && moves[i].uci() != expected[i]) {
          std::cout << " *";
        }
        std::cout << std::endl;
      }
      throw std::runtime_error("test_moves error");
    }
  }

  std::cout << "tested " << counter << " positions' move generations" << std::endl;
}

void test1() {
  assert(compute_colored_piece(Piece::PAWN, Color::WHITE) == ColoredPiece::WHITE_PAWN);
  assert(compute_colored_piece(Piece::KING, Color::BLACK) == ColoredPiece::BLACK_KING);

  assert(cp2color(ColoredPiece::WHITE_PAWN) == Color::WHITE);
  assert(cp2color(ColoredPiece::WHITE_KING) == Color::WHITE);
  assert(cp2color(ColoredPiece::BLACK_PAWN) == Color::BLACK);
  assert(cp2color(ColoredPiece::BLACK_KING) == Color::BLACK);
  assert(cp2color(ColoredPiece::NO_COLORED_PIECE) == Color::NO_COLOR);

  assert(colored_piece_to_char(ColoredPiece::WHITE_PAWN) == 'P');
  assert(colored_piece_to_char(ColoredPiece::WHITE_KING) == 'K');
  assert(colored_piece_to_char(ColoredPiece::BLACK_PAWN) == 'p');
  assert(colored_piece_to_char(ColoredPiece::BLACK_KING) == 'k');

  assert(std::vector<std::string>({"a", "b"}) == split("a b", ' '));
  assert(std::vector<std::string>({"", "a", "b"}) == split(" a b", ' '));
  assert(std::vector<std::string>({"", "a", "b", ""}) == split(" a b ", ' '));

  for (int i = 0; i < 64; ++i) {
    assert(string_to_square(square_to_string(Square(i))) == i);
  }

  std::random_device rd;
  std::mt19937_64 e2(rd());
  std::uniform_int_distribution<uint64_t> dist(0, uint64_t(-1));
  for (int i = 0; i < 64; ++i) {
    for (int j = 0; j < 64; ++j) {
      Square sq = Square(j);
      Bitboard randboard = dist(e2);
      uint64_t a = std::popcount(randboard & diag::kSouthWestDiagonalMask[sq]);
      uint64_t b = std::popcount(diag::southwest_diag_to_byte(sq, randboard));
      assert(a == b);
    }
  }

  {
    Position pos = Position::init();
    const std::vector<std::string> expected = {
      "Pa2a3",
      "Pb2b3",
      "Pc2c3",
      "Pd2d3",
      "Pe2e3",
      "Pf2f3",
      "Pg2g3",
      "Ph2h3",
      "Pa2a4",
      "Pb2b4",
      "Pc2c4",
      "Pd2d4",
      "Pe2e4",
      "Pf2f4",
      "Pg2g4",
      "Ph2h4",
      "Nb1a3",
      "Nb1c3",
      "Ng1f3",
      "Ng1h3",
    };
    ExtMove moves[kMaxNumMoves];
    ExtMove *end = compute_moves<Color::WHITE, MoveGenType::ALL_MOVES>(pos, moves);
    assert(expected.size() == (end - moves));
    for (size_t i = 0; i < expected.size(); ++i) {
      assert(expected[i] == moves[i].str());
    }
  }
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

void handler(int sig) {
  void *array[40];
  size_t size;

  // get void*'s for all entries on the stack
  size = backtrace(array, 40);

  // print out all the frames to stderr
  fprintf(stderr, "Error: signal %d:\n", sig);
  backtrace_symbols_fd(array, size, STDERR_FILENO);

  #ifndef NDEBUG
  if (gDebugPos != nullptr) {
    std::cout << "<gDebugPos>" << std::endl;
    if (gDebugPos->turn_ == Color::WHITE) {
      std::cout << "turn = white" << std::endl;
    } else {
      std::cout << "turn = black" << std::endl;
    }
    std::cout << "ep = " << unsigned(gDebugPos->currentState_.epSquare) << std::endl;
    for (size_t i = 0; i < gDebugPos->history_.size(); ++i) {
      std::cout << gDebugPos->history_[i].uci() << std::endl;
    }
    std::cout << bstr(gDebugPos->colorBitboards_[Color::BLACK]) << std::endl;
    std::cout << bstr(gDebugPos->colorBitboards_[Color::WHITE]) << std::endl;
    std::cout << "</gDebugPos>" << std::endl;
  }
  #endif

  exit(1);
}

template<Color TURN>
void print_feature_vec(Position *pos, const std::string& originalFen, bool humanReadable, bool makeQuiet, int depth) {
  if (makeQuiet) {
    SearchResult<Color::WHITE> r = to_white(gThinker.qsearch<TURN>(pos, depth, kMinEval, kMaxEval));
    if (r.score > kMaxEval - 100 || r.score < kMinEval + 100) {
      std::cout << "PRINT FEATURE VEC FAIL (MATE)" << std::endl;
      return;
    }
    if (r.move != kNullMove) {
      make_move<TURN>(pos, r.move);
      print_feature_vec<opposite_color<TURN>()>(pos, originalFen, humanReadable, true, depth + 1);
      undo<TURN>(pos);
      return;
    }
  }

  // gThinker.evaluator.features[EF::OUR_PAWNS] = 10;
  Evaluation e = gThinker.evaluator.score<TURN>(*pos);
  // if (gThinker.evaluator.features[EF::OUR_PAWNS] == 10) {
  //   std::cout << "PRINT FEATURE VEC FAIL (SHORT-CIRCUIT)" << std::endl;
  // }
  // if (humanReadable) {
  //   std::cout << "ORIGINAL_FEN " << originalFen << std::endl;
  //   std::cout << "FEN " << pos->fen() << std::endl;
  //   std::cout << "SCORE " << e << std::endl;
  //   const Evaluator& evaluator = gThinker.evaluator;
  //   const int32_t t = evaluator.features[EF::TIME];
  //   for (size_t i = 0; i < EF::NUM_EVAL_FEATURES; ++i) {
  //     const int32_t x = evaluator.features[i];
  //     const int32_t s = (evaluator.earlyW[i] * x * t + evaluator.lateW[i] * x * (16 - t)) / 16 + evaluator.clippedW[i] * x;
  //     std::cout << gThinker.evaluator.features[i] << " " << std::setfill(' ') << std::setw(4) << s << " " << EFSTR[i] << std::endl;
  //   }
  //   std::cout << "bonus" << " " << std::setfill(' ') << std::setw(4) << evaluator.bonus << std::endl;
  // } else {
  //   std::cout << pos->fen() << std::endl;
  //   for (size_t i = 0; i < EF::NUM_EVAL_FEATURES; ++i) {
  //     if (i != 0) {
  //       std::cout << " ";
  //     }
  //     std::cout << gThinker.evaluator.features[i];
  //   }
  //   std::cout << std::endl;
  // }
}

bool is_uci(const std::string& text) {
  if (text.size() < 4) {
    return false;
  }
  if (text.size() > 5) {
    return false;
  }
  if (text[0] < 'a' | text[0] > 'h') {
    return false;
  }
  if (text[1] < '1' | text[1] > '8') {
    return false;
  }
  if (text[2] < 'a' | text[2] > 'h') {
    return false;
  }
  if (text[3] < '1' | text[3] > '8') {
    return false;
  }
  if (text.size() == 4) {
    return true;
  }
  return text[4] == 'n' || text[4] == 'b' || text[4] == 'r' || text[4] == 'q';
}

void mymain(std::vector<Position>& positions, const std::string& mode, double timeLimitMs, Depth depth, uint64_t nodeLimit, bool makeQuiet) {
  if (mode == "printvec" || mode == "printvec-cpu") {
    for (auto pos : positions) {
      if (pos.turn_ == Color::WHITE) {
        print_feature_vec<Color::WHITE>(&pos, pos.fen(), mode == "printvec", makeQuiet, 0);
      } else {
        print_feature_vec<Color::BLACK>(&pos, pos.fen(), mode == "printvec", makeQuiet, 0);
      }
    }
    return;
  } else if (mode == "analyze") {
    for (auto pos : positions) {
      gThinker.reset_stuff();
      SearchResult<Color::WHITE> results(Evaluation(0), kNullMove, false);
      time_t tstart = clock();
      for (size_t i = 1; i <= depth; ++i) {
        SearchResult<Color::WHITE> r = gThinker.search(&pos, i, results);
        if (r.analysisComplete) {
          results = r;
        }
        if (positions.size() == 1) {
          const double secs = double(clock() - tstart) / CLOCKS_PER_SEC;
          std::cout << i << " : " << results.move << " : " << results.score << " (" << secs << " secs, " << gThinker.nodeCounter << " nodes, " << gThinker.nodeCounter / secs / 1000 << " kNodes/sec)" << std::endl;
        }
        if (gThinker.nodeCounter >= nodeLimit) {
          break;
        }
        if (double(clock() - tstart)/CLOCKS_PER_SEC*1000 >= timeLimitMs) {
          break;
        }
      }

      if (positions.size() > 1) {
        std::cout << "FEN: " << pos.fen() << std::endl;
      }

      std::vector<SearchResult<Color::WHITE>> topVariations;
      {
        ExtMove moves[256];
        ExtMove *end;
        if (pos.turn_ == Color::WHITE) {
          end = compute_legal_moves<Color::WHITE>(&pos, moves);
        } else {
          end = compute_legal_moves<Color::BLACK>(&pos, moves);
        }

        std::deque<SearchResult<Color::WHITE>> variations;
        for (ExtMove *move = moves; move < end; ++move) {
          if (pos.turn_ == Color::WHITE) {
            make_move<Color::WHITE>(&pos, move->move);
          } else {
            make_move<Color::BLACK>(&pos, move->move);
          }
          CacheResult cr = gThinker.cache.find(pos.hash_);
          if (isNullCacheResult(cr)) {
            undo<Color::WHITE>(&pos);
            continue;
          }
          if (pos.turn_ == Color::WHITE) {
            // TODO: think about why the multiPVth variation is inaccurate (e.g. if mutliPV = 4, then
            // the 4th variation's score is inaccurate).
            if (move->move == results.move) {
              variations.push_front(SearchResult<Color::WHITE>(cr.eval, move->move));
            } else {
              variations.push_back(SearchResult<Color::WHITE>(cr.eval, move->move));
            }
          } else {
            if (move->move == results.move) {
              variations.push_front(SearchResult<Color::WHITE>(-cr.eval, move->move));
            } else {
              variations.push_back(SearchResult<Color::WHITE>(-cr.eval, move->move));
            }
          }
          if (pos.turn_ == Color::BLACK) {
            undo<Color::WHITE>(&pos);
          } else {
            undo<Color::BLACK>(&pos);
          }
        }
        if (pos.turn_ == Color::WHITE) {
          std::sort(
            variations.begin() + 1,
            variations.end(),
            [](SearchResult<Color::WHITE> a, SearchResult<Color::WHITE> b) -> bool {
              return a.score > b.score;
          });
        } else {
          std::sort(
            variations.begin() + 1,
            variations.end(),
            [](SearchResult<Color::WHITE> a, SearchResult<Color::WHITE> b) -> bool {
              return a.score < b.score;
          });
        }
        for (size_t i = 0; i < variations.size(); ++i) {
          topVariations.push_back(variations[i]);
          if (topVariations.size() >= gThinker.multiPV) {
            break;
          }
        }
      }
      for (size_t i = 0; i < topVariations.size(); ++i) {
        std::cout << "PV " << (i + 1) << ": ";
        gThinker.print_variation(&pos, topVariations[i].move);
      }
    }
  } else if (mode == "play") {
    for (auto pos : positions) {
      while (true) {
        gThinker.reset_stuff();
        SearchResult<Color::WHITE> results(Evaluation(0), kNullMove);
        time_t tstart = clock();
        for (size_t i = 1; i <= depth; ++i) {
          SearchResult<Color::WHITE> r = gThinker.search(&pos, i, results);
          if (r.analysisComplete) {
            results = r;
          }
          if (gThinker.nodeCounter >= nodeLimit) {
            break;
          }
          if (double(clock() - tstart)/CLOCKS_PER_SEC*1000 >= timeLimitMs) {
            break;
          }
        }
        if (results.move == kNullMove) {
          break;
        }
        std::cout << results.move << " " << std::flush;
        if (pos.turn_ == Color::WHITE) {
          make_move<Color::WHITE>(&pos, results.move);
        } else {
          make_move<Color::BLACK>(&pos, results.move);
        }
        if (pos.is_draw() || gThinker.evaluator.is_material_draw(pos)) {
          break;
        }
      }
      std::cout << std::endl;
      std::cout << pos << std::endl;
    }
  }
}

void make_uci_move(Position *pos, std::string uciMove) {
  ExtMove moves[256];
  ExtMove *end;
  if (pos->turn_ == Color::WHITE) {
    end = compute_legal_moves<Color::WHITE>(pos, moves);
  } else {
    end = compute_legal_moves<Color::BLACK>(pos, moves);
  }
  ExtMove *move;
  for (move = moves; move < end; ++move) {
    if (move->move.uci() == uciMove) {
      break;
    }
  }
  if (move->move.uci() != uciMove) {
    throw std::runtime_error("Unrecognized uci move \"" + uciMove + "\"");
    exit(1);
  }
  if (pos->turn_ == Color::WHITE) {
    make_move<Color::WHITE>(pos, move->move);
  } else {
    make_move<Color::BLACK>(pos, move->move);
  }
}

int main(int argc, char *argv[]) {
  signal(SIGSEGV, handler);
  #ifndef NDEBUG
  signal(SIGABRT, handler);
  #endif

  initialize_geometry();
  initialize_zorbrist();
  initialize_movegen();

  std::vector<std::string> args;
  for (size_t i = 1; i < argc; ++i) {
    args.push_back(argv[i]);
  }

  // test1();
  // test_moves();

  Depth depth = 50;
  std::string mode = "analyze";
  uint64_t timeLimitMs = 60000.0;
  std::string fenFile;
  std::vector<std::string> fens;
  uint64_t limitfens = 999999999;
  bool makeQuiet = false;
  std::vector<std::string> uciMoves;
  size_t nodeLimit = -1;

  gThinker.load_weights_from_file("weights.txt");

  while (args.size() > 0) {
    if (args.size() >= 7 && args[0] == "fen") {
      std::vector<std::string> fenVec(args.begin() + 1, args.begin() + 7);
      args = std::vector<std::string>(args.begin() + 7, args.end());
      fens.push_back(join(fenVec, " "));
    } else if (args.size() >= 2 && args[0] == "depth") {
      depth = std::stoi(args[1]);
      if (depth < 0) {
        throw std::invalid_argument("");
        exit(1);
      }
      args = std::vector<std::string>(args.begin() + 2, args.end());
    } else if (args.size() >= 2 && args[0] == "multiPV") {
      gThinker.multiPV = std::stoi(args[1]);
      if (gThinker.multiPV < 0 || gThinker.multiPV > 99) {
        throw std::invalid_argument("");
        exit(1);
      }
      args = std::vector<std::string>(args.begin() + 2, args.end());
    } else if (args.size() >= 2 && args[0] == "time") {
      timeLimitMs = std::stoi(args[1]);
      if (timeLimitMs < 1) {
        throw std::invalid_argument("");
        exit(1);
      }
      args = std::vector<std::string>(args.begin() + 2, args.end());
    } else if (args.size() >= 2 && args[0] == "nodes") {
      nodeLimit = std::stoi(args[1]);
      if (nodeLimit < 1) {
        throw std::invalid_argument("");
        exit(1);
      }
      args = std::vector<std::string>(args.begin() + 2, args.end());
    } else if (args.size() >= 2 && args[0] == "mode") {
      mode = args[1];
      args = std::vector<std::string>(args.begin() + 2, args.end());
    } else if (args.size() >= 2 && args[0] == "fens") {
      fenFile = args[1];
      args = std::vector<std::string>(args.begin() + 2, args.end());
    } else if (args.size() >= 2 && args[0] == "limitfens") {
      limitfens = std::stoi(args[1]);
      args = std::vector<std::string>(args.begin() + 2, args.end());
    } else if (args.size() >= 1 && args[0] == "moves") {
      size_t i = 0;
      while (++i < args.size() && is_uci(args[i])) {
        uciMoves.push_back(args[i]);
      }
      args = std::vector<std::string>(args.begin() + uciMoves.size() + 1, args.end());
    } else if (args.size() >= 2 && args[0] == "makequiet") {
      if (args[1] != "0" && args[1] != "1") {
        std::cout << "makequiet must be \"0\" or \"1\" but is \"" << args[1] << "\"" << std::endl;
        return 1;
      }
      makeQuiet = (args[1] == "1");
      args = std::vector<std::string>(args.begin() + 2, args.end());
    } else if (args.size() >= 2 && args[0] == "loadweights") {
      gThinker.load_weights_from_file(args[1]);
      args = std::vector<std::string>(args.begin() + uciMoves.size() + 2, args.end());
    } else if (args.size() >= 2 && args[0] == "saveweights") {
      gThinker.save_weights_to_file(args[1]);
      return 0;
    } else {
      std::cout << "Cannot understand arguments" << std::endl;
      return 1;
    }
  }

  gThinker.stopThinkingCondition = std::make_unique<OrStopCondition>(
    new StopThinkingNodeCountCondition(nodeLimit),
    new StopThinkingTimeCondition(timeLimitMs)
  );

  if (mode != "evaluate" && mode != "analyze" && mode != "play" && mode != "printvec" && mode != "printvec-cpu" && mode != "print-weights") {
    throw std::runtime_error("Cannot recognize mode \"" + mode + "\"");
  }
  if (mode == "print-weights") {
    for (size_t i = 0; i < EF::NUM_EVAL_FEATURES; ++i) {
      std::cout << EFSTR[i] << " " << gThinker.evaluator.earlyW[i] << " " << gThinker.evaluator.lateW[i] << " " << gThinker.evaluator.clippedW[i] << " " << gThinker.evaluator.lonelyKingW[i] << std::endl;
    }
    return 0;
  }
  if (limitfens < 1) {
    throw std::runtime_error("limitfens cannot be less than 1");
  }
  if (fenFile.size() == 0 && fens.size() == 0) {
    fens.push_back("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
  }
  if (depth < 0) {
    throw std::runtime_error("invalid depth");
  }

  std::vector<Position> positions;

  for (const auto& fen : fens) {
    if (positions.size() >= limitfens) {
      break;
    }
    positions.push_back(Position(fen));
  }

  if (fenFile.size() > 0) {
    std::ifstream infile(fenFile);
    std::string line;
    while (std::getline(infile, line)) {
      if (positions.size() >= limitfens) {
        break;
      }
      positions.push_back(Position(line));
    }
  }

  if (uciMoves.size() != 0 && positions.size() != 1) {
    throw std::runtime_error("cannot provide moves if there is more than one fen");
  }

  if (uciMoves.size() > 0) {
    for (size_t i = 0; i < uciMoves.size(); ++i) {
      make_uci_move(&positions[0], uciMoves[i]);
    }
  }

  mymain(positions, mode, timeLimitMs, depth, nodeLimit, makeQuiet);

  return 0;
}
