// g++ src/uci.cpp src/game/*.cpp -std=c++20 -O3 -DNDEBUG -o main
// g++ src/uci.cpp src/game/*.cpp -std=c++20 -O3 -DNDEBUG -DSquareControl -o sc-old

#import "game/search.h"
#import "game/Position.h"
#import "game/movegen.h"
#import "game/utils.h"
#import "game/string_utils.h"

#include <deque>
#include <iostream>

using namespace ChessEngine;

struct UciEngine {
  Thinker thinker;
  Position pos;
  std::shared_ptr<StopThinkingSwitch> stopThinkingSwitch;
  std::thread *thinkingThread;
  UciEngine() : stopThinkingSwitch(nullptr), thinkingThread(nullptr) {
    pos = Position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    #ifndef SquareControl
    this->thinker.load_weights_from_file("weights.txt");
    #else
    this->thinker.load_weights_from_file("weights-square-control.txt");
    #endif
  }
  void start(std::istream& cin) {
    while (true) {
      if (std::cin.eof()) {
        break;
      }
      std::string line;
      getline(std::cin, line);
      if (line == "quit") {
        exit(0);
        break;
      }

      // Skip empty lines.
      if(line.find_first_not_of(' ') == std::string::npos) {
        continue;
      }

      this->handle_uci_command(&line);
    }
  }
  void handle_uci_command(std::string *command) {
    remove_excess_whitespace(command);
    std::vector<std::string> parts = split(*command, ' ');
    if (parts[0] == "position") {
      handle_position(parts);
    } else if (parts[0] == "go") {
      handle_go(parts);
    } else if (parts[0] == "setoption") {
      handle_set_option(parts);
    } else if (parts[0] == "ucinewgame") {
      this->thinker.reset_stuff();
    } else if (parts[0] == "stop") {
      if (this->stopThinkingSwitch != nullptr) {
        this->stopThinkingSwitch->stop();
      }
    } else if (parts[0] == "loadweights") {  // Custom commands below this line.
      this->thinker.load_weights_from_file(parts[1]);
    } else if (parts[0] == "play") {
      handle_play(parts);
    } else if (parts[0] == "move") {
      handle_move(parts);
    } else if (parts[0] == "eval") {
      Evaluator& evaulator = this->thinker.evaluator;
      if (this->pos.turn_ == Color::WHITE) {
        std::cout << evaulator.score<Color::WHITE>(this->pos) << std::endl;
      } else {
        std::cout << -evaulator.score<Color::BLACK>(this->pos) << std::endl;
      }
      for (int i = 0; i < EF::NUM_EVAL_FEATURES; ++i) {
        if (evaulator.features[i] != 0) {
          std::cout << evaulator.features[i] << " " << EFSTR[i] << std::endl;
        }
      }
    } else if (parts[0] == "probe") {
      Position query = Position::init();
      size_t i = 1;
      if (parts[1] == "fen") {
        std::vector<std::string> fen(parts.begin() + 2, parts.begin() + 8);
        query = Position(join(fen, " "));
        i = 8;
      }
      if (parts[i] == "moves") {
        while (++i < parts.size()) {
          if (!make_uci_move(&query, parts[i])) {
            std::cout << "Invalid uci move " << repr(parts[i]) << std::endl;
            return;
          }
        }
      }

      std::pair<CacheResult, std::vector<Move>> variation = this->thinker.get_variation(&query, kNullMove);

      if (isNullCacheResult(variation.first)) {
        std::cout << "Cache result for " << query.fen() << " is missing" << std::endl;
      }
      std::cout << variation.first.eval;
      for (const auto& move : variation.second) {
        std::cout << " " << move;
      }
      std::cout << std::endl;
    } else {
      std::cout << "Unrecognized command " << repr(*command) << std::endl;
    }
  }

  void invalid(const std::string& command) {
    std::cout << "Invalid use of " << repr(command) << " command" << std::endl;
  }

  void handle_position(const std::vector<std::string>& command) {
    if (command.size() < 2) {
      invalid(join(command, " "));
      return;
    }
    size_t i;
    if (command[1] == "startpos") {
      this->pos = Position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
      i = 2;
    } else if (command.size() >= 8 && command[1] == "fen") {
      std::vector<std::string> fen(command.begin() + 2, command.begin() + 8);
      i = 8;
      this->pos = Position(join(fen, " "));
    } else {
      invalid(join(command, " "));
      return;
    }
    if (i == command.size()) {
      std::cout << "Position set to \"" << this->pos.fen() << std::endl;
      return;
    }
    if (command[i] != "moves") {
      invalid(join(command, " "));
      return;
    }
    while (++i < command.size()) {
      std::string uciMove = command[i];
      ExtMove moves[kMaxNumMoves];
      ExtMove *end;
      if (this->pos.turn_ == Color::WHITE) {
        end = compute_legal_moves<Color::WHITE>(&this->pos, moves);
      } else {
        end = compute_legal_moves<Color::BLACK>(&this->pos, moves);
      }
      bool foundMove = false;
      for (ExtMove *move = moves; move < end; ++move) {
        if (move->move.uci() == uciMove) {
          foundMove = true;
          if (this->pos.turn_ == Color::WHITE) {
            make_move<Color::WHITE>(&this->pos, move->move);
          } else {
            make_move<Color::BLACK>(&this->pos, move->move);
          }
          break;
        }
      }
      if (!foundMove) {
        std::cout << "Could not find move " << repr(uciMove) << std::endl;
        return;
      }
    }
  }

  bool make_uci_move(Position* position, const std::string& uciMove) {
    ExtMove moves[kMaxNumMoves];
    ExtMove *end;
    if (position->turn_ == Color::WHITE) {
      end = compute_legal_moves<Color::WHITE>(position, moves);
    } else {
      end = compute_legal_moves<Color::BLACK>(position, moves);
    }
    for (ExtMove *move = moves; move < end; ++move) {
      if (move->move.uci() == uciMove) {
        if (position->turn_ == Color::WHITE) {
          make_move<Color::WHITE>(position, move->move);
        } else {
          make_move<Color::BLACK>(position, move->move);
        }
        return true;
      }
    }
    return false;
  }

  void handle_go(const std::vector<std::string>& command) {
    size_t nodeLimit = size_t(-1);
    uint64_t depthLimit = 99;
    uint64_t timeLimitMs = 1000 * 60 * 60;

    if (command.size() != 3) {
      invalid(join(command, " "));
      return;
    }

    if (command.at(1) == "depth") {
      depthLimit = stoi(command.at(2));
    } else if (command.at(1) == "nodes") {
      nodeLimit = stoi(command.at(2));
    } else if (command.at(1) == "time") {
      timeLimitMs = stoi(command.at(2));
    } else if (command[1] == "time") {
      timeLimitMs = stoi(command[2]);
    } else {
      invalid(join(command, " "));
      return;
    }

    this->thinker.stopThinkingCondition = std::make_unique<OrStopCondition>(
      std::make_shared<StopThinkingNodeCountCondition>(nodeLimit),
      std::make_shared<StopThinkingTimeCondition>(timeLimitMs)
    );

    // TODO: get rid of this (selfplay2 sometimes crashes when we try to get rid of it now).
    this->thinker.reset_stuff();

    SearchResult<Color::WHITE> result = search(&this->thinker, &this->pos, depthLimit, [this](Position *position, SearchResult<Color::WHITE> results, size_t depth, double secs) {
      this->_print_variations(position, depth, secs, this->thinker.multiPV);
    });

    if (this->pos.turn_ == Color::WHITE) {
      make_move<Color::WHITE>(&this->pos, result.move);
    } else {
      make_move<Color::BLACK>(&this->pos, result.move);
    }
    CacheResult cr = this->thinker.cache.find<false>(this->pos.hash_);
    if (this->pos.turn_ == Color::WHITE) {
      undo<Color::BLACK>(&this->pos);
    } else {
      undo<Color::WHITE>(&this->pos);
    }

    std::cout << "bestmove " << result.move;
    if (!isNullCacheResult(cr)) {
      std::cout << " ponder " << cr.bestMove;
    }
    std::cout << std::endl;
  }

  void handle_move(const std::vector<std::string>& command) {
    size_t i = 0;
    while (++i < command.size()) {
      std::string uciMove = command[i];
      ExtMove moves[kMaxNumMoves];
      ExtMove *end;
      if (this->pos.turn_ == Color::WHITE) {
        end = compute_legal_moves<Color::WHITE>(&this->pos, moves);
      } else {
        end = compute_legal_moves<Color::BLACK>(&this->pos, moves);
      }
      bool foundMove = false;
      for (ExtMove *move = moves; move < end; ++move) {
        if (move->move.uci() == uciMove) {
          foundMove = true;
          if (this->pos.turn_ == Color::WHITE) {
            make_move<Color::WHITE>(&this->pos, move->move);
          } else {
            make_move<Color::BLACK>(&this->pos, move->move);
          }
          break;
        }
      }
      if (!foundMove) {
        std::cout << "Could not find move " << repr(uciMove) << std::endl;
        return;
      }
    }
  }

  void handle_play(const std::vector<std::string>& command) {
    size_t nodeLimit = size_t(-1);
    uint64_t depthLimit = 99;
    uint64_t timeLimitMs = 1000 * 60 * 60;

    if (command.size() != 3) {
      invalid(join(command, " "));
      return;
    }

    if (command[1] == "depth") {
      depthLimit = stoi(command[2]);
    } else if (command[1] == "nodes") {
      nodeLimit = stoi(command[2]);
    } else if (command[1] == "time") {
      timeLimitMs = stoi(command[2]);
    } else {
      invalid(join(command, " "));
      return;
    }

    Position pos(this->pos);

    while (true) {
      this->thinker.stopThinkingCondition = std::make_unique<OrStopCondition>(
        std::make_shared<StopThinkingNodeCountCondition>(nodeLimit),
        std::make_shared<StopThinkingTimeCondition>(timeLimitMs)
      );

      // TODO: get rid of this (selfplay2 sometimes crashes when we try to get rid of it now).
      this->thinker.reset_stuff();

      SearchResult<Color::WHITE> result = search(&this->thinker, &pos, depthLimit);
      if (result.move == kNullMove) {
        break;
      }
      std::cout << " " << result.move << std::flush;
      if (pos.turn_ == Color::WHITE) {
        make_move<Color::WHITE>(&pos, result.move);
        if (is_checkmate<Color::BLACK>(&pos)) {
          break;
        }
      } else {
        make_move<Color::BLACK>(&pos, result.move);
        if (is_checkmate<Color::WHITE>(&pos)) {
          break;
        }
      }
    }
    std::cout << std::endl;
  }

  void _print_variations(Position* position, int depth, double secs, size_t multiPV) const {
    const uint64_t timeMs = secs * 1000;
    std::vector<SearchResult<Color::WHITE>> variations = this->thinker.variations;
    if (variations.size() == 0) {
      throw std::runtime_error("No variations found!");
    }
    for (size_t i = 0; i < std::min(multiPV, variations.size()); ++i) {
      std::pair<CacheResult, std::vector<Move>> variation = this->thinker.get_variation(position, variations[i].move);
      std::cout << "info depth " << depth;
      std::cout << " multipv " << (i + 1);
      std::cout << " score cp " << variation.first.eval;
      std::cout << " nodes " << this->thinker.nodeCounter;
      std::cout << " nps " << uint64_t(double(this->thinker.nodeCounter) / secs);
      std::cout << " time " << timeMs;
      std::cout << " pv";
      for (const auto& move : variation.second) {
        std::cout << " " << move.uci();
      }
      std::cout << std::endl;
    }
  }

  void handle_set_option(const std::vector<std::string>& command) {
    if (command.size() != 5 && command.size() != 3) {
      invalid(join(command, " "));
      return;
    }
    if (command[1] != "name") {
      invalid(join(command, " "));
      return;
    }
    const std::string name = command[2];
    if (command.size() == 3) {
      if (name == "clear-tt") {
        this->thinker.reset_stuff();
        return;
      }
      std::cout << "Unrecognized option " << repr(name) << std::endl;
    } else {
      if (command[3] != "value") {
        invalid(join(command, " "));
        return;
      }
      const std::string value = command[4];
      if (name == "MultiPV") {
        int multiPV;
        try {
          multiPV = stoi(value);
          if (multiPV <= 0) {
            throw std::invalid_argument("Value must be at least 1");
          }
        } catch (std::invalid_argument&) {
          std::cout << "Value must be an integer" << std::endl;
          return;
        }
        if (multiPV < 1) {
          std::cout << "Value must be positive" << std::endl;
          return;
        }
        this->thinker.multiPV = multiPV;
        return;
      } else if (name == "Threads") {
        int numThreads;
        try {
          numThreads = stoi(value);
          if (numThreads <= 0) {
            throw std::invalid_argument("Value must be at least 1");
          }
        } catch (std::invalid_argument&) {
          std::cout << "Value must be an integer" << std::endl;
          return;
        }
        if (numThreads < 1) {
          std::cout << "Value must be positive" << std::endl;
          return;
        }
        this->thinker.numThreads = numThreads;
        return;
      }
      std::cout << "Unrecognized option " << repr(name) << std::endl;
    }
  }
};

int main(int argc, char *argv[]) {
  std::cout << "Chess Engine" << std::endl;

  // Wait for "uci" command.
  while (true) {
    std::string line;
    getline(std::cin, line);
    if (line == "uci") {
      break;
    } else {
      std::cout << "Unrecognized command " << repr(line) << std::endl;
    }
  }

  initialize_geometry();
  initialize_zorbrist();
  initialize_movegen();

  UciEngine engine;
  engine.start(std::cin);
}
