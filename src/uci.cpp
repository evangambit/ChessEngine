// g++ src/uci.cpp src/game/*.cpp -std=c++20 -O3 -DNDEBUG -o uci
// g++ src/uci.cpp src/game/*.cpp -std=c++20 -O3 -DNDEBUG -DSquareControl -o sc-old

#import "game/search.h"
#import "game/Position.h"
#import "game/movegen.h"
#import "game/utils.h"
#import "game/string_utils.h"

#include <condition_variable>
#include <deque>
#include <iostream>
#include <mutex>
#include <unordered_set>

using namespace ChessEngine;

void invalid(const std::string& command) {
  std::cout << "Invalid use of " << repr(command) << " command" << std::endl;
}

void invalid(const std::string& command, const std::string& message) {
  std::cout << "Invalid use of " << repr(command) << " command (" << message << ")" << std::endl;
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


struct UciEngineState;

class Task {
 public:
  virtual void start(UciEngineState *state) = 0;

  virtual bool is_running() {
    return false;
  }

  bool is_slow() {
    return false;
  }
};

struct UciEngineState {
  Thinker thinker;
  Position pos;
  std::shared_ptr<StopThinkingSwitch> stopThinkingSwitch;

  std::mutex mutex;
  std::condition_variable condVar;

  std::deque<std::shared_ptr<Task>> taskQueue;
  SpinLock taskQueueLock;
  std::shared_ptr<Task> currentTask;
};

class UnrecognizedCommandTask : public Task {
 public:
  UnrecognizedCommandTask(std::deque<std::string> command) : command(command) {}
  void start(UciEngineState *state) {
    std::cout << "Unrecognized command \"" << join(command, " ") << "\"" << std::endl;
  }
 private:
  std::deque<std::string> command;
};

class ProbeTask : public Task {
 public:
  ProbeTask(std::deque<std::string> command) : command(command) {}
  void start(UciEngineState *state) {
    Position query = Position::init();
    size_t i = 1;
    if (command[1] == "fen") {
      std::vector<std::string> fen(command.begin() + 2, command.begin() + 8);
      query = Position(join(fen, " "));
      i = 8;
    }
    if (command[i] == "moves") {
      while (++i < command.size()) {
        if (!make_uci_move(&query, command[i])) {
          std::cout << "Invalid uci move " << repr(command[i]) << std::endl;
          return;
        }
      }
    }

    std::pair<CacheResult, std::vector<Move>> variation = state->thinker.get_variation(&query, kNullMove);

    if (isNullCacheResult(variation.first)) {
      std::cout << "Cache result for " << query.fen() << " is missing" << std::endl;
    }
    std::cout << variation.first.eval;
    for (const auto& move : variation.second) {
      std::cout << " " << move;
    }
    std::cout << std::endl;
  }
 private:
  std::deque<std::string> command;
};

class EvalTask : public Task {
 public:
  EvalTask(std::deque<std::string> command) : command(command) {}
  void start(UciEngineState *state) {
    if (command.size() > 1 && command.at(1) == "quiet") {
      Thread thread(0, state->pos, state->thinker.evaluator, compute_legal_moves_set(&state->pos));
      SearchResult<Color::WHITE> result;
      if (state->pos.turn_ == Color::WHITE) {
        result = qsearch<Color::WHITE>(&state->thinker, &thread, 0, 0, kMinEval, kMaxEval);
      } else {
        result = to_white(qsearch<Color::BLACK>(&state->thinker, &thread, 0, 0, kMinEval, kMaxEval));
      }
      std::cout << result.score << std::endl;
      return;
    }
    Evaluator& evaulator = state->thinker.evaluator;
    if (state->pos.turn_ == Color::WHITE) {
      std::cout << evaulator.score<Color::WHITE>(state->pos) << std::endl;
    } else {
      std::cout << -evaulator.score<Color::BLACK>(state->pos) << std::endl;
    }
  }
 private:
  std::deque<std::string> command;
};

class PlayTask : public Task {
 public:
  PlayTask(std::deque<std::string> command) : command(command), isRunning(false) {}
  void start(UciEngineState *state) override {
    std::cout << "PlayTask::start" << std::endl;
    assert(!isRunning);
    isRunning = true;
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
    
    this->thread = new std::thread(PlayTask::_threaded_think, state, &this->isRunning, depthLimit, nodeLimit, timeLimitMs);
  }
  bool is_running() override {
    return isRunning;
  }
 private:
  static void _threaded_think(UciEngineState *state, bool *isRunning, size_t depthLimit, uint64_t nodeLimit, uint64_t timeLimitMs) {
    Position pos(state->pos);

    while (true) {
      state->thinker.stopThinkingCondition = std::make_unique<OrStopCondition>(
        std::make_shared<StopThinkingNodeCountCondition>(nodeLimit),
        std::make_shared<StopThinkingTimeCondition>(timeLimitMs)
      );

      // TODO: get rid of this (selfplay2 sometimes crashes when we try to get rid of it now).
      state->thinker.reset_stuff();

      SearchResult<Color::WHITE> result = search(&state->thinker, &pos, depthLimit);
      if (result.move == kNullMove) {
        break;
      }
      std::cout << " " << result.move << std::flush;
      if (pos.turn_ == Color::WHITE) {
        make_move<Color::WHITE>(&pos, result.move);
        if (is_checkmate<Color::BLACK>(&pos)) {
          std::cout << std::endl;
          break;
        }
      } else {
        make_move<Color::BLACK>(&pos, result.move);
        if (is_checkmate<Color::WHITE>(&pos)) {
          std::cout << std::endl;
          break;
        }
      }
      if (pos.is_draw(0)) {
        std::cout << std::endl;
        break;
      }
    }

    *isRunning = false;

    // Notify run-loop that it can start running a new command.
    std::unique_lock<std::mutex> lock(state->mutex);
    state->condVar.notify_one();
  }
  bool isRunning;
  std::thread *thread;
  std::deque<std::string> command;
};

class MoveTask : public Task {
 public:
  MoveTask(std::deque<std::string> command) : command(command) {}
  void start(UciEngineState *state) {
    size_t i = 0;
    while (++i < command.size()) {
      std::string uciMove = command[i];
      ExtMove moves[kMaxNumMoves];
      ExtMove *end;
      if (state->pos.turn_ == Color::WHITE) {
        end = compute_legal_moves<Color::WHITE>(&state->pos, moves);
      } else {
        end = compute_legal_moves<Color::BLACK>(&state->pos, moves);
      }
      bool foundMove = false;
      for (ExtMove *move = moves; move < end; ++move) {
        if (move->move.uci() == uciMove) {
          foundMove = true;
          if (state->pos.turn_ == Color::WHITE) {
            make_move<Color::WHITE>(&state->pos, move->move);
          } else {
            make_move<Color::BLACK>(&state->pos, move->move);
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
 private:
  std::deque<std::string> command;
};

class PrintOptionsTask : public Task {
 public:
  void start(UciEngineState *state) {
    std::cout << "MultiPV: " << state->thinker.multiPV << " variations" << std::endl;
    std::cout << "Threads: " << state->thinker.numThreads << " threads" << std::endl;
    std::cout << "Hash: " << state->thinker.cache.kb_size() << " kilobytes" << std::endl;
  }
};

class StopTask : public Task {
 public:
  void start(UciEngineState *state) {
    // TODO: deque all future thinking tasks?
    if (state->stopThinkingSwitch != nullptr) {
      state->stopThinkingSwitch->stop();
    }
  }
};

class LoadWeightsTask : public Task {
 public:
  LoadWeightsTask(std::deque<std::string> command) : command(command) {}
  void start(UciEngineState *state) {
    assert(command.at(0) == "loadweights");
    if (command.size() != 2) {
      invalid(join(command, " "));
    }
    state->thinker.load_weights_from_file(command.at(1));
  }
  std::deque<std::string> command;
};

class NewGameTask : public Task {
 public:
  void start(UciEngineState *state) {
    state->thinker.reset_stuff();
  }
};

class SetOptionTask : public Task {
 public:
  SetOptionTask(std::deque<std::string> command) : command(command) {}
  void start(UciEngineState *state) {
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
        state->thinker.reset_stuff();
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
        state->thinker.multiPV = multiPV;
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
        state->thinker.numThreads = numThreads;
        return;
      } else if (name == "Hash") {
        int cacheSize;
        try {
          cacheSize = stoi(value);
          if (cacheSize <= 0) {
            throw std::invalid_argument("Value must be at least 1");
          }
        } catch (std::invalid_argument&) {
          std::cout << "Value must be an integer" << std::endl;
          return;
        }
        state->thinker.set_cache_size(cacheSize);
        return;
      }
      std::cout << "Unrecognized option " << repr(name) << std::endl;
    }
  }
 private:
  std::deque<std::string> command;
};

class PositionTask : public Task {
 public:
  PositionTask(std::deque<std::string> command) : command(command) {}
  void start(UciEngineState *state) {
    if (command.size() < 2) {
      invalid(join(command, " "));
      return;
    }
    size_t i;
    if (command[1] == "startpos") {
      state->pos = Position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
      i = 2;
    } else if (command.size() >= 8 && command[1] == "fen") {
      std::vector<std::string> fen(command.begin() + 2, command.begin() + 8);
      i = 8;
      state->pos = Position(join(fen, " "));
    } else {
      invalid(join(command, " "));
      return;
    }
    if (i == command.size()) {
      std::cout << "Position set to \"" << state->pos.fen() << std::endl;
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
      if (state->pos.turn_ == Color::WHITE) {
        end = compute_legal_moves<Color::WHITE>(&state->pos, moves);
      } else {
        end = compute_legal_moves<Color::BLACK>(&state->pos, moves);
      }
      bool foundMove = false;
      for (ExtMove *move = moves; move < end; ++move) {
        if (move->move.uci() == uciMove) {
          foundMove = true;
          if (state->pos.turn_ == Color::WHITE) {
            make_move<Color::WHITE>(&state->pos, move->move);
          } else {
            make_move<Color::BLACK>(&state->pos, move->move);
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
 private:
  std::deque<std::string> command;
};

class GoTask : public Task {
  struct GoCommand {
    GoCommand()
    : depthLimit(-1), nodeLimit(-1), timeLimitMs(-1),
    wtimeMs(-1), btimeMs(-1), wIncrementMs(-1), bIncrementMs(-1), movesUntilTimeControl(-1) {}
    size_t depthLimit;
    uint64_t nodeLimit;
    uint64_t timeLimitMs;
    std::unordered_set<std::string> moves;

    uint64_t wtimeMs;
    uint64_t btimeMs;
    uint64_t wIncrementMs;
    uint64_t bIncrementMs;
    uint64_t movesUntilTimeControl;
  };
 public:
  GoTask(std::deque<std::string> command) : command(command), isRunning(false), thread(nullptr) {}
  void start(UciEngineState *state) override {
    assert(!isRunning);
    isRunning = true;
    assert(command.at(0) == "go");
    command.pop_front();

    GoCommand goCommand;

    std::string lastCommand = "";
    while (command.size() > 0) {
      std::string part = command.front();
      command.pop_front();

      if (part == "depth"
        || part == "nodes"
        || part == "movetime"
        || part == "wtime"
        || part == "btime"
        || part == "winc"
        || part == "binc"
        || part == "movestogo"
        || part == "searchmoves"
        ) {
        lastCommand = part;
      } else if (lastCommand == "depth") {
        goCommand.depthLimit = stoi(part);
      } else if (lastCommand == "nodes") {
        goCommand.nodeLimit = stoi(part);
      } else if (lastCommand == "movetime") {
        goCommand.timeLimitMs = stoi(part);
      } else if (lastCommand == "wtime") {
        goCommand.wtimeMs = stoi(part);
      } else if (lastCommand == "btime") {
        goCommand.btimeMs = stoi(part);
      } else if (lastCommand == "winc") {
        goCommand.wIncrementMs = stoi(part);
      } else if (lastCommand == "binc") {
        goCommand.bIncrementMs = stoi(part);
      } else if (lastCommand == "movestogo") {
        goCommand.movesUntilTimeControl = stoi(part);
      } else if (lastCommand == "searchmoves") {
        goCommand.moves.insert(part);
      } else {
        lastCommand = part;
      }
    }

    std::unordered_set<std::string> legalMoves = compute_legal_moves_set(&state->pos);
    if (goCommand.moves.size() == 0) {
      goCommand.moves = legalMoves;
    }

    for (const auto& move : goCommand.moves) {
      if (!legalMoves.contains(move)) {
        invalid(join(command, " "), "Invalid move \"" + move + "\"");
        return;
      }
    }

    this->thread = new std::thread(GoTask::_threaded_think, state, goCommand, &this->isRunning);
  }

  ~GoTask() {
    assert(!isRunning);
    assert(this->thread != nullptr);
    this->thread->join();
    delete this->thread;
  }

  static void _threaded_think(UciEngineState *state, GoCommand goCommand, bool *isRunning) {
    state->stopThinkingSwitch = std::make_shared<StopThinkingSwitch>();

    state->thinker.stopThinkingCondition = std::make_unique<OrStopCondition>(
      std::make_shared<StopThinkingNodeCountCondition>(goCommand.nodeLimit),
      std::make_shared<StopThinkingTimeCondition>(goCommand.timeLimitMs),
      state->stopThinkingSwitch
    );

    // TODO: get rid of this (selfplay2 sometimes crashes when we try to get rid of it now).
    state->thinker.reset_stuff();

    SearchResult<Color::WHITE> result = search(&state->thinker, &state->pos, goCommand.depthLimit, goCommand.moves, [state](Position *position, SearchResult<Color::WHITE> results, size_t depth, double secs) {
      GoTask::_print_variations(state, depth, secs);
    });

    if (state->pos.turn_ == Color::WHITE) {
      make_move<Color::WHITE>(&state->pos, result.move);
    } else {
      make_move<Color::BLACK>(&state->pos, result.move);
    }
    CacheResult cr = state->thinker.cache.find<false>(state->pos.hash_);
    if (state->pos.turn_ == Color::WHITE) {
      undo<Color::BLACK>(&state->pos);
    } else {
      undo<Color::WHITE>(&state->pos);
    }

    std::cout << "bestmove " << result.move;
    if (!isNullCacheResult(cr)) {
      std::cout << " ponder " << cr.bestMove;
    }
    std::cout << std::endl;

    *isRunning = false;

    // Notify run-loop that it can start running a new command.
    std::unique_lock<std::mutex> lock(state->mutex);
    state->condVar.notify_one();
  }
  bool is_running() override {
    return isRunning;
  }
 private:
  static void _print_variations(UciEngineState* state, int depth, double secs) {
    const size_t multiPV = state->thinker.multiPV;
    const uint64_t timeMs = secs * 1000;
    std::vector<SearchResult<Color::WHITE>> variations = state->thinker.variations;
    if (variations.size() == 0) {
      throw std::runtime_error("No variations found!");
    }
    const int32_t pawnValue = state->thinker.evaluator.earlyW[EF::PAWNS] + state->thinker.evaluator.clippedW[EF::PAWNS];
    for (size_t i = 0; i < std::min(multiPV, variations.size()); ++i) {
      std::pair<CacheResult, std::vector<Move>> variation = state->thinker.get_variation(&state->pos, variations[i].move);
      std::cout << "info depth " << depth;
      std::cout << " multipv " << (i + 1);
      if (variation.first.eval <= kLongestForcedMate) {
        std::cout << " score mate " << -(variation.first.eval - kCheckmate + 1) / 2;
      } else if (variation.first.eval >= -kLongestForcedMate) {
        std::cout << " score mate " << (-variation.first.eval - kCheckmate + 1) / 2;
      } else {
        std::cout << " score cp " << (variation.first.eval * 100 / pawnValue);
      }
      std::cout << " nodes " << state->thinker.nodeCounter;
      std::cout << " nps " << uint64_t(double(state->thinker.nodeCounter) / secs);
      std::cout << " time " << timeMs;
      std::cout << " pv";
      for (const auto& move : variation.second) {
        std::cout << " " << move.uci();
      }
      std::cout << std::endl;
    }
  }
  std::deque<std::string> command;
  std::thread *thread;
  bool isRunning;
};

struct UciEngine {
  UciEngineState state;

  UciEngine() {
    this->state.stopThinkingSwitch = nullptr;
    this->state.pos = Position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    this->state.thinker.load_weights_from_file("weights.txt");
  }
  void start(std::istream& cin) {
    UciEngineState *state = &this->state;
    std::thread eventRunner([state]() {
      while (true) {
        std::unique_lock<std::mutex> lock(state->mutex);
        state->condVar.wait(lock);  // Wait for data

        state->taskQueueLock.lock();
        bool isBusy = state->currentTask != nullptr && state->currentTask->is_running();
        while (!isBusy) {
          state->currentTask = nullptr;
          if (state->taskQueue.size() == 0) {
            state->taskQueueLock.unlock();
            break;
          }
          state->currentTask = state->taskQueue.front();
          state->taskQueue.pop_front();
          state->currentTask->start(state);
          isBusy = state->currentTask != nullptr && state->currentTask->is_running();
        }
        state->taskQueueLock.unlock();
      }
    });
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

      this->handle_uci_command(state, &line);

      // Notify run-loop that there may be a new command.
      std::unique_lock<std::mutex> lock(this->state.mutex);
      this->state.condVar.notify_one();
    }
    eventRunner.join();
  }
  static void handle_uci_command(UciEngineState *state, std::string *command) {
    remove_excess_whitespace(command);
    std::vector<std::string> rawParts = split(*command, ' ');

    std::deque<std::string> parts;
    for (const auto& part : rawParts) {
      if (part.size() > 0) {
        parts.push_back(part);
      }
    }

    if (parts[0] == "position") {
      state->taskQueueLock.lock();
      state->taskQueue.push_back(std::make_shared<PositionTask>(parts));
      state->taskQueueLock.unlock();
    } else if (parts[0] == "go") {
      state->taskQueueLock.lock();
      state->taskQueue.push_back(std::make_shared<GoTask>(parts));
      state->taskQueueLock.unlock();
    } else if (parts[0] == "setoption") {
      state->taskQueueLock.lock();
      state->taskQueue.push_back(std::make_shared<SetOptionTask>(parts));
      state->taskQueueLock.unlock();
    } else if (parts[0] == "ucinewgame") {
      state->taskQueueLock.lock();
      state->taskQueue.push_back(std::make_shared<NewGameTask>());
      state->taskQueueLock.unlock();
    } else if (parts[0] == "stop") {
      // This runs immediately.
      StopTask task;
      state->taskQueueLock.lock();
      task.start(state);
      state->taskQueueLock.unlock();
    } else if (parts[0] == "loadweights") {  // Custom commands below this line.
      state->taskQueueLock.lock();
      state->taskQueue.push_back(std::make_shared<LoadWeightsTask>(parts));
      state->taskQueueLock.unlock();
    } else if (parts[0] == "play") {
      state->taskQueueLock.lock();
      state->taskQueue.push_back(std::make_shared<PlayTask>(parts));
      state->taskQueueLock.unlock();
    } else if (parts[0] == "printoptions") {
      state->taskQueueLock.lock();
      state->taskQueue.push_back(std::make_shared<PrintOptionsTask>());
      state->taskQueueLock.unlock();
    } else if (parts[0] == "move") {
      state->taskQueueLock.lock();
      state->taskQueue.push_back(std::make_shared<MoveTask>(parts));
      state->taskQueueLock.unlock();
    } else if (parts[0] == "eval") {
      state->taskQueueLock.lock();
      state->taskQueue.push_back(std::make_shared<EvalTask>(parts));
      state->taskQueueLock.unlock();
    } else if (parts[0] == "probe") {
      state->taskQueueLock.lock();
      state->taskQueue.push_back(std::make_shared<ProbeTask>(parts));
      state->taskQueueLock.unlock();
    } else {
      state->taskQueueLock.lock();
      state->taskQueue.push_back(std::make_shared<UnrecognizedCommandTask>(parts));
      state->taskQueueLock.unlock();
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
