#include "game/search.h"
#include "game/Position.h"
#include "game/movegen.h"
#include "game/utils.h"
#include "game/Thinker.h"
#include "game/string_utils.h"

#include "weights.h"

#include <condition_variable>
#include <deque>
#include <iostream>
#include <mutex>
#include <sstream>
#include <unordered_set>

// TODO: Add support for "ponder" and "ponderhit" command.
// TODO: Add support for remaining options.

using namespace ChessEngine;

GoCommand make_go_command(std::deque<std::string> *command, Position *pos) {
  GoCommand goCommand;

  goCommand.pos = *pos;

  std::unordered_set<std::string> moves;
  std::string lastCommand = "";
  while (command->size() > 0) {
    std::string part = command->front();
    command->pop_front();

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
      moves.insert(part);
    } else {
      lastCommand = part;
    }
  }

  std::unordered_set<std::string> legalMoves = compute_legal_moves_set(pos);
  if (moves.size() == 0) {
    moves = legalMoves;
  }

  // Remove invalid moves.
  for (const auto& move : moves) {
    if (legalMoves.contains(move)) {
      goCommand.moves.insert(move);
    }
  }

  return goCommand;
}

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
  ThinkerInterface *thinkerInterface() {
    return &thinker;
  }
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

template<class T>
void pop_front(std::deque<T> *A, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    A->pop_front();
  }
}

class ProbeTask : public Task {
 public:
  ProbeTask(std::deque<std::string> command) : command(command) {}
  void start(UciEngineState *state) {
    if (command[0] != "probe") {
      throw std::runtime_error("Expected command to start with \"probe\"");
    }
    command.pop_front();

    Position query = state->pos;
    if (command[0] == "fen") {
      std::vector<std::string> fen(command.begin() + 1, command.begin() + 7);
      query = Position(join(fen, " "));
      pop_front(&command, 6);
    }
    if (command[0] == "moves") {
      command.pop_front();
      while (command.size()) {
        if (!make_uci_move(&query, command[0])) {
          std::cout << "Invalid uci move " << repr(command[0]) << std::endl;
          return;
        }
        command.pop_front();
      }
    }
    if (command.size() > 0) {
      throw std::runtime_error("Unexpected token \"" + command[0] + "\" in probe command");
    }

    ThinkerInterface *thinker = state->thinkerInterface();
    CacheResult cr = thinker->probe_tt(query.hash_);
    if (query.turn_ == Color::BLACK) {
      cr = cr.flip();
    }
    std::cout << cr;

    if (cr.bestMove != kNullMove)  {
      if (query.turn_ == Color::WHITE) {
        make_move<Color::WHITE>(&query, cr.bestMove);
      } else {
        make_move<Color::BLACK>(&query, cr.bestMove);
      }
      cr = thinker->probe_tt(query.hash_);
      if (!isNullCacheResult(cr)) {
        std::cout << "(response " << cr.bestMove << ")";
      }
    }

    std::cout << std::endl;
  }
 private:
  std::deque<std::string> command;
};

class HashTask : public Task {
 public:
  void start(UciEngineState *state) {
    std::cout << state->pos.hash_ << std::endl;
  }
};

class PrintFenTask : public Task {
 public:
  void start(UciEngineState *state) {
    std::cout << state->pos.fen() << std::endl;
  }
};

class SilenceTask : public Task {
 public:
  SilenceTask(std::deque<std::string> command) : command(command) {}
  void start(UciEngineState *state) {
    if (command.at(1) == "1") {
      std::cout.setstate(std::ios::failbit);
    } else {
      std::cout.clear();
    }
  }
 private:
  std::deque<std::string> command;
};

#ifdef PRINT_DEBUG
class PrintDebugTask : public Task {
 public:
  PrintDebugTask(std::deque<std::string> command) : command(command) {}
  void start(UciEngineState *state) {
    gPrintDebug = true;
  }
 private:
  std::deque<std::string> command;
};
#endif

class EvalTask : public Task {
 public:
  EvalTask(std::deque<std::string> command) : command(command) {}
  void start(UciEngineState *state) {
    if (command.size() > 1 && command.at(1) == "quiet") {
      #if NNUE_EVAL
      Thread thread(0, state->pos, compute_legal_moves_set(&state->pos));
      #else
      Thread thread(0, state->pos, state->thinkerInterface()->get_evaluator(), compute_legal_moves_set(&state->pos));
      #endif
      if (state->pos.turn_ == Color::WHITE) {
        SearchResult<Color::WHITE> result = qsearch<Color::WHITE>(&state->thinker, &thread, 0, 0, kMinEval, kMaxEval);
        std::cout << result.score << std::endl;
      } else {
        SearchResult<Color::BLACK> result = qsearch<Color::BLACK>(&state->thinker, &thread, 0, 0, kMinEval, kMaxEval);
        std::cout << result.score << std::endl;
      }
      return;
    }

    #if NNUE_EVAL
      state->pos.set_network(state->thinkerInterface()->get_nnue());
      if (state->pos.turn_ == Color::WHITE) {
        std::cout << nnue_evaluate<Color::WHITE>(state->pos) << std::endl;
      } else {
        std::cout << nnue_evaluate<Color::BLACK>(state->pos) << std::endl;
      }
    #else
      Evaluator& evaluator = state->thinkerInterface()->get_evaluator();
      state->pos.set_piece_maps(state->thinkerInterface()->get_piece_maps());
      if (state->pos.turn_ == Color::WHITE) {
        std::cout << evaluator.score<Color::WHITE>(state->pos) << std::endl;
      } else {
        std::cout << evaluator.score<Color::BLACK>(state->pos) << std::endl;
      }
      if (command.size() > 1 && command.at(1) == "vec") {
        const int32_t time = evaluator.features[EF::TIME];
        const int32_t ineq = std::min<int32_t>(1, std::max<int32_t>(-1, (evaluator.features[EF::OUR_KNIGHTS] - evaluator.features[EF::THEIR_KNIGHTS]) * 3
          + (evaluator.features[EF::OUR_BISHOPS] - evaluator.features[EF::THEIR_BISHOPS]) * 3
          + (evaluator.features[EF::OUR_ROOKS] - evaluator.features[EF::THEIR_ROOKS]) * 5
          + (evaluator.features[EF::OUR_QUEENS] - evaluator.features[EF::THEIR_QUEENS]) * 9));
        for (int i = 0; i < EF::NUM_EVAL_FEATURES; ++i) {
          int x = evaluator.features[i];
          int w = evaluator.earlyW[i] * (18 - time) / 18 + evaluator.lateW[i] * time / 18 + evaluator.ineqW[i] * ineq;
          std::cout << rjust(std::to_string(x), 6)
                    << rjust(std::to_string(w * x), 6)
                    << "  // " << EFSTR[i] << std::endl;
        }
        for (int i = 0; i < PieceMapType::PieceMapTypeCount; ++i) {
          std::cout << state->pos.pieceMapScores[i] << "  // piece map" << std::endl;
        }
      }
    #endif
  }
 private:
  std::deque<std::string> command;
};

class PlayTask : public Task {
 public:
  PlayTask(std::deque<std::string> command) : command(command), isRunning(false) {}
  void start(UciEngineState *state) override {
    assert(!isRunning);
    isRunning = true;
    assert(command.at(0) == "play");
    command.pop_front();
    GoCommand goCommand = make_go_command(&command, &state->pos);
    this->thread = new std::thread(PlayTask::_threaded_think, state, goCommand, &this->isRunning);
  }
  bool is_running() override {
    return isRunning;
  }
 private:
  static void _threaded_think(UciEngineState *state, GoCommand goCommand, bool *isRunning) {
    while (true) {
      goCommand.moves = compute_legal_moves_set(&goCommand.pos);
      SearchResult<Color::WHITE> result = Search::search(&state->thinker, goCommand, nullptr);
      if (result.move == kNullMove) {
        break;
      }
      std::cout << " " << result.move << std::flush;

      if (goCommand.pos.turn_ == Color::WHITE) {
        make_move<Color::WHITE>(&goCommand.pos, result.move);
        if (is_checkmate<Color::BLACK>(&goCommand.pos)) {
          break;
        }
      } else {
        make_move<Color::BLACK>(&goCommand.pos, result.move);
        if (is_checkmate<Color::WHITE>(&goCommand.pos)) {
          break;
        }
      }
      if (goCommand.pos.is_draw_assuming_no_checkmate()) {
        break;
      }
    }

    std::cout << std::endl;

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

class UndoTask : public Task {
 public:
  UndoTask(std::deque<std::string> command) {}
  void start(UciEngineState *state) {
    if (state->pos.history_.size() == 0) {
      std::cout << "No moves to undo" << std::endl;
      return;
    }
    if (state->pos.turn_ == Color::WHITE) {
      undo<Color::BLACK>(&state->pos);
    } else {
      undo<Color::WHITE>(&state->pos);
    }
  }
};


class PrintOptionsTask : public Task {
 public:
  void start(UciEngineState *state) {
    std::cout << "MultiPV: " << state->thinkerInterface()->get_multi_pv() << " variations" << std::endl;
    std::cout << "Threads: " << state->thinkerInterface()->get_num_threads() << " threads" << std::endl;
    std::cout << "Hash: " << state->thinkerInterface()->get_cache_size_kb() << " kilobytes" << std::endl;
  }
};

class QuitTask : public Task {
 public:
  void start(UciEngineState *state) {
    exit(0);
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

#if NNUE_EVAL
class LoadNnueTask : public Task {
 public:
  LoadNnueTask(std::deque<std::string> command) : command(command) {}
  void start(UciEngineState *state) {
    assert(command.at(0) == "loadnnue");
    if (command.size() != 2) {
      invalid(join(command, " "));
    }

    std::ifstream myfile;
    myfile.open(command.at(1));
    if (!myfile.is_open()) {
      std::cout << "Error opening file \"" << command.at(1) << "\"" << std::endl;
      exit(0);
    }
    state->thinkerInterface()->load_nnue(myfile);
    myfile.close();
  }
  std::deque<std::string> command;
};
#else
class LoadWeightsTask : public Task {
 public:
  LoadWeightsTask(std::deque<std::string> command) : command(command) {}
  void start(UciEngineState *state) {
    assert(command.at(0) == "loadweights");
    if (command.size() != 2) {
      invalid(join(command, " "));
    }
    std::ifstream myfile;
    myfile.open(command.at(1));
    if (!myfile.is_open()) {
      std::cout << "Error opening file \"" << command.at(1) << "\"" << std::endl;
      exit(0);
    }
    state->thinkerInterface()->load_weights(myfile);
    myfile.close();
  }
  std::deque<std::string> command;
};
struct DumpEncodedWeights : public Task {
 public:
  void start(UciEngineState *state) {
    // Save weights to string
    std::string weights;
    {
      std::ostringstream stream;
      state->thinkerInterface()->save_weights(stream);
      weights = stream.str();
    }

    std::ofstream myfile;
    myfile.open("src/weights.h");
    myfile << "const char *kDefaultWeights = \"";
    for (size_t i = 0; i < weights.size(); ++i) {
      uint8_t val = weights[i];
      myfile << char(val % 16 + '0');
      myfile << char(val / 16 + '0');
    }
    myfile << "\";\n";
    myfile.close();
  }
};
struct ReadEncodedWeights : public Task {
  public:
    void start(UciEngineState *state) {
      // Decode
      std::string weights;
      const char *c = kDefaultWeights;
      while (*c != '\0') {
        weights += char((*c - '0') + (*(c + 1) - '0') * 16);
        c += 2;
      }
      std::stringstream stream(weights);
      state->thinkerInterface()->load_weights(stream);
    }
};
#endif

class NewGameTask : public Task {
 public:
  void start(UciEngineState *state) {
    state->thinkerInterface()->clear_tt();
  }
};

bool does_pattern_match(const std::deque<std::string>& text, const std::vector<std::string>& pattern) {
  if (text.size() != pattern.size()) {
    return false;
  }
  for (size_t i = 0; i < text.size(); ++i) {
    if (pattern[i] != "*" && text[i] != pattern[i]) {
      return false;
    }
  }
  return true;
}

class SetOptionTask : public Task {
 public:
  SetOptionTask(std::deque<std::string> command) : command(command) {}
  void start(UciEngineState *state) {
    assert(command.size() > 0 && command[0] == "setoption");
    command.pop_front();
    if (command.size() == 0 || command[0] != "name") {
      invalid(join(command, " "));
      return;
    }
    command.pop_front();

    if(does_pattern_match(command, {"Move", "Overhead", "value", "*"})) {
      state->thinkerInterface()->set_move_overhead_ms(std::stoi(command[3]));
    } else if (does_pattern_match(command, {"Clear", "Hash"})) {
      state->thinkerInterface()->clear_tt();
      return;
    } else if (does_pattern_match(command, {"MultiPV", "value", "*"})) {
      int multiPV;
      try {
        multiPV = std::stoi(command[2]);
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
      state->thinkerInterface()->set_multi_pv(multiPV);
      return;
    } else if (does_pattern_match(command, {"Threads", "value", "*"})) {
      int numThreads;
      try {
        numThreads = std::stoi(command[2]);
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
      state->thinkerInterface()->set_num_threads(numThreads);
      return;
    } else if (does_pattern_match(command, {"Hash", "value", "*"})) {
      int cacheSize;
      try {
        cacheSize = std::stoi(command[2]);
        if (cacheSize <= 0) {
          throw std::invalid_argument("Value must be at least 1");
        }
      } catch (std::invalid_argument&) {
        std::cout << "Value must be an integer" << std::endl;
        return;
      }
      state->thinkerInterface()->set_cache_size(cacheSize * 1000);
      return;
    } else if (does_pattern_match(command, {"SyzygyPath", "value", "*"})) {
      // TODO
      return;
    } else if (does_pattern_match(command, {"UCI_ShowWDL", "value", "*"})) {
      // TODO
      return;
    } else {
      std::cout << "Unrecognized option" << std::endl;
    }
  }
 private:
  std::deque<std::string> command;
};

class IsReadyTask : public Task {
 public:
  void start(UciEngineState *state) {
    std::cout << "readyok" << std::endl;
  }
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
 public:
  GoTask(std::deque<std::string> command) : command(command), isRunning(false), thread(nullptr) {}
  void start(UciEngineState *state) override {
    assert(!isRunning);
    isRunning = true;
    assert(command.at(0) == "go");
    command.pop_front();
    GoCommand goCommand = make_go_command(&command, &state->pos);
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

    SearchResult<Color::WHITE> result = Search::search(&state->thinker, goCommand, state->stopThinkingSwitch, [state](Position *position, VariationHead<Color::WHITE> results, size_t depth, double secs) {
      GoTask::_print_variations(state, depth, secs);
    });

    if (state->thinkerInterface()->get_variations().size() > 0) {
      VariationHead<Color::WHITE> head = state->thinkerInterface()->get_variations()[0];
      std::cout << "bestmove " << head.move;
      if (head.response != kNullMove) {
        std::cout << " ponder " << head.response;
      }
      std::cout << std::endl;
    }

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
    const size_t multiPV = state->thinkerInterface()->get_multi_pv();
    const uint64_t timeMs = secs * 1000;
    std::vector<VariationHead<Color::WHITE>> variations = state->thinkerInterface()->get_variations();
    if (variations.size() == 0) {
      if (isStalemate(&state->pos)) {
        std::cout << "info depth 0 score cp 0" << std::endl;
        return;
      } else {
        throw std::runtime_error("todo");
      }
    }
    for (size_t i = 0; i < std::min(multiPV, variations.size()); ++i) {
      std::pair<Evaluation, std::vector<Move>> variation = state->thinkerInterface()->get_variation(&state->pos, variations[i].move);

      Evaluation eval = variation.first;
      if (state->pos.turn_ == Color::BLACK) {
        // Score should be from mover's perspective, not white's.
        eval *= -1;
      }

      std::cout << "info depth " << depth;
      std::cout << " multipv " << (i + 1);
      if (eval <= kLongestForcedMate) {
        std::cout << " score mate " << -(eval - kCheckmate + 1) / 2;
      } else if (eval >= -kLongestForcedMate) {
        std::cout << " score mate " << -(eval + kCheckmate - 1) / 2;
      } else {
        std::cout << " score cp " << eval;
      }
      std::cout << " nodes " << state->thinkerInterface()->get_node_count();
      std::cout << " nps " << uint64_t(double(state->thinkerInterface()->get_node_count()) / secs);
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

void wait_for_task(UciEngineState *state) {
  state->taskQueueLock.lock();
  if (state->taskQueue.size() > 0) {
    state->taskQueueLock.unlock();
    return;
  }
  state->taskQueueLock.unlock();
  while (true) {
    std::unique_lock<std::mutex> lock(state->mutex);
    state->condVar.wait(lock);  // Wait for data
    state->taskQueueLock.lock();
    if (state->taskQueue.size() > 0) {
      state->taskQueueLock.unlock();
      return;
    }
    state->taskQueueLock.unlock();
  }
}

struct UciEngine {
  UciEngineState state;

  UciEngine() {
    this->state.stopThinkingSwitch = nullptr;
    this->state.pos = Position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

    #if NNUE_EVAL
    LoadNnueTask task({"loadnnue", "nnue-776-48-16-1.bin"});
    task.start(&this->state);
    #else
    #if INCLUDE_WEIGHTS
    ReadEncodedWeights task;
    #else
    LoadWeightsTask task({"loadweights", "weights.txt"});
    #endif
    task.start(&this->state);
    #endif
  }
  void start(std::istream& cin, const std::vector<std::string>& commands) {
    UciEngineState *state = &this->state;

    if (commands.size() == 0) {
      std::cout << "id name Pumpkin 0.0" << std::endl;
      std::cout << "id author Morgan Redding" << std::endl << std::endl;

      // Garbage boiler plate to make the GUI happy.
      std::cout << "option name Threads type spin default 1 min 1 max 1024" << std::endl;
      std::cout << "option name Hash type spin default 16 min 1 max 33554432" << std::endl;
      std::cout << "option name Clear Hash type button" << std::endl;
      std::cout << "option name Ponder type check default false" << std::endl;
      std::cout << "option name MultiPV type spin default 1 min 1 max 500" << std::endl;
      std::cout << "option name Skill Level type spin default 20 min 0 max 20" << std::endl;
      std::cout << "option name Move Overhead type spin default 10 min 0 max 5000" << std::endl;
      std::cout << "option name Slow Mover type spin default 100 min 10 max 1000" << std::endl;
      std::cout << "option name nodestime type spin default 0 min 0 max 10000" << std::endl;
      std::cout << "option name UCI_Chess960 type check default false" << std::endl;
      std::cout << "option name UCI_AnalyseMode type check default false" << std::endl;
      std::cout << "option name UCI_LimitStrength type check default false" << std::endl;
      std::cout << "option name UCI_Elo type spin default 1320 min 1320 max 3190" << std::endl;
      std::cout << "option name UCI_ShowWDL type check default false" << std::endl;
      std::cout << "option name SyzygyPath type string default <empty>" << std::endl;
      std::cout << "option name SyzygyProbeDepth type spin default 1 min 1 max 100" << std::endl;
      std::cout << "option name Syzygy50MoveRule type check default true" << std::endl;
      std::cout << "option name SyzygyProbeLimit type spin default 7 min 0 max 7" << std::endl;

      std::cout << "uciok" << std::endl;
    }

    for (std::string command : commands) {
      if(command.find_first_not_of(' ') == std::string::npos) {
        continue;
      }
      this->handle_uci_command(state, &command);
    }

    std::thread eventRunner([state]() {
      while (true) {
        wait_for_task(state);

        // Wait until not busy.
        state->taskQueueLock.lock();
        while (state->currentTask != nullptr && state->currentTask->is_running()) {
          state->taskQueueLock.unlock();
          std::unique_lock<std::mutex> lock(state->mutex);
          state->condVar.wait(lock);  // Wait for data
          state->taskQueueLock.lock();
        }

        if (state->taskQueue.size() == 0) {
          throw std::runtime_error("No task to enque");
        }

        state->currentTask = state->taskQueue.front();
        state->taskQueue.pop_front();
        state->currentTask->start(state);
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

    state->taskQueueLock.lock();
    if (parts[0] == "position" || parts[0] == "p") {
      state->taskQueue.push_back(std::make_shared<PositionTask>(parts));
    } else if (parts[0] == "go") {
      state->taskQueue.push_back(std::make_shared<GoTask>(parts));
    } else if (parts[0] == "setoption" || parts[0] == "so") {
      state->taskQueue.push_back(std::make_shared<SetOptionTask>(parts));
    } else if (parts[0] == "ucinewgame") {
      state->taskQueue.push_back(std::make_shared<NewGameTask>());
    } else if (parts[0] == "stop") {
      // This runs immediately.
      StopTask task;
      task.start(state);
#if NNUE_EVAL
    } else if (parts[0] == "loadnnue") {  // Custom commands below this line.
      state->taskQueue.push_back(std::make_shared<LoadNnueTask>(parts));
#else
    } else if (parts[0] == "loadweights" || parts[0] == "lw") {  // Custom commands below this line.
      state->taskQueue.push_back(std::make_shared<LoadWeightsTask>(parts));
    } else if (parts[0] == "dumpweights") {
      state->taskQueue.push_back(std::make_shared<DumpEncodedWeights>());
#endif
    } else if (parts[0] == "play") {
      state->taskQueue.push_back(std::make_shared<PlayTask>(parts));
    } else if (parts[0] == "printoptions") {
      state->taskQueue.push_back(std::make_shared<PrintOptionsTask>());
    } else if (parts[0] == "isready") {
      state->taskQueue.push_back(std::make_shared<IsReadyTask>());
    } else if (parts[0] == "move" || parts[0] == "m") {
      state->taskQueue.push_back(std::make_shared<MoveTask>(parts));
    } else if (parts[0] == "undo") {
      state->taskQueue.push_back(std::make_shared<UndoTask>(parts));
    } else if (parts[0] == "eval" || parts[0] == "e") {
      state->taskQueue.push_back(std::make_shared<EvalTask>(parts));
    } else if (parts[0] == "probe") {
      state->taskQueue.push_back(std::make_shared<ProbeTask>(parts));
    } else if (parts[0] == "hash") {
      state->taskQueue.push_back(std::make_shared<HashTask>());
    } else if (parts[0] == "lazyquit") {
      state->taskQueue.push_back(std::make_shared<QuitTask>());
    } else if (parts[0] == "printfen") {
      state->taskQueue.push_back(std::make_shared<PrintFenTask>());
    } else if (parts[0] == "silence") {
      state->taskQueue.push_back(std::make_shared<SilenceTask>(parts));
    #ifdef PRINT_DEBUG
    } else if (parts[0] == "printdebug") {
      state->taskQueue.push_back(std::make_shared<PrintDebugTask>(parts));
    #endif
    } else if (parts[0] == "ponderhit") {
      // Ignore. (TODO: handle pondering better).
    } else {
      state->taskQueue.push_back(std::make_shared<UnrecognizedCommandTask>(parts));
    }
    state->taskQueueLock.unlock();

    std::unique_lock<std::mutex> lock(state->mutex);
    state->condVar.notify_one();
  }
};

int main(int argc, char *argv[]) {
  std::cout << "Pumpkin 0.0" << std::endl;

  std::vector<std::string> commands;
  for (int i = 1; i < argc; ++i) {
    commands.push_back(argv[i]);
  }

  // Wait for "uci" command.
  if (commands.size() == 0) {
    while (true) {
      std::string line;
      getline(std::cin, line);
      if (line == "uci") {
        break;
      } else {
        std::cout << "Unrecognized command " << repr(line) << std::endl;
      }
    }
  }

  initialize_geometry();
  initialize_zorbrist();
  initialize_movegen();

  UciEngine engine;
  engine.start(std::cin, commands);
}
