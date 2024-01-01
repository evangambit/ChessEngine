#ifndef THINKER_H
#define THINKER_H

#include "utils.h"
#include "Evaluator.h"
#include "TranspositionTable.h"

namespace ChessEngine {

static constexpr unsigned kNumSearchManagerCounters = 32768;
static constexpr unsigned kNumSearchManagerLocks = 256;
struct SearchManager {
  uint8_t counters[kNumSearchManagerCounters];
  SpinLock locks[kNumSearchManagerLocks];
  SearchManager() {
    std::fill_n(counters, kNumSearchManagerCounters, 0);
  }
  bool should_start_searching(uint64_t hash) {
    size_t idx = hash % kNumSearchManagerCounters;
    SpinLock& lock = locks[hash % kNumSearchManagerLocks];
    lock.lock();
    bool r = counters[idx] == 0;
    if (r) {
      counters[idx] += 1;
    }
    lock.unlock();
    return r;
  }
  void start_searching(uint64_t hash) {
    size_t idx = hash % kNumSearchManagerCounters;
    SpinLock& lock = locks[hash % kNumSearchManagerLocks];
    lock.lock();
    counters[idx] += 1;
    lock.unlock();
  }
  void finished_searching(uint64_t hash) {
    size_t idx = hash % kNumSearchManagerCounters;
    SpinLock& lock = locks[hash % kNumSearchManagerLocks];
    lock.lock();
    counters[idx] -= 1;
    lock.unlock();
  }
};

struct Thinker;
struct StopThinkingCondition {
  virtual void start_thinking(const Thinker& thinker) = 0;
  virtual bool should_stop_thinking(const Thinker&) = 0;
  virtual ~StopThinkingCondition() = default;
};

struct NeverStopThinkingCondition : public StopThinkingCondition {
  void start_thinking(const Thinker& thinker) {}
  bool should_stop_thinking(const Thinker&) {
    return false;
  }
};

struct Thinker {
  size_t nodeCounter;
  Evaluator evaluator;
  TranspositionTable cache;
  uint32_t historyHeuristicTable[Color::NUM_COLORS][64][64];
  size_t multiPV;
  size_t numThreads;

  PieceMaps pieceMaps;
  uint64_t lastRootHash;

  std::vector<SearchResult<Color::WHITE>> variations;

  SpinLock stopThinkingLock;

  Thinker() : cache(10000), stopThinkingCondition(new NeverStopThinkingCondition()), lastRootHash(0) {
    reset_stuff();
    multiPV = 1;
    numThreads = 1;
  }

  void set_cache_size(size_t kilobytes) {
    cache.set_cache_size(kilobytes);
  }

  void make_move(Position* pos, Move move) const {
    if (pos->turn_ == Color::WHITE) {
      make_move<Color::WHITE>(pos, move);
    } else {
      make_move<Color::BLACK>(pos, move);
    }
  }

  void undo(Position* pos) const {
    if (pos->turn_ == Color::BLACK) {
      undo<Color::WHITE>(pos);
    } else {
      undo<Color::BLACK>(pos);
    }
  }

  void save_weights_to_file(const std::string& filename) {
    std::ofstream myfile;
    myfile.open(filename);
    if (!myfile.is_open()) {
      std::cout << "Error opening file \"" << filename << "\"" << std::endl;
      exit(0);
    }
    this->evaluator.save_weights_to_file(myfile);
    this->pieceMaps.save_weights_to_file(myfile);
    myfile.close();
  }


  void load_weights_from_file(const std::string& filename) {
    std::ifstream myfile;
    myfile.open(filename);
    if (!myfile.is_open()) {
      std::cout << "Error opening file \"" << filename << "\"" << std::endl;
      exit(0);
    }
    this->evaluator.load_weights_from_file(myfile);
    this->pieceMaps.load_weights_from_file(myfile);
    myfile.close();
  }

  std::pair<CacheResult, std::vector<Move>> get_variation(Position *pos, Move move) const {
    std::vector<Move> moves;
    if (move != kNullMove) {
      moves.push_back(move);
      this->make_move(pos, move);
    }
    CacheResult originalCacheResult = this->cache.unsafe_find(pos->hash_);
    CacheResult cr = originalCacheResult;

    if (isNullCacheResult(cr)) {
      this->undo(pos);
      throw std::runtime_error("Could not get variation starting with " + move.uci());
      return std::make_pair(kMissingCacheResult, moves);
    }
    if (pos->turn_ == Color::BLACK) {
      originalCacheResult.eval *= -1;
    }
    originalCacheResult.eval = int64_t(originalCacheResult.eval) * 100 / this->evaluator.pawnValue();
    while (!isNullCacheResult(cr) && cr.bestMove != kNullMove && moves.size() < 10) {
      moves.push_back(cr.bestMove);
      this->make_move(pos, cr.bestMove);
      cr = cache.unsafe_find(pos->hash_);
    }
    for (size_t i = 0; i < moves.size(); ++i) {
      this->undo(pos);
    }
    return std::make_pair(originalCacheResult, moves);
  }

  void reset_stuff() {
    this->nodeCounter = 0;
    cache.clear();
    std::fill_n(historyHeuristicTable[Color::WHITE][0], 64 * 64, 0);
    std::fill_n(historyHeuristicTable[Color::BLACK][0], 64 * 64, 0);
  }

  SearchManager _manager;
  std::unique_ptr<StopThinkingCondition> stopThinkingCondition;
};

}  // ChessEngine

#endif  // THINKER_H