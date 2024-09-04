#ifndef THINKER_H
#define THINKER_H

#include "utils.h"
#include "Evaluator.h"
#include "TranspositionTable.h"
#include "SearchResult.h"
#include "Position.h"
// #include "nnue.h"

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

template<Color PERSPECTIVE>
struct VariationHead {
  VariationHead(Evaluation score, Move move, Move response)
  : score(score), move(move), response(response) {}
  SearchResult<PERSPECTIVE> to_search_result() const {
    return SearchResult<PERSPECTIVE>(score, move);
  }
  Evaluation score;
  Move move;
  Move response;
};

template<Color PERSPECTIVE>
VariationHead<Color::WHITE> to_white(const VariationHead<PERSPECTIVE> vh) {
  return vh;
}

template<>
VariationHead<Color::WHITE> to_white(const VariationHead<Color::BLACK> vh) {
  return VariationHead<Color::WHITE>(-vh.score, vh.move, vh.response);
}

template<Color COLOR>
std::ostream& operator<<(std::ostream& os, const VariationHead<COLOR>& vh) {
  os << vh.move.uci() << " " << vh.response.uci() << " " << eval2str(vh.score);
  return os;
}

struct Thinker {
  size_t nodeCounter;
  Evaluator evaluator;
  TranspositionTable cache;
  uint32_t historyHeuristicTable[Color::NUM_COLORS][Piece::NUM_PIECES][64][64];
  size_t multiPV;
  size_t numThreads;

  // UCI options.
  int64_t moveOverheadMs;

  PieceMaps pieceMaps;
  #if NNUE_EVAL
  std::shared_ptr<NnueNetwork> nnue;
  #endif
  uint64_t lastRootHash;

  std::vector<VariationHead<Color::WHITE>> variations;

  SpinLock stopThinkingLock;

  Thinker() : cache(10000), stopThinkingCondition(new NeverStopThinkingCondition()), lastRootHash(0), moveOverheadMs(0) {
    this->clear_history_heuristic();
    this->nodeCounter = 0;
    #if NNUE_EVAL
    this->nnue = std::make_shared<NnueNetwork>();
    #endif
    multiPV = 1;
    numThreads = 1;
  }

  void set_cache_size(size_t kilobytes) {
    cache.set_cache_size(kilobytes);
  }

  void _make_move(Position* pos, Move move) const {
    if (pos->turn_ == Color::WHITE) {
      make_move<Color::WHITE>(pos, move);
    } else {
      make_move<Color::BLACK>(pos, move);
    }
  }

  void _undo(Position* pos) const {
    if (pos->turn_ == Color::BLACK) {
      undo<Color::WHITE>(pos);
    } else {
      undo<Color::BLACK>(pos);
    }
  }

  #if NNUE_EVAL
  void load_nnue(const std::string& filename) {
    std::ifstream myfile;
    myfile.open(filename);
    if (!myfile.is_open()) {
      std::cout << "Error opening file \"" << filename << "\"" << std::endl;
      exit(0);
    }
    this->load_nnue(myfile);
    myfile.close();
  }
  void load_nnue(std::istream& myfile) {
    this->nnue->load(myfile);
  }
  #else
  void load_weights(std::istream& myfile) {
    this->evaluator.load_weights_from_file(myfile);
    this->pieceMaps.load_weights_from_file(myfile);
  }
  void save_weights(std::ostream& myfile) {
    this->evaluator.save_weights_to_file(myfile);
    this->pieceMaps.save_weights_to_file(myfile);
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
    this->load_weights(myfile);
    myfile.close();
  }
  #endif

  std::pair<Evaluation, std::vector<Move>> get_variation(Position *pos, Move move) const {
    VariationHead<Color::WHITE> const * vh = nullptr;
    for (const VariationHead<Color::WHITE>& v : this->variations) {
      if (v.move == move) {
        vh = &v;
        break;
      }
    }

    if (vh == nullptr) {
      throw std::runtime_error("Could not get variation starting with " + move.uci());
    }

    std::vector<Move> moves;

    if (vh->move != kNullMove) {
      this->_make_move(pos, vh->move);
      moves.push_back(vh->move);
    }

    if (vh->response != kNullMove) {
      this->_make_move(pos, vh->response);
      moves.push_back(vh->response);
    }

    CacheResult cr = this->cache.unsafe_find(pos->hash_);
    while (!isNullCacheResult(cr) && cr.bestMove != kNullMove && moves.size() < 10) {
      moves.push_back(cr.bestMove);
      this->_make_move(pos, cr.bestMove);
      cr = cache.unsafe_find(pos->hash_);
    }

    for (size_t i = 0; i < moves.size(); ++i) {
      this->_undo(pos);
    }

    return std::make_pair(vh->score, moves);
  }

  void clear_tt() {
    cache.clear();
    // If you're clearing the tranposition table you probably want to clear the history heuristi too.
    this->clear_history_heuristic();
  }

  void clear_history_heuristic() {
    std::fill_n(historyHeuristicTable[Color::WHITE][0][0], Piece::NUM_PIECES * 64 * 64, 0);
    std::fill_n(historyHeuristicTable[Color::BLACK][0][0], Piece::NUM_PIECES * 64 * 64, 0);
  }

  SearchManager _manager;
  std::unique_ptr<StopThinkingCondition> stopThinkingCondition;
};

}  // ChessEngine

#endif  // THINKER_H
