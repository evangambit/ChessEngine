#ifndef THINKER_H
#define THINKER_H

#include "utils.h"
#include "Evaluator.h"
#include "TranspositionTable.h"
#include "SearchResult.h"
#include "Position.h"
#include "StopThinkingCondition.h"
#include "ThinkerInterface.h"

namespace ChessEngine {

struct RecommendedMoves {
  Move moves[2];
  RecommendedMoves() {
    std::fill_n(moves, 2, kNullMove);
  }
  inline void add(Move move) {
    if (move != moves[0]) {
      moves[1] = moves[0];
      moves[0] = move;
    }
  }
};

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

struct Search;
struct Thinker : public ThinkerInterface {
  friend Search;

  Thinker() : cache(10000), stopThinkingCondition(new NeverStopThinkingCondition()), lastRootHash(0), moveOverheadMs(0) {
    this->clear_history_heuristic();
    this->nodeCounter = 0;
    #if NNUE_EVAL
    this->nnue = std::make_shared<NnueNetwork>();
    #endif
    multiPV = 1;
    numThreads = 1;
  }

  RecommendedMoves killerMoves[127];

private:

  size_t nodeCounter;
  TranspositionTable cache;
  uint32_t historyHeuristicTable[Color::NUM_COLORS][Piece::NUM_PIECES][64][64];
  size_t multiPV;
  size_t numThreads;

  // UCI options.
  int64_t moveOverheadMs;

  uint64_t lastRootHash;

  std::vector<VariationHead<Color::WHITE>> variations;

  SpinLock stopThinkingLock;

  void set_multi_pv(size_t multiPV) override {
    this->multiPV = multiPV;
  }

  void set_num_threads(size_t numThreads) override {
    this->numThreads = numThreads;
  }

  void set_move_overhead_ms(int64_t moveOverheadMs) override {
    this->moveOverheadMs = moveOverheadMs;
  }

  void set_stop_thinking_condition(std::shared_ptr<StopThinkingCondition> condition) override {
    stopThinkingLock.lock();
    this->stopThinkingCondition = condition;
    stopThinkingLock.unlock();
  }


  uint64_t get_node_count() const override {
    return nodeCounter;
  }

  size_t get_num_threads() const override {
    return numThreads;
  }
  size_t get_multi_pv() const override {
    return multiPV;
  }

  #if NNUE_EVAL
  std::shared_ptr<NnueNetwork> get_nnue() override {
    return nnue;
  }
  #endif

  size_t get_cache_size_kb() const override {
    return cache.kb_size();
  }

  const std::vector<VariationHead<Color::WHITE>>& get_variations() override {
    return variations;
  }

  void set_cache_size(size_t kilobytes) override {
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
  std::shared_ptr<NnueNetwork> nnue;
  #endif

  Evaluator evaluator;
  PieceMaps pieceMaps;

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
  void load_nnue(std::istream& myfile) override {
    this->nnue->load(myfile);
  }
  #endif

  Evaluator& get_evaluator() override {
    return evaluator;
  }

  PieceMaps& get_piece_maps() override {
    return pieceMaps;
  }
  void load_weights(std::istream& myfile) override {
    this->evaluator.load_weights_from_file(myfile);
    this->pieceMaps.load_weights_from_file(myfile);
  }
  void save_weights(std::ostream& myfile) override {
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

  CacheResult probe_tt(uint64_t hash) override {
    return cache.find<false>(hash);
  }

  std::pair<Evaluation, std::vector<Move>> get_variation(Position *pos, Move move) override {
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

  void clear_tt() override {
    cache.clear();
    // If you're clearing the tranposition table you probably want to clear the history heuristi too.
    this->clear_history_heuristic();
  }

  void clear_history_heuristic() override {
    std::fill_n(historyHeuristicTable[Color::WHITE][0][0], Piece::NUM_PIECES * 64 * 64, 0);
    std::fill_n(historyHeuristicTable[Color::BLACK][0][0], Piece::NUM_PIECES * 64 * 64, 0);
  }

  SearchManager _manager;
  std::shared_ptr<StopThinkingCondition> stopThinkingCondition;
};

}  // ChessEngine

#endif  // THINKER_H
