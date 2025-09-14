#ifndef THINKER_INTERFACE_H
#define THINKER_INTERFACE_H

#include <memory>
#include <istream>

#include "StopThinkingCondition.h"

namespace ChessEngine {

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

struct Position;
struct Move;
struct Evaluator;
struct PieceMaps;
struct ThinkerInterface {
  virtual void set_cache_size(size_t kilobytes) = 0;
  virtual void set_multi_pv(size_t multiPV) = 0;
  virtual void set_num_threads(size_t numThreads) = 0;
  virtual void set_move_overhead_ms(int64_t moveOverheadMs) = 0;
  virtual void set_stop_thinking_condition(std::shared_ptr<StopThinkingCondition> condition) = 0;
#if NNUE_EVAL
  virtual void load_nnue(const std::string& filename) = 0;
  virtual std::shared_ptr<NnueNetworkInterface> get_nnue() = 0;
#endif

  virtual void load_weights(std::istream& myfile) = 0;
  virtual void save_weights(std::ostream& myfile) = 0;
  virtual Evaluator& get_evaluator() = 0;
  virtual PieceMaps& get_piece_maps() = 0;
  virtual void clear_tt() = 0;
  virtual void clear_history_heuristic() = 0;

  virtual uint64_t get_node_count() const = 0;
  virtual size_t get_num_threads() const = 0;
  virtual size_t get_multi_pv() const = 0;
  virtual size_t get_cache_size_kb() const = 0;

  virtual CacheResult probe_tt(uint64_t hash) = 0;

  virtual const std::vector<VariationHead<Color::WHITE>>& get_variations() = 0;

  virtual std::pair<Evaluation, std::vector<Move>> get_variation(Position *pos, Move move) = 0;
};

}  // namespace ChessEngine

#endif