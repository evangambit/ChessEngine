#ifndef SEARCH_H
#define SEARCH_H

#include <cassert>
#include <cstdint>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <thread>
#include <memory>

#include "geometry.h"
#include "utils.h"
#include "Position.h"
#include "movegen.h"
#include "movegen/sliding.h"
#include "Evaluator.h"

#define COMPLEX_SEARCH 0
#define PARALLEL 0
#define USE_CACHE 1

namespace ChessEngine {

typedef int8_t Depth;

// PV-nodes ("principal variation" nodes) have a score that lies between alpha and beta; their scores are exact.

// Cut-nodes are nodes which contain a beta cutoff; score is a lower bound

// All-Nodes are nodes where no score exceeded alpha; score is an upper bound
enum NodeType {
  NodeTypeAll_UpperBound,
  NodeTypeCut_LowerBound,
  NodeTypePV,
  NodeTypeQC,
  NodeTypeQ,
};

enum SearchType {
  SearchTypeRoot,
  SearchTypeNormal,
  SearchTypeNullWindow,
};

const int32_t kMaxCachePriority = 65535;

struct CacheResult {  // 16 bytes
  uint64_t positionHash;  // 8 bytes
  Depth depthRemaining;   // 1 byte
  Evaluation eval;        // 2 bytes
  Move bestMove;          // 2 bytes
  NodeType nodeType;      // 1 byte
  uint16_t priority;      // 2 bytes; kMaxCachePriority if primary variation
  inline Evaluation lowerbound() const {
    return (nodeType == NodeTypeAll_UpperBound || nodeType == NodeTypeQ) ? kMinEval : eval;
  }
  inline Evaluation upperbound() const {
    return (nodeType == NodeTypeCut_LowerBound || nodeType == NodeTypeQ) ? kMaxEval : eval;
  }
};

struct SpinLock {
  std::atomic<bool> lock_ = {false};
  void lock() { while(lock_.exchange(true)); }
  void unlock() { lock_.store(false); }
};

// Transposition table is guaranteed to be a multiple of this.
constexpr int64_t kTranspositionTableFactor = 256;
const CacheResult kMissingCacheResult = CacheResult{
  0,
  -99,  // depth is -99 so this is always a useless result
  0, kNullMove, NodeTypePV, 0 // these values should never matter
};
inline bool isNullCacheResult(const CacheResult& cr) {
  return cr.depthRemaining == -1;
}

std::ostream& operator<<(std::ostream& stream, CacheResult cr) {
  if (isNullCacheResult(cr)) {
    return stream << "[NULL]" << std::endl;
  }
  return stream << "[ hash:" << cr.positionHash << " depth:" << uint16_t(cr.depthRemaining) << " eval:" << cr.eval << " move:" << cr.bestMove << " nodeType:" << unsigned(cr.nodeType) << " priority:" << unsigned(cr.priority) << " ]";
}

#if USE_CACHE
constexpr size_t kTranspositionTableMaxSteps = 3;
struct TranspositionTable {
  TranspositionTable(size_t kilobytes) {
    if (kilobytes < 1) {
      kilobytes = 1;
    }
    data = nullptr;
    this->set_cache_size(kilobytes);
  }
  TranspositionTable(const TranspositionTable&);  // Not implemented.
  TranspositionTable& operator=(const TranspositionTable& table);  // Not implemented.
  ~TranspositionTable() {
    delete[] data;
  }

  void set_cache_size(size_t kilobytes) {
    size = kilobytes * 1024 / sizeof(CacheResult);
    size = size / kTranspositionTableFactor;
    if (size == 0) {
      size = 1;
    }
    size *= kTranspositionTableFactor;
    delete[] data;
    data = new CacheResult[size];
    this->clear();
  }

  #if PARALLEL
  SpinLock spinLocks[kTranspositionTableFactor];
  #endif

  void insert(const CacheResult& cr) {
    size_t idx = cr.positionHash % size;
    const size_t delta = (cr.positionHash >> 32) % 16;

    for (size_t i = 0; i < kTranspositionTableMaxSteps; ++i) {
      #if PARALLEL
      spinLocks[idx % kTranspositionTableFactor].lock();
      #endif
      CacheResult *it = &data[idx];
      if (cr.positionHash == it->positionHash) {
        if (cr.depthRemaining > it->depthRemaining || (cr.nodeType == NodeTypePV && it->nodeType != NodeTypePV)) {
          *it = cr;
          #if PARALLEL
          spinLocks[idx % kTranspositionTableFactor].unlock();
          #endif
        }
        return;
      } else if (cr.depthRemaining > it->depthRemaining || (cr.depthRemaining == it->depthRemaining && cr.priority > it->priority)) {
        *it = cr;
        #if PARALLEL
        spinLocks[idx % kTranspositionTableFactor].unlock();
        #endif
        return;
      }
      #if PARALLEL
      spinLocks[idx % kTranspositionTableFactor].unlock();
      #endif
      idx = (idx + delta) % size;
    }
  }
  void clear() {
    std::fill_n((uint8_t *)data, sizeof(CacheResult) * size, 0);
  }
  CacheResult find(uint64_t hash) const {
    size_t idx = hash % size;
    const size_t delta = (hash >> 32) % 16;
    for (size_t i = 0; i < kTranspositionTableMaxSteps; ++i) {
      #if PARALLEL
      spinLocks[idx % kTranspositionTableFactor].lock();
      #endif
      CacheResult *cr = &data[idx];
      if (cr->priority != 0 && cr->positionHash == hash) {
        #if PARALLEL
        spinLocks[idx % kTranspositionTableFactor].unlock();
        #endif
        return *cr;
      }
      #if PARALLEL
      spinLocks[idx % kTranspositionTableFactor].unlock();
      #endif
      idx = (idx + delta) % size;
    }
    return kMissingCacheResult;
  }

  CacheResult create_cache_result(
    uint64_t hash,
    Depth depthRemaining,
    Evaluation eval,
    Move bestMove,
    NodeType nodeType,
    int32_t distFromPV) {
    return CacheResult{
      hash,
      depthRemaining,
      eval,
      bestMove,
      nodeType,
      uint8_t(std::max(0, kMaxCachePriority - distFromPV)),
    };
  }
 private:
  CacheResult *data;
  size_t size;
};
#else  // USE_CACHE
struct TranspositionTable {
  TranspositionTable(size_t kilobytes) {}
  TranspositionTable(const TranspositionTable&);  // Not implemented.
  TranspositionTable& operator=(const TranspositionTable& table);  // Not implemented.

  void set_cache_size(size_t kilobytes) {}

  void insert(const CacheResult& cr) {
  }
  void clear() {
  }
  CacheResult find(uint64_t hash) {
    return kMissingCacheResult;
  }
};
#endif  // USE_CACHE

constexpr int kQSimplePieceValues[7] = {
  // Note "NO_PIECE" has a score of 200 since this
  // encourages qsearch to value checks. (+0.02625)
  200, 100, 450, 500, 1000, 2000, 9999
};

constexpr Evaluation kMoveOrderPieceValues[7] = {
  0, 100,  300,  300,  500, 900,  900
};

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
struct SearchResult {
  SearchResult() : score(0), move(kNullMove) {}
  SearchResult(Evaluation score, Move move) : score(score), move(move), analysisComplete(true) {}
  SearchResult(Evaluation score, Move move, bool analysisComplete)
   : score(score), move(move), analysisComplete(analysisComplete) {}
  Evaluation score;
  Move move;
  bool analysisComplete;
};

template<Color PERSPECTIVE>
std::ostream& operator<<(std::ostream& stream, SearchResult<PERSPECTIVE> sr) {
  return stream << "(" << sr.move << " " << sr.score << ")" << std::endl;
}

// template<Color color>
// bool operator<(SearchResult<color> a, SearchResult<color> b) {
//   if (color == Color::WHITE) {
//     return a.score < b.score;
//   } else {
//     return a.score > b.score;
//   }
// }

std::ostream& operator<<(std::ostream& stream, SearchResult<Color::WHITE> sr) {
  stream << "(" << sr.score << ", " << sr.move << ")";
  return stream;
}

template<Color COLOR>
SearchResult<opposite_color<COLOR>()> flip(SearchResult<COLOR> r) {
  return SearchResult<opposite_color<COLOR>()>(r.score * -1, r.move);
}

template<Color COLOR>
SearchResult<Color::WHITE> to_white(SearchResult<COLOR> r);

template<>
SearchResult<Color::WHITE> to_white(SearchResult<Color::WHITE> r) {
  return r;
}

template<>
SearchResult<Color::WHITE> to_white(SearchResult<Color::BLACK> r) {
  return flip(r);
}

struct Thinker {
  size_t leafCounter;
  size_t nodeCounter;
  Evaluator evaluator;
  TranspositionTable cache;
  uint32_t historyHeuristicTable[Color::NUM_COLORS][64][64];
  size_t multiPV;

  PieceMaps pieceMaps;
  uint64_t lastRootHash;

  Thinker() : cache(10000), stopThinkingCondition(new NeverStopThinkingCondition()), lastRootHash(0) {
    reset_stuff();
    multiPV = 1;
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
    CacheResult originalCacheResult = this->cache.find(pos->hash_);
    CacheResult cr = originalCacheResult;
    if (isNullCacheResult(cr)) {
      this->undo(pos);
      throw std::runtime_error("Could not get variation starting with " + move.uci());
      return std::make_pair(kMissingCacheResult, moves);
    }
    if (pos->turn_ == Color::BLACK) {
      originalCacheResult.eval *= -1;
    }
    while (!isNullCacheResult(cr) && cr.bestMove != kNullMove && moves.size() < 10) {
      moves.push_back(cr.bestMove);
      this->make_move(pos, cr.bestMove);
      cr = cache.find(pos->hash_);
    }
    for (size_t i = 0; i < moves.size(); ++i) {
      this->undo(pos);
    }
    return std::make_pair(originalCacheResult, moves);
  }

  // TODO: qsearch can leave you in check
  template<Color TURN>
  SearchResult<TURN> qsearch(Position *pos, int32_t depth, Evaluation alpha, Evaluation beta) {
    ++this->nodeCounter;

    if (pos->is_draw()) {
      return SearchResult<TURN>(0, kNullMove);
    }

    if (std::popcount(pos->pieceBitboards_[coloredPiece<TURN, Piece::KING>()]) == 0) {
      return SearchResult<TURN>(kMissingKing, kNullMove);
    }

    const bool lookAtChecksToo = depth < 2;

    constexpr Color opposingColor = opposite_color<TURN>();
    constexpr ColoredPiece moverKing = coloredPiece<TURN, Piece::KING>();
    const bool inCheck = can_enemy_attack<TURN>(*pos, lsb(pos->pieceBitboards_[moverKing]));

    ExtMove moves[kMaxNumMoves];
    ExtMove *end;
    if (lookAtChecksToo) {
      end = compute_moves<TURN, MoveGenType::CHECKS_AND_CAPTURES>(*pos, moves);
    } else {
      end = compute_moves<TURN, MoveGenType::CAPTURES>(*pos, moves);
    }

    if (moves == end && inCheck) {
      return SearchResult<TURN>(kCheckmate, kNullMove);
    }

    // If we can stand pat for a beta cutoff, or if we have no moves, return.
    Threats<TURN> threats(*pos);
    SearchResult<TURN> r(this->evaluator.score<TURN>(*pos, threats), kNullMove);
    if (moves == end || r.score >= beta) {
      return r;
    }

    if (inCheck) {
      // Cannot stand pat if you're in check.
      r.score = kCheckmate;
    }

    for (ExtMove *move = moves; move < end; ++move) {
      move->score = kQSimplePieceValues[move->capture];
      move->score -= value_or_zero(
        ((threats.badForOur[move->piece] & bb(move->move.to)) > 0) && !((threats.badForOur[move->piece] & bb(move->move.from)) > 0),
        kQSimplePieceValues[move->piece]
      );
      move->score += (move->capture != Piece::NO_PIECE) * 1000;
    }

    std::sort(moves, end, [](ExtMove a, ExtMove b) {
      return a.score > b.score;
    });

    for (ExtMove *move = moves; move < end; ++move) {
      if (move->score < 0 && r.score > kLongestForcedMate) {
        break;
      }

      make_move<TURN>(pos, move->move);

      SearchResult<TURN> child = flip(qsearch<opposingColor>(pos, depth + 1, -beta, -alpha));
      child.score -= (child.score > -kLongestForcedMate);
      child.score += (child.score < kLongestForcedMate);

      undo<TURN>(pos);

      if (child.score > r.score) {
        r.score = child.score;
        r.move = move->move;
        if (r.score >= beta) {
          break;
        }
      }

      alpha = std::max(alpha, child.score);
    }

    return r;
  }

  #if PARALLEL
  struct SearchManager {
    uint8_t counters[32768];
    SpinLock locks[32];
    bool should_start_searching(uint64_t hash) {
      size_t idx = hash % 32768;
      SpinLock& lock = locks[hash % 32];
      lock.lock();
      bool r = counters[idx] == 0;
      if (r) {
        counters[idx] += 1;
      }
      lock.unlock();
      return r;
    }
    void start_searching(uint64_t hash) {
      size_t idx = hash % 32768;
      SpinLock& lock = locks[hash % 32];
      lock.lock();
      counters[idx] += 1;
      lock.unlock();
    }
    void finished_searching(uint64_t hash) {
      size_t idx = hash % 32768;
      SpinLock& lock = locks[hash % 32];
      lock.lock();
      counters[idx] -= 1;
      lock.unlock();
    }
  };
  #else
  struct SearchManager {
    bool should_start_searching(uint64_t hash) {
      return true;
    }
    void start_searching(uint64_t hash) {}
    void finished_searching(uint64_t hash) {}
  };
  #endif

  void reset_stuff() {
    this->leafCounter = 0;
    this->nodeCounter = 0;
    cache.clear();
    std::fill_n(historyHeuristicTable[Color::WHITE][0], 64 * 64, 0);
    std::fill_n(historyHeuristicTable[Color::BLACK][0], 64 * 64, 0);
  }

  template<Color TURN, SearchType SEARCH_TYPE>
  SearchResult<TURN> search(
    Position* pos,
    const Depth depthRemaining,
    const Depth plyFromRoot,
    Evaluation alpha, const Evaluation beta,
    RecommendedMoves recommendedMoves,
    uint16_t distFromPV,
    uint16_t threadID) {

    const Evaluation originalAlpha = alpha;
    const Evaluation originalBeta = beta;

    // alpha: a score we're guaranteed to get
    //  beta: a score our opponent is guaranteed to get
    //
    // if r.score >= beta
    //   we know our opponent will never let this position occur
    //
    // if r.score >= alpha
    //   we have just found a way to do better

    ++this->nodeCounter;

    constexpr Color opposingColor = opposite_color<TURN>();
    constexpr ColoredPiece moverKing = coloredPiece<TURN, Piece::KING>();

    if (depthRemaining >= 4 && this->stopThinkingCondition->should_stop_thinking(*this)) {
      return SearchResult<TURN>(0, kNullMove, false);
    }

    if (std::popcount(pos->pieceBitboards_[coloredPiece<TURN, Piece::KING>()]) == 0) {
      return SearchResult<TURN>(kMissingKing, kNullMove);
    }

    if (pos->is_draw() || this->evaluator.is_material_draw(*pos)) {
      return SearchResult<TURN>(Evaluation(0), kNullMove);
    }

    CacheResult cr = this->cache.find(pos->hash_);
    if (!isNullCacheResult(cr) && cr.depthRemaining >= depthRemaining) {
      if (cr.nodeType == NodeTypePV || cr.lowerbound() >= beta || cr.upperbound() <= alpha) {
        return SearchResult<TURN>(cr.eval, cr.bestMove);
      }
    }

    if (depthRemaining <= 0) {
      ++this->leafCounter;
      // Quiescence Search (0.4334 ± 0.0053)
      SearchResult<TURN> r = qsearch<TURN>(pos, 0, alpha, beta);

      NodeType nodeType = NodeTypePV;
      if (r.score >= beta) {
        nodeType = NodeTypeCut_LowerBound;
      } else if (r.score <= alpha) {
        nodeType = NodeTypeAll_UpperBound;
      }
      const CacheResult cr = this->cache.create_cache_result(
        pos->hash_,
        depthRemaining,
        r.score,
        r.move,
        nodeType,
        distFromPV
      );
      this->cache.insert(cr);
      return r;
    }

    // Futility pruning
    //  nodes/position, gain from futility pruning
    //   100, -0.0475 ± 0.0215
    //   300,  0.0325 ± 0.0292
    //  1000,  0.0800 ± 0.0266
    //  3000,  0.1325 ± 0.0302
    // 10000,  0.1630 ± 0.0274
    // 20000,  0.1761 ± 0.0256
    // 30000,  0.2292 ± 0.0305
    //
    // Note that not having *any* depth limit for futility pruning is terrible. For example, if
    // there is a line that loses a queen in one move but leads to forced mate in K ply, you won't
    // find the forced mate until you search to (roughly) a depth of
    // queenValue / futilityThreshold + K
    // This is really bad when you factor in the expoential relationship between depth and time.
    //
    // A simple solution is to just require the engine to search to a *least* a given depth (say 7)
    // when evaluating any position, but this seems hacky and we'd really like to have a strong
    // engine that works well at any depth (e.g. when we're doing self-play at a shallow depth), so
    // instead we increase the futility depth limit as a function of the total search depth.
    // Increasing by 1 every depth falls into the same problem, so instead we decrease by 0.5 every
    // depth. This should guarantee we find any mate-in-n-ply after searching 2*n ply.
    //
    // Also note that most people recommend giving a bonus when comparing against beta because we
    // should be able to find a quiet move that improves our score. In our opinion this is bad
    // because:
    // 1) A good evaluation function should account for a tempo bonus (e.g. we give bonuses for
    // hanging pieces)
    // 2) We're not just pruning with depth = 1, so it's not even clear if we have a tempo
    // advantage
    // 3) If we're using a score from the transposition table, the previous search already looked
    // at all of our moves.
    #if COMPLEX_SEARCH
    const int totalDepth = plyFromRoot + depthRemaining;
    const int kFutilityPruningDepthLimit = totalDepth / 2;
    const Evaluation futilityThreshold = 30;
    if (depthRemaining <= cr.depthRemaining + kFutilityPruningDepthLimit) {
      const int delta = futilityThreshold * (depthRemaining - cr.depthRemaining);
      if (cr.lowerbound() >= beta + delta || cr.upperbound() <= alpha - delta) {
        return SearchResult<TURN>(cr.eval, cr.bestMove);
      }
    }
    if (isNullCacheResult(cr) && depthRemaining <= kFutilityPruningDepthLimit) {
      SearchResult<TURN> r = qsearch<TURN>(pos, 0, alpha, beta);
      const int delta = futilityThreshold * depthRemaining;
      if (r.score >= beta + delta || r.score <= alpha - delta) {
        return r;
      }
    }
    #endif

    const bool inCheck = can_enemy_attack<TURN>(*pos, lsb(pos->pieceBitboards_[moverKing]));

    Move lastFoundBestMove = (isNullCacheResult(cr) ? kNullMove : cr.bestMove);

    SearchResult<TURN> r(
      kMinEval + 1,
      kNullMove
    );

    ExtMove moves[kMaxNumMoves];
    ExtMove *movesEnd = compute_moves<TURN, MoveGenType::ALL_MOVES>(*pos, moves);

    if (movesEnd - moves == 0) {
      if (inCheck) {
        return SearchResult<TURN>(kCheckmate, kNullMove);
      } else {
        return SearchResult<TURN>(Evaluation(0), kNullMove);
      }
    }

    // const Move lastMove = pos->history_.size() > 0 ? pos->history_.back().move : kNullMove;
    // TODO: use lastMove (above) to sort better.
    for (ExtMove *move = moves; move < movesEnd; ++move) {
      move->score = 0;

      // Bonus for capturing a piece.  (+0.136 ± 0.012)
      move->score += kMoveOrderPieceValues[move->capture];

      // Bonus if it was the last-found best move.  (0.048 ± 0.014)
      move->score += value_or_zero((move->move == lastFoundBestMove) && (depthRemaining == 1), 5000);
      move->score += value_or_zero((move->move == lastFoundBestMove) && (depthRemaining == 2), 5000);
      move->score += value_or_zero((move->move == lastFoundBestMove) && (depthRemaining >= 3), 5000);

      // Bonus if siblings like a move, though this seems statistically insignificant.
      move->score += value_or_zero(move->move == recommendedMoves.moves[0], 50);
      move->score += value_or_zero(move->move == recommendedMoves.moves[1], 50);

      // History Heuristic (+0.10)
      const int32_t history = this->historyHeuristicTable[TURN][move->move.from][move->move.to];
      move->score += value_or_zero(history > 0, 20);
      move->score += value_or_zero(history > 4, 20);
      move->score += value_or_zero(history > 16, 20);
      move->score += value_or_zero(history > 64, 20);
      move->score += value_or_zero(history > 256, 20);
    }

    std::sort(moves, movesEnd, [](ExtMove a, ExtMove b) {
      return a.score > b.score;
    });

    RecommendedMoves recommendationsForChildren;

    ExtMove deferredMoves[kMaxNumMoves];
    ExtMove *deferredMovesEnd = &deferredMoves[0];

    // Should be optimized away if SEARCH_TYPE != SearchTypeRoot.
    std::vector<SearchResult<TURN>> children;
    size_t numValidMoves = 0;
    for (int isDeferred = 0; isDeferred <= 1; ++isDeferred) {
      ExtMove *start = (isDeferred ? deferredMoves : &moves[0]);
      ExtMove *end = (isDeferred ? deferredMovesEnd : movesEnd);
      for (ExtMove *extMove = start; extMove < end; ++extMove) {

        make_move<TURN>(pos, extMove->move);

        // Don't move into check.
        if (can_enemy_attack<TURN>(*pos, lsb(pos->pieceBitboards_[moverKing]))) {
          undo<TURN>(pos);
          continue;
        }

        if (isDeferred == 0) {
          if (extMove == moves) {
            _manager.start_searching(pos->hash_);
          } else if (!_manager.should_start_searching(pos->hash_)) {
            *(deferredMovesEnd++) = *extMove;
            undo<TURN>(pos);
            continue;
          }
        } else {
          _manager.start_searching(pos->hash_);
        }

        ++numValidMoves;

        SearchResult<TURN> a = flip(search<opposingColor, SearchTypeNormal>(pos, depthRemaining - 1, plyFromRoot + 1, -beta, -alpha, recommendationsForChildren, distFromPV + (extMove != moves), threadID));
        a.score -= (a.score > -kLongestForcedMate);
        a.score += (a.score < kLongestForcedMate);

        _manager.finished_searching(pos->hash_);
        undo<TURN>(pos);

        // We don't bother to break here, since all of our children will automatically return a low-cost,
        // good-faith estimate (unless our depth is 4, in which case this will be true for the depth above
        // us).
        r.analysisComplete &= a.analysisComplete;

        if (SEARCH_TYPE == SearchTypeRoot) {
          a.move = extMove->move;
          children.push_back(a);
          std::sort(
            children.begin(),
            children.end(),
            [](SearchResult<TURN> a, SearchResult<TURN> b) -> bool {
              return a.score > b.score;
          });
          if (children.size() > multiPV) {
            children.pop_back();
          }
          if (a.score > r.score) {
            r.score = a.score;
            r.move = extMove->move;
            recommendationsForChildren.add(a.move);
          }
          if (children.size() >= multiPV) {
            alpha = std::max(alpha, children.back().score);
          }
        } else {
          if (a.score > r.score) {
            r.score = a.score;
            r.move = extMove->move;
            recommendationsForChildren.add(a.move);
            if (r.score >= beta) {
              // NOTE: Unlike other engines, we include captures in our history heuristic, as this
              // orders captures that are materially equal.
              // TODO: make this thread safe.
              this->historyHeuristicTable[TURN][r.move.from][r.move.to] += depthRemaining * depthRemaining;
              break;
            }
            if (r.score > alpha) {
              alpha = r.score;
            }
          }
        }
      }
    }

    if (numValidMoves == 0) {
      r.score = inCheck ? kCheckmate : 0;
      r.move = kNullMove;
    }

    r.analysisComplete = !this->stopThinkingCondition->should_stop_thinking(*this);

    if (r.analysisComplete) {
      NodeType nodeType = NodeTypePV;
      if (r.score >= originalBeta) {
        nodeType = NodeTypeCut_LowerBound;
      } else if (r.score <= originalAlpha) {
        nodeType = NodeTypeAll_UpperBound;
      }
      const CacheResult cr = this->cache.create_cache_result(
        pos->hash_,
        depthRemaining,
        r.score,
        r.move,
        nodeType,
        distFromPV
      );
      this->cache.insert(cr);
    }

    return r;
  }

  template<Color TURN>
  static SearchResult<TURN> _search_with_aspiration_window(Thinker* thinker, Position* pos, Depth depth, SearchResult<TURN> lastResult, uint16_t threadID) {
    Position copy(*pos);
    // It's important to call this at the beginning of a search, since if we're sharing Position (e.g. selfplay.cpp) we
    // need to recompute piece map scores using our own weights.
    copy.set_piece_maps(thinker->pieceMaps);
    // TODO: the aspiration window technique used here should probably be implemented for internal nodes too.
    // Even just using this at the root node gives my engine a +0.25 (n=100) score against itself.
    // Table of historical experiments (program with window vs program without)
    // 100: 0.099 ± 0.021
    //  75: 0.152 ± 0.021
    //  50: 0.105 ± 0.019
    #if COMPLEX_SEARCH
    constexpr Evaluation kBuffer = 75;
    SearchResult<TURN> r = thinker->search<TURN, SearchTypeRoot>(&copy, depth, 0, lastResult.score - kBuffer, lastResult.score + kBuffer, RecommendedMoves(), 0, threadID);
    if (r.score > lastResult.score - kBuffer && r.score < lastResult.score + kBuffer) {
      return r;
    }
    #endif
    return thinker->search<TURN, SearchTypeRoot>(&copy, depth, 0, kMinEval, kMaxEval, RecommendedMoves(), 0, threadID);
  }

  // TODO: making threads work with multiPV seems really nontrivial.

  // Gives scores from white's perspective
  SearchResult<Color::WHITE> search(Position* pos, size_t depthLimit, std::function<void(Position *, size_t, double)> callback) {
    time_t tstart = clock();

    this->nodeCounter = 0;
    this->leafCounter = 0;
    stopThinkingCondition->start_thinking(*this);

    SearchResult<Color::WHITE> results(Evaluation(0), kNullMove);
    for (size_t depth = 1; depth <= depthLimit; ++depth) {
      SearchResult<Color::WHITE> r = this->_search(pos, Depth(depth), results);
      if (r.analysisComplete) {
        results = r;
      }
      const double secs = double(clock() - tstart)/CLOCKS_PER_SEC;
      callback(pos, depth, secs);
      if (this->stopThinkingCondition->should_stop_thinking(*this)) {
        break;
      }
    }

    return results;
  }

  SearchResult<Color::WHITE> _search(Position* pos, Depth depth, SearchResult<Color::WHITE> lastResult) {
    if (pos->turn_ == Color::WHITE) {
      std::thread t1(
        Thinker::_search_with_aspiration_window<Color::WHITE>,
        this,
        pos,
        depth,
        lastResult,
        0
      );
      t1.join();
    } else {
      std::thread t1(
        Thinker::_search_with_aspiration_window<Color::BLACK>,
        this,
        pos,
        depth,
        flip(lastResult),
        0
      );
      t1.join();
    }

    CacheResult cr = this->cache.find(pos->hash_);
    if (isNullCacheResult(cr)) {
      throw std::runtime_error("Null result from search");
    }
    if (pos->turn_ == Color::WHITE) {
      return SearchResult<Color::WHITE>(cr.eval, cr.bestMove);
    } else {
      return flip(SearchResult<Color::BLACK>(cr.eval, cr.bestMove));
    }
  }

  SearchManager _manager;
  std::unique_ptr<StopThinkingCondition> stopThinkingCondition;

};

struct StopThinkingNodeCountCondition : public StopThinkingCondition {
  StopThinkingNodeCountCondition(size_t numNodes)
  : numNodes(numNodes) {}
  void start_thinking(const Thinker& thinker) {
    offset = thinker.nodeCounter;
  }
  bool should_stop_thinking(const Thinker& thinker) {
    assert(thinker.nodeCounter >= offset);
    return thinker.nodeCounter - offset > this->numNodes;
  }
  size_t offset;
  size_t numNodes;
};

struct StopThinkingTimeCondition : public StopThinkingCondition {
  StopThinkingTimeCondition(uint64_t milliseconds) : milliseconds(milliseconds) {}
  void start_thinking(const Thinker& thinker) {
    startTime = this->current_time();
  }
  uint64_t current_time() const {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
  }
  bool should_stop_thinking(const Thinker& thinker) {
    uint64_t dt = this->current_time() - startTime;
    return dt > milliseconds;
  }
  uint64_t startTime;
  uint64_t milliseconds;
};

struct OrStopCondition : public StopThinkingCondition {
  OrStopCondition(StopThinkingCondition *a, StopThinkingCondition *b) : a(a), b(b) {}
  void start_thinking(const Thinker& thinker) {
    a->start_thinking(thinker);
    b->start_thinking(thinker);
  }
  bool should_stop_thinking(const Thinker& thinker) {
    return a->should_stop_thinking(thinker) || b->should_stop_thinking(thinker);
  }
  std::unique_ptr<StopThinkingCondition> a, b;
};


// TODO: there is a bug where
// "./a.out fen 8/8/8/1k6/3P4/8/8/3K4 w - - 0 1 depth 17"
// claims white is winning.

}  // namespace ChessEngine

#endif  // SEARCH_H