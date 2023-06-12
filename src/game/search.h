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

namespace ChessEngine {

void for_all_moves(Position *position, std::function<void(const Position&, ExtMove)> f) {
  ExtMove moves[kMaxNumMoves];
  ExtMove *end;
  if (position->turn_ == Color::WHITE) {
    end = compute_legal_moves<Color::WHITE>(position, moves);
  } else {
    end = compute_legal_moves<Color::BLACK>(position, moves);
  }
  for (ExtMove *move = moves; move < end; ++move) {
    if (position->turn_ == Color::WHITE) {
      make_move<Color::WHITE>(position, move->move);
    } else {
      make_move<Color::BLACK>(position, move->move);
    }
    f(*position, *move);
    if (position->turn_ == Color::WHITE) {
      undo<Color::BLACK>(position);
    } else {
      undo<Color::WHITE>(position);
    }
  }
}

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
  SearchTypeExtended,
};

const int32_t kMaxCachePriority = 16383;
const int32_t kNumRootCounters = 8;

struct CacheResult {  // 16 bytes
  uint64_t positionHash;  // 8 bytes
  Depth depthRemaining;   // 1 byte
  Evaluation eval;        // 2 bytes
  Move bestMove;          // 2 bytes
  NodeType nodeType;      // 1 byte
  unsigned priority : 14;      // 2 bytes; kMaxCachePriority if primary variation
  unsigned rootCounter : 3;
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

// TODO: it's possible for the same position to occur multiple times in this table.
// 1) Insert P1
// 2) Insert P2, which collides with P1, so it goes to it's second spot
// 3) make a move
// 4) Insert P2; it's first spot is free so it is inserted
//
// The saving grace is that it's second spot is still marked as obsolete, so it should be cleared eventually,
// but this kind of behavior seems really scary. I should a *least* write some tests for the transposition table,
// and possible refactor how it handles duplicates.

constexpr size_t kTranspositionTableMaxSteps = 3;
struct TranspositionTable {
  TranspositionTable(size_t kilobytes) : rootCounter(0), currentRootHash(0) {
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

  void starting_search(uint64_t rootHash) {
    if (rootHash != this->currentRootHash) {
      this->currentRootHash = rootHash;
      this->rootCounter = (this->rootCounter + 1) % kNumRootCounters;
    }
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

  SpinLock spinLocks[kTranspositionTableFactor];

  template<bool IS_PARALLEL>
  void insert(const CacheResult& cr) {
    size_t idx = cr.positionHash % size;
    const size_t delta = (cr.positionHash >> 32) % 16;

    for (size_t i = 0; i < kTranspositionTableMaxSteps; ++i) {
      if (IS_PARALLEL) {
        spinLocks[idx % kTranspositionTableFactor].lock();
      }
      CacheResult *it = &data[idx];
      if (cr.positionHash == it->positionHash) {
        if (cr.depthRemaining > it->depthRemaining || (cr.nodeType == NodeTypePV && it->nodeType != NodeTypePV)) {
          *it = cr;
        } else {
          // Mark this entry as "fresh".
          it->rootCounter = this->rootCounter;
        }
        if (IS_PARALLEL) {
          spinLocks[idx % kTranspositionTableFactor].unlock();
        }
        return;
      } else if (
        // We add a penalty to entries with a different root so that we slowly remove stale entries
        // from the table.
        // TODO: maybe penalty should depend on distance from rootCounter to more aggressively
        // remove very old entries. For example:
        // (this->rootCounter - it->rootCounter + kNumRootCounters) % kNumRootCounters
        cr.depthRemaining > it->depthRemaining - (it->rootCounter != this->rootCounter)
        ||
        (cr.depthRemaining == it->depthRemaining && cr.priority > it->priority)) {
        *it = cr;
        if (IS_PARALLEL) {
          spinLocks[idx % kTranspositionTableFactor].unlock();
        }
        return;
      }
      if (IS_PARALLEL) {
        spinLocks[idx % kTranspositionTableFactor].unlock();
      }
      idx = (idx + delta) % size;
    }
  }
  void clear() {
    std::fill_n((uint8_t *)data, sizeof(CacheResult) * size, 0);
  }
  template<bool IS_PARALLEL>
  CacheResult find(uint64_t hash) {
    size_t idx = hash % size;
    const size_t delta = (hash >> 32) % 16;
    for (size_t i = 0; i < kTranspositionTableMaxSteps; ++i) {
      if (IS_PARALLEL) {
        spinLocks[idx % kTranspositionTableFactor].lock();
      }
      CacheResult *cr = &data[idx];
      if (cr->priority != 0 && cr->positionHash == hash) {
        if (IS_PARALLEL) {
          spinLocks[idx % kTranspositionTableFactor].unlock();
        }
        return *cr;
      }
      if (IS_PARALLEL) {
        spinLocks[idx % kTranspositionTableFactor].unlock();
      }
      idx = (idx + delta) % size;
    }
    return kMissingCacheResult;
  }
  CacheResult unsafe_find(uint64_t hash) const {
    size_t idx = hash % size;
    const size_t delta = (hash >> 32) % 16;
    for (size_t i = 0; i < kTranspositionTableMaxSteps; ++i) {
      CacheResult *cr = &data[idx];
      if (cr->priority != 0 && cr->positionHash == hash) {
        return *cr;
      }
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
      this->rootCounter,
    };
  }
 private:
  uint64_t currentRootHash;
  unsigned rootCounter;
  CacheResult *data;
  size_t size;
};

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
  size_t numThreads;

  PieceMaps pieceMaps;
  uint64_t lastRootHash;

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

  struct Thread {
    Thread(uint64_t id, const Position& pos, const Evaluator& e)
    : id(id), pos(pos), evaluator(e), nodeCounter(0), leafCounter(0) {}
    uint64_t id;
    Position pos;
    Evaluator evaluator;
    uint64_t nodeCounter, leafCounter;
  };

  // TODO: qsearch can leave you in check
  template<Color TURN>
  static SearchResult<TURN> qsearch(Thinker *thinker, Thread *thread, int32_t depth, int32_t plyFromRoot, Evaluation alpha, Evaluation beta) {
    ++thread->nodeCounter;

    if (std::popcount(thread->pos.pieceBitboards_[coloredPiece<TURN, Piece::KING>()]) == 0) {
      return SearchResult<TURN>(kMissingKing, kNullMove);
    }

    const bool lookAtChecksToo = depth < 2;

    constexpr Color opposingColor = opposite_color<TURN>();
    constexpr ColoredPiece moverKing = coloredPiece<TURN, Piece::KING>();
    const bool inCheck = can_enemy_attack<TURN>(thread->pos, lsb(thread->pos.pieceBitboards_[moverKing]));

    ExtMove moves[kMaxNumMoves];
    ExtMove *end;
    if (lookAtChecksToo) {
      end = compute_moves<TURN, MoveGenType::CHECKS_AND_CAPTURES>(thread->pos, moves);
    } else {
      end = compute_moves<TURN, MoveGenType::CAPTURES>(thread->pos, moves);
    }

    if (moves == end && inCheck) {
      return SearchResult<TURN>(kQCheckmate + plyFromRoot, kNullMove);
    }

    // If we can stand pat for a beta cutoff, or if we have no moves, return.
    Threats<TURN> threats(thread->pos);
    SearchResult<TURN> r(thread->evaluator.score<TURN>(thread->pos, threats), kNullMove);
    {
      // Add a penalty to standing pat if we have hanging pieces.
      // (+0.0444 ± 0.0077) after 1024 games at 50,000 nodes/move
      // Note: k=200 is worse.
      Threats<opposingColor> enemyThreats(thread->pos);
      constexpr int k = 50;
      r.score -= value_or_zero((enemyThreats.badForTheir[Piece::PAWN] & thread->pos.pieceBitboards_[coloredPiece<TURN>(Piece::PAWN)]) > 0, k);
      r.score -= value_or_zero((enemyThreats.badForTheir[Piece::KNIGHT] & thread->pos.pieceBitboards_[coloredPiece<TURN>(Piece::KNIGHT)]) > 0, k);
      r.score -= value_or_zero((enemyThreats.badForTheir[Piece::BISHOP] & thread->pos.pieceBitboards_[coloredPiece<TURN>(Piece::BISHOP)]) > 0, k);
      r.score -= value_or_zero((enemyThreats.badForTheir[Piece::ROOK] & thread->pos.pieceBitboards_[coloredPiece<TURN>(Piece::ROOK)]) > 0, k);
      r.score -= value_or_zero((enemyThreats.badForTheir[Piece::QUEEN] & thread->pos.pieceBitboards_[coloredPiece<TURN>(Piece::QUEEN)]) > 0, k);
    }
    if (moves == end || r.score >= beta) {
      return r;
    }

    // info depth 8 multipv 0 score cp 164 nodes 2629635 nps 3349853 time  785 pv g1f3 g8f6 d2d4 d7d5 b1c3 b8c6 c1f4 c8f5
    // info depth 9 multipv 0 score cp 73 nodes 19448426 nps 3338785 time 5825 pv e2e4 e7e5 g1f3 b8c6 b1c3 g8f6
    // bestmove e2e4 ponder e7e5
    // info depth 8 multipv 0 score cp 164 nodes 2215797 nps 2193858 time 1010 pv d2d4 d7d5 b1c3 b8c6 g1f3 g8f6 c1f4 c8f5
    // info depth 9 multipv 0 score cp 83 nodes 14812146 nps 2310787 time 6410 pv e2e4 e7e5 g1f3 b8c6 b1c3 g8f6
    // bestmove e2e4 ponder e7e5



    if (inCheck) {
      // Cannot stand pat if you're in check.
      r.score = kQLongestForcedMate;
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
      if (move->score < 0 && r.score > kQLongestForcedMate) {
        break;
      }

      make_move<TURN>(&thread->pos, move->move);

      SearchResult<TURN> child = flip(Thinker::qsearch<opposingColor>(thinker, thread, depth + 1, plyFromRoot + 1, -beta, -alpha));
      child.score -= (child.score > -kQLongestForcedMate);
      child.score += (child.score <  kQLongestForcedMate);

      undo<TURN>(&thread->pos);

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

  static constexpr unsigned kNumSearchManagerCounters = 32768;
  static constexpr unsigned kNumSearchManagerLocks = 32;
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

  void reset_stuff() {
    this->leafCounter = 0;
    this->nodeCounter = 0;
    cache.clear();
    std::fill_n(historyHeuristicTable[Color::WHITE][0], 64 * 64, 0);
    std::fill_n(historyHeuristicTable[Color::BLACK][0], 64 * 64, 0);
  }

  template<Color TURN, SearchType SEARCH_TYPE, bool IS_PARALLEL>
  static SearchResult<TURN> search(
    Thinker *thinker,
    Thread *thread,
    const Depth depthRemaining,
    const Depth plyFromRoot,
    Evaluation alpha, const Evaluation beta,
    RecommendedMoves recommendedMoves,
    uint16_t distFromPV) {


    const Evaluation originalAlpha = alpha;
    const Evaluation originalBeta = beta;

    // alpha: a score we're guaranteed to get
    //  beta: a score our opponent is guaranteed to get
    //
    // if r.score >= beta
    //   we know our opponent will never let thinker position occur
    //
    // if r.score >= alpha
    //   we have just found a way to do better

    ++thread->nodeCounter;
    if (thread->id == 0) {
      ++thinker->nodeCounter;
    }

    constexpr Color opposingColor = opposite_color<TURN>();
    constexpr ColoredPiece moverKing = coloredPiece<TURN, Piece::KING>();

    if (depthRemaining >= 4 && thinker->stopThinkingCondition->should_stop_thinking(*thinker)) {
      return SearchResult<TURN>(0, kNullMove, false);
    }

    if (std::popcount(thread->pos.pieceBitboards_[coloredPiece<TURN, Piece::KING>()]) == 0) {
      return SearchResult<TURN>(kMissingKing, kNullMove);
    }

    if (thread->pos.is_draw(plyFromRoot)) {
      return SearchResult<TURN>(Evaluation(0), kNullMove);
    }
    if (SEARCH_TYPE != SearchTypeRoot && thread->evaluator.is_material_draw(thread->pos)) {
      return SearchResult<TURN>(Evaluation(0), kNullMove);
    }

    CacheResult cr = thinker->cache.find<IS_PARALLEL>(thread->pos.hash_);
    // Short-circuiting due to a cached result.
    // (+0.0254 ± 0.0148) after 256 games at 50,000 nodes/move
    if (!isNullCacheResult(cr) && cr.depthRemaining >= depthRemaining) {
      if (cr.nodeType == NodeTypePV || cr.lowerbound() >= beta || cr.upperbound() <= alpha) {
        return SearchResult<TURN>(cr.eval, cr.bestMove);
      }
    }

    if (depthRemaining <= 0) {
      if (thread->id == 0) {
        ++thinker->leafCounter;
      }
      ++thread->leafCounter;
      // Quiescence Search
      // (+0.4453 ± 0.0072) after 256 games at 50,000 nodes/move
      SearchResult<TURN> r = Thinker::qsearch<TURN>(thinker, thread, 0, plyFromRoot, alpha, beta);

      // Extensions
      // (0.0413 ± 0.0081) after 1024 games at 50,000 nodes/move
      if (SEARCH_TYPE == SearchTypeNormal) {
        if (r.score >= alpha && r.score <= beta) {
          r = search<TURN, SearchTypeExtended, IS_PARALLEL>(
            thinker,           // thinker
            thread,
            2,                 // depthRemaining
            plyFromRoot,       // plyFromRoot
            alpha,             // alpha
            beta,              // beta
            recommendedMoves,  // recommendedMoves
            distFromPV         // distFromPV
          );
        }
      }

      NodeType nodeType = NodeTypePV;
      if (r.score >= beta) {
        nodeType = NodeTypeCut_LowerBound;
      } else if (r.score <= alpha) {
        nodeType = NodeTypeAll_UpperBound;
      }
      const CacheResult cr = thinker->cache.create_cache_result(
        thread->pos.hash_,
        depthRemaining,
        r.score,
        r.move,
        nodeType,
        distFromPV
      );
      thinker->cache.insert<IS_PARALLEL>(cr);
      return r;
    }

    // Futility pruning
    //  nodes/position, gain from futility pruning
    // 100_000, 0.0703 ± 0.0218
    //
    // Note that not having *any* depth limit for futility pruning is terrible. For example, if
    // there is a line that loses a queen in one move but leads to forced mate in K ply, you won't
    // find the forced mate until you search to (roughly) a depth of
    // queenValue / futilityThreshold + K
    // thinker is really bad when you factor in the expoential relationship between depth and time.
    //
    // A simple solution is to just require the engine to search to a *least* a given depth (say 7)
    // when evaluating any position, but thinker seems hacky and we'd really like to have a strong
    // engine that works well at any depth (e.g. when we're doing self-play at a shallow depth), so
    // instead we increase the futility depth limit as a function of the total search depth.
    // Increasing by 1 every depth falls into the same problem, so instead we decrease by 0.5 every
    // depth. thinker should guarantee we find any mate-in-n-ply after searching 2*n ply.
    //
    // Also note that most people recommend giving a bonus when comparing against beta because we
    // should be able to find a quiet move that improves our score. In our opinion thinker is bad
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
    const int32_t futilityThreshold = 100;
    if (depthRemaining <= cr.depthRemaining + kFutilityPruningDepthLimit) {
      const int delta = futilityThreshold * (depthRemaining - cr.depthRemaining);
      if (cr.lowerbound() >= beta + delta || cr.upperbound() <= alpha - delta) {
        return SearchResult<TURN>(cr.eval, cr.bestMove);
      }
    }
    if (isNullCacheResult(cr) && depthRemaining <= kFutilityPruningDepthLimit) {
      SearchResult<TURN> r = Thinker::qsearch<TURN>(thinker, pos, 0, 0, alpha, beta);
      const int32_t delta = futilityThreshold * depthRemaining;
      if (r.score >= beta + delta || r.score <= alpha - delta) {
        return r;
      }
    }
    #endif

    const bool inCheck = can_enemy_attack<TURN>(thread->pos, lsb(thread->pos.pieceBitboards_[moverKing]));

    Move lastFoundBestMove = (isNullCacheResult(cr) ? kNullMove : cr.bestMove);

    ExtMove moves[kMaxNumMoves];
    ExtMove *movesEnd = compute_moves<TURN, MoveGenType::ALL_MOVES>(thread->pos, moves);

    if (movesEnd - moves == 0) {
      if (inCheck) {
        return SearchResult<TURN>(kCheckmate + plyFromRoot, kNullMove);
      } else {
        return SearchResult<TURN>(Evaluation(0), kNullMove);
      }
    }

    // const ExtMove lastMove = thread->pos.history_.size() > 0 ? thread->pos.history_.back() : kNullExtMove;
    // TODO: use lastMove (above) to sort better.
    for (ExtMove *move = moves; move < movesEnd; ++move) {
      move->score = 0;

      // Bonus for capturing a piece.
      // (+0.1042 ± 0.0146) after 256 games at 50,000 nodes/move
      move->score += kMoveOrderPieceValues[move->capture];

      // Bonus if it was the last-found best move.
      // (+0.0703 ± 0.0148) after 256 games at 50,000 nodes/move
      move->score += value_or_zero((move->move == lastFoundBestMove) && (depthRemaining >= 1), 5000);

      // Bonus if siblings like a move.
      // (+0.0144 ± 0.0074) after 1024 games at 50,000 nodes/move
      move->score += value_or_zero(move->move == recommendedMoves.moves[0], 50);
      move->score += value_or_zero(move->move == recommendedMoves.moves[1], 50);

      // History Heuristic
      // (+0.0310 ± 0.0073) after 1024 games at 50,000 nodes/move
      const int32_t history = thinker->historyHeuristicTable[TURN][move->move.from][move->move.to];
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

    SearchResult<TURN> r(
      kMinEval + 1,
      kNullMove
    );

    // Should be optimized away if SEARCH_TYPE != SearchTypeRoot.
    std::vector<SearchResult<TURN>> children;
    size_t numValidMoves = 0;
    for (int isDeferred = 0; isDeferred <= 1; ++isDeferred) {
      ExtMove *start = (isDeferred ? deferredMoves : &moves[0]);
      ExtMove *end = (isDeferred ? deferredMovesEnd : movesEnd);
      for (ExtMove *extMove = start; extMove < end; ++extMove) {

        make_move<TURN>(&thread->pos, extMove->move);

        // Don't move into check.
        if (can_enemy_attack<TURN>(thread->pos, lsb(thread->pos.pieceBitboards_[moverKing]))) {
          undo<TURN>(&thread->pos);
          continue;
        }

        if (IS_PARALLEL) {
          if (depthRemaining > 3 && isDeferred == 0) {
            if (extMove == moves) {
              thinker->_manager.start_searching(thread->pos.hash_);
            } else if (!thinker->_manager.should_start_searching(thread->pos.hash_)) {
              *(deferredMovesEnd++) = *extMove;
              undo<TURN>(&thread->pos);
              continue;
            }
          } else {
            thinker->_manager.start_searching(thread->pos.hash_);
          }
        }

        ++numValidMoves;

        // Null-window search.
        // (+0.0269 ± 0.0072) after 1024 games at 50,000 nodes/move
        SearchResult<TURN> a(0, kNullMove);
        constexpr SearchType kChildSearchType = SEARCH_TYPE == SearchTypeRoot ? SearchTypeNormal : SEARCH_TYPE;
        if (extMove == moves) {
          a = flip(Thinker::search<opposingColor, kChildSearchType, IS_PARALLEL>(thinker, thread, depthRemaining - 1, plyFromRoot + 1, -beta, -alpha, recommendationsForChildren, distFromPV + (extMove != moves)));
        } else {
          a = flip(Thinker::search<opposingColor, SearchTypeNullWindow, IS_PARALLEL>(thinker, thread, depthRemaining - 1, plyFromRoot + 1, -alpha - 1, -alpha, recommendationsForChildren, distFromPV + (extMove != moves)));
          if (a.score > alpha && a.score < beta) {
            a = flip(Thinker::search<opposingColor, kChildSearchType, IS_PARALLEL>(thinker, thread, depthRemaining - 1, plyFromRoot + 1, -beta, -alpha, recommendationsForChildren, distFromPV + (extMove != moves)));
          }
        }

        if (IS_PARALLEL) {
          thinker->_manager.finished_searching(thread->pos.hash_);
        }
        undo<TURN>(&thread->pos);

        // We don't bother to break here, since all of our children will automatically return a low-cost,
        // good-faith estimate (unless our depth is 4, in which case thinker will be true for the depth above
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
          if (a.score > r.score) {
            r.score = a.score;
            r.move = extMove->move;
            recommendationsForChildren.add(a.move);
          }
          if (children.size() >= thinker->multiPV) {
            alpha = std::max(alpha, children[thinker->multiPV - 1].score);
          }
        } else {
          if (a.score > r.score) {
            r.score = a.score;
            r.move = extMove->move;
            recommendationsForChildren.add(a.move);
            if (r.score >= beta) {
              // TODO: make thinker thread safe.
              thinker->historyHeuristicTable[TURN][r.move.from][r.move.to] += value_or_zero(extMove->capture == Piece::NO_PIECE, depthRemaining * depthRemaining);
              break;
            }
            if (r.score > alpha) {
              alpha = r.score;
            }
          }
        }
      }
      if (!IS_PARALLEL) {
        break;
      }
    }

    if (SEARCH_TYPE == SearchTypeRoot) {
      // We rely on the stability of std::sort to guarantee that children that
      // are PV nodes are sorted above children that are not PV nodes (but have
      // the same score).
      // TODO: make this thread safe.
      thinker->variations.clear();
      for (size_t i = 0; i < std::min(children.size(), thinker->multiPV); ++i) {
        thinker->variations.push_back(to_white(children[i]));
      }
    }

    if (numValidMoves == 0) {
      r.score = inCheck ? kCheckmate + plyFromRoot : 0;
      r.move = kNullMove;
    }

    r.analysisComplete = !thinker->stopThinkingCondition->should_stop_thinking(*thinker);

    if (r.analysisComplete) {
      NodeType nodeType = NodeTypePV;
      if (r.score >= originalBeta) {
        nodeType = NodeTypeCut_LowerBound;
      } else if (r.score <= originalAlpha) {
        nodeType = NodeTypeAll_UpperBound;
      }
      const CacheResult cr = thinker->cache.create_cache_result(
        thread->pos.hash_,
        depthRemaining,
        r.score,
        r.move,
        nodeType,
        distFromPV
      );
      thinker->cache.insert<IS_PARALLEL>(cr);
    }


    return r;
  }

  std::vector<SearchResult<Color::WHITE>> variations;

  template<Color TURN>
  static void _search_with_aspiration_window(Thinker* thinker, std::vector<Thread> *threadObjs, Depth depth) {
    // TODO: aspiration window

    CacheResult cr = thinker->cache.find<false>((*threadObjs)[0].pos.hash_);

    if (threadObjs->size() == 0) {
      throw std::runtime_error("");
    }

    std::vector<std::thread> threads;
    for (size_t i = 0; i < threadObjs->size(); ++i) {
      threads.push_back(std::thread(
        Thinker::search<TURN, SearchTypeRoot, true>,
        thinker,
        &((*threadObjs)[i]),
        depth,
        0,
        kMinEval,
        kMaxEval,
        RecommendedMoves(),
        0
      ));
    }
    for (size_t i = 0; i < threads.size(); ++i) {
      threads[i].join();
    }
  }

  // TODO: making threads work with multiPV seems really nontrivial.

  SearchResult<Color::WHITE> search(Position* pos, size_t depthLimit, std::function<void(Position *, SearchResult<Color::WHITE>, size_t, double)> callback) {
    std::chrono::time_point<std::chrono::steady_clock> tstart = std::chrono::steady_clock::now();

    Position copy(*pos);
    // It's important to call this at the beginning of a search, since if we're sharing Position (e.g. selfplay.cpp) we
    // need to recompute piece map scores using our own weights.
    copy.set_piece_maps(this->pieceMaps);

    this->nodeCounter = 0;
    this->leafCounter = 0;
    stopThinkingCondition->start_thinking(*this);
    this->cache.starting_search(pos->hash_);

    std::vector<Thread> threadObjs;
    for (size_t i = 0; i < std::max<size_t>(1, this->numThreads); ++i) {
      threadObjs.push_back(Thread(i, *pos, this->evaluator));
    }

    size_t depth;
    bool stoppedEarly = false;
    for (depth = 1; depth <= depthLimit; ++depth) {
      if (pos->turn_ == Color::WHITE) {
        this->_search_fixed_depth<Color::WHITE>(pos, &threadObjs, Depth(depth));
      } else {
        this->_search_fixed_depth<Color::BLACK>(pos, &threadObjs, Depth(depth));
      }
      std::chrono::duration<double> delta = std::chrono::steady_clock::now() - tstart;
      const double secs = std::max(0.001, std::chrono::duration_cast<std::chrono::milliseconds>(delta).count() / 1000.0);
      if (this->stopThinkingCondition->should_stop_thinking(*this)) {
        stoppedEarly = true;
        break;
      }
      callback(pos, this->variations[0], depth, secs);
    }

    // Before we return, we make one last pass through our children. This is important if our interrupted search has proven
    // our old best move was terrible, but isn't done analyzing all its siblings yet.
    // (+0.0869 ± 0.0160) after 256 games at 50,000 nodes/move
    if (stoppedEarly) {
      std::vector<SearchResult<Color::WHITE>> children;
      for_all_moves(pos, [this, &children](const Position& pos, ExtMove move) {
        CacheResult cr = this->cache.unsafe_find(pos.hash_);
        if (isNullCacheResult(cr)) {
          return;
        }
        if (cr.nodeType != NodeTypePV) {
          return;
        }
        Evaluation eval = cr.lowerbound();
        if (pos.turn_ == Color::BLACK) {
          eval *= -1;
        }
        children.push_back(SearchResult<Color::WHITE>(eval, move.move));
      });
      if (pos->turn_ == Color::WHITE) {
        std::sort(
          children.begin(),
          children.end(),
          [](SearchResult<Color::WHITE> a, SearchResult<Color::WHITE> b) -> bool {
            return a.score > b.score;
        });
      } else {
        std::sort(
          children.begin(),
          children.end(),
          [](SearchResult<Color::WHITE> a, SearchResult<Color::WHITE> b) -> bool {
            return a.score < b.score;
        });
      }

      // This may not always be true. For instance, if we're using an aspiration window,
      // we might stop after analyzing the first move, and the first move may fail low,
      // in which case we will have no primary variations in the cache. In this case,
      // we simply use the result of the last search. Unfortunately this means using
      // a move whose score just dropped, but since all our other moves only have bounds
      // (not exact scores) it's not clear that we can do better than this. Also this
      // function probably plays weirdly with incomplete, multi-pv, aspiration-window
      // searches, since the 1st PV might be completely searched, worse than the
      // aspiration bounds, and marked as non-PV, while the 5th PV hasn't been searched
      // at this depth yet, and so it is considered "better" than the 1st PV. If might
      // be worth using special PV-ness logic for in the root node to prevent this, but
      // for now we're focused on improving the performance when multiPV=1.
      if (children.size() > 0) {
        this->variations.clear();
        for (size_t i = 0; i < std::min(children.size(), this->multiPV); ++i) {
          this->variations.push_back(children[i]);
        }
        std::chrono::duration<double> delta = std::chrono::steady_clock::now() - tstart;
        const double secs = std::max(0.001, std::chrono::duration_cast<std::chrono::milliseconds>(delta).count() / 1000.0);
        callback(pos, this->variations[0], depth, secs);
      }
    }

    return this->variations[0];
  }

  SearchResult<Color::WHITE> search(Position *pos, size_t depthLimit) {
    return this->search(pos, depthLimit, [](Position *position, SearchResult<Color::WHITE> results, size_t depth, double secs) {});
  }

  template<Color TURN>
  SearchResult<Color::WHITE> _search_fixed_depth(Position* pos, std::vector<Thread> *threadObjs, Depth depth) {

    if (pos->turn_ == Color::WHITE) {
      Thinker::_search_with_aspiration_window<TURN>(this, threadObjs, depth);
    } else {
      Thinker::_search_with_aspiration_window<TURN>(this, threadObjs, depth);
    }

    CacheResult cr = this->cache.find<false>(pos->hash_);
    if (isNullCacheResult(cr)) {
      throw std::runtime_error("Null result from search");
    }
    return to_white(SearchResult<TURN>(cr.eval, cr.bestMove));
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
  std::chrono::time_point<std::chrono::steady_clock> current_time() const {
    return std::chrono::steady_clock::now();
  }
  bool should_stop_thinking(const Thinker& thinker) {
    std::chrono::duration<double> delta = this->current_time() - startTime;
    return std::chrono::duration_cast<std::chrono::milliseconds>(delta).count() > milliseconds;
  }
 private:
  std::chrono::high_resolution_clock::time_point startTime;
  uint64_t milliseconds;
};

struct OrStopCondition : public StopThinkingCondition {
  OrStopCondition(
    const std::shared_ptr<StopThinkingCondition>& a,
    const std::shared_ptr<StopThinkingCondition>& b) : a(a), b(b), c(nullptr) {}
  OrStopCondition(
    const std::shared_ptr<StopThinkingCondition>& a,
    const std::shared_ptr<StopThinkingCondition>& b,
    const std::shared_ptr<StopThinkingCondition>& c) : a(a), b(b), c(c) {}
  void start_thinking(const Thinker& thinker) {
    if (a != nullptr) {
      a->start_thinking(thinker);
    }
    if (b != nullptr) {
      b->start_thinking(thinker);
    }
    if (c != nullptr) {
      c->start_thinking(thinker);
    }
  }
  bool should_stop_thinking(const Thinker& thinker) {
    if (a != nullptr && a->should_stop_thinking(thinker)) {
      return true;
    }
    if (b != nullptr && b->should_stop_thinking(thinker)) {
      return true;
    }
    if (c != nullptr && c->should_stop_thinking(thinker)) {
      return true;
    }
    return false;
  }
 private:
  std::shared_ptr<StopThinkingCondition> a, b, c;
};

struct StopThinkingSwitch : public StopThinkingCondition {
  StopThinkingSwitch() {}
  void start_thinking(const Thinker& thinker) {
    this->lock.lock();
    this->shouldStop = false;
    this->lock.unlock();
  }
  bool should_stop_thinking(const Thinker& thinker) {
    this->lock.lock();
    bool r = this->shouldStop;
    this->lock.unlock();
    return r;
  }
  void stop() {
    this->lock.lock();
    this->shouldStop = true;
    this->lock.unlock();
  }
 private:
  SpinLock lock;
  bool shouldStop;
};


// TODO: there is a bug where
// "./a.out fen 8/8/8/1k6/3P4/8/8/3K4 w - - 0 1 depth 17"
// claims white is winning.

}  // namespace ChessEngine

#endif  // SEARCH_H