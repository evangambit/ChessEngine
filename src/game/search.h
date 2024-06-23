#ifndef SEARCH_H
#define SEARCH_H

#include <cassert>
#include <cstdint>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <thread>
#include <memory>
#include <unordered_set>

#include "geometry.h"
#include "utils.h"
#include "Position.h"
#include "Thinker.h"
#include "TranspositionTable.h"
#include "movegen.h"
#include "movegen/sliding.h"
#include "Evaluator.h"

#ifndef COMPLEX_SEARCH
#define COMPLEX_SEARCH 0
#endif

#ifndef SIMPLE_SEARCH
#define SIMPLE_SEARCH 0
#endif

namespace ChessEngine {

bool history_is(const Position& pos, std::string movesString) {
  std::vector<std::string> moves = split(movesString, ' ');
  if (pos.history_.size() != moves.size()) {
    return false;
  }
  for (size_t i = 0; i < moves.size(); ++i) {
    if (pos.history_[i].uci() != moves[i]) {
      return false;
    }
  }
  return true;
}

struct GoCommand {
  GoCommand()
  : depthLimit(100), nodeLimit(-1), timeLimitMs(-1),
  wtimeMs(0), btimeMs(0), wIncrementMs(0), bIncrementMs(0), movesUntilTimeControl(-1) {}

  Position pos;

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

std::unordered_set<std::string> compute_legal_moves_set(Position *pos) {
  std::unordered_set<std::string> legalMoves;
  ExtMove moves[kMaxNumMoves];
  ExtMove *end;
  if (pos->turn_ == Color::WHITE) {
    end = compute_legal_moves<Color::WHITE>(pos, moves);
  } else {
    end = compute_legal_moves<Color::BLACK>(pos, moves);
  }
  for (ExtMove *move = moves; move < end; ++move) {
    legalMoves.insert(move->uci());
  }
  return legalMoves;
}

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

enum SearchType {
  SearchTypeRoot,
  SearchTypeNormal,
  SearchTypeNullWindow,
  SearchTypeExtended,
};

constexpr int kQSimplePieceValues[7] = {
  // Note "NO_PIECE" has a score of 200 since this
  // encourages qsearch to value checks. (+0.02625)
  50, 100, 450, 500, 1000, 2000, 9999
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

constexpr int kSyncDepth = 2;

struct Thread {
  Thread(uint64_t id, const Position& pos, const Evaluator& e, const std::unordered_set<std::string>& moves)
  : id(id), pos(pos), evaluator(e), nodeCounter(0), moves(moves) {}
  uint64_t id;
  Position pos;
  Evaluator evaluator;
  uint64_t nodeCounter;
  const std::unordered_set<std::string>& moves;
};

// TODO: qsearch can leave you in check
template<Color TURN>
static SearchResult<TURN> qsearch(Thinker *thinker, Thread *thread, int32_t depth, int32_t plyFromRoot, Evaluation alpha, Evaluation beta) {
  ++thread->nodeCounter;

  if (std::popcount(thread->pos.pieceBitboards_[coloredPiece<TURN, Piece::KING>()]) == 0) {
    return SearchResult<TURN>(kMissingKing, kNullMove);
  }

  // +0.0402 ± 0.0124 after 512 games at 50,000 nodes/move
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
    return SearchResult<TURN>(depth == 0 ? kCheckmate : kQCheckmate, kNullMove);
  }

  // If we can stand pat for a beta cutoff, or if we have no moves, return.
  Threats<TURN> threats(thread->pos);
  SearchResult<TURN> r(thread->evaluator.score<TURN>(thread->pos, threats), kNullMove);
  {
    // Add a penalty to standing pat if we have hanging pieces.
    // k=50 is actually worse than k=0 (0.0254 ± 0.0079)
    // Threats<opposingColor> enemyThreats(thread->pos);
    // constexpr int k = 50;
    // r.score -= value_or_zero((enemyThreats.badForTheir[Piece::PAWN] & thread->pos.pieceBitboards_[coloredPiece<TURN>(Piece::PAWN)]) > 0, k);
    // r.score -= value_or_zero((enemyThreats.badForTheir[Piece::KNIGHT] & thread->pos.pieceBitboards_[coloredPiece<TURN>(Piece::KNIGHT)]) > 0, k);
    // r.score -= value_or_zero((enemyThreats.badForTheir[Piece::BISHOP] & thread->pos.pieceBitboards_[coloredPiece<TURN>(Piece::BISHOP)]) > 0, k);
    // r.score -= value_or_zero((enemyThreats.badForTheir[Piece::ROOK] & thread->pos.pieceBitboards_[coloredPiece<TURN>(Piece::ROOK)]) > 0, k);
    // r.score -= value_or_zero((enemyThreats.badForTheir[Piece::QUEEN] & thread->pos.pieceBitboards_[coloredPiece<TURN>(Piece::QUEEN)]) > 0, k);
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
    // Should be "(move->score < 0 && !inCheck)" ?
    if (move->score < 0 && r.score > kQLongestForcedMate) {
      break;
    }

    make_move<TURN>(&thread->pos, move->move);

    SearchResult<TURN> child = flip(qsearch<opposingColor>(thinker, thread, depth + 1, plyFromRoot + 1, -beta, -alpha));
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

constexpr int kThreadingDepth = 2;

#define IS_PRINT_NODE 0
// #define IS_PRINT_NODE history_is(thread->pos, "h8f8 c7c8 f8c8 c1c8 a8b7")
// #define IS_PRINT_NODE (thread->pos.hash_ == 17112545643981194285LLU)
// #define IS_PRINT_NODE (thread->pos.hash_  % 425984 == 958)

std::string pad(int n) {
  std::string r = "";
  for (int i = 0; i < n; ++i) {
    r += "  ";
  }
  return r;
}

template<Color TURN, SearchType SEARCH_TYPE, bool IS_PARALLEL>
static SearchResult<TURN> search(
  Thinker *thinker,
  Thread *thread,
  const Depth depthRemaining,
  const Depth plyFromRoot,
  Evaluation alpha, Evaluation beta,
  RecommendedMoves recommendedMoves,
  uint16_t distFromPV) {

  if (IS_PRINT_NODE) {
    std::cout << pad(plyFromRoot) << "start " << thread->pos.hash_ << " (depth:" << int(depthRemaining) << " alpha:" << alpha << " beta:" << beta << ") " << thread->pos.history_ << std::endl;
  }


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

  constexpr Color opposingColor = opposite_color<TURN>();
  constexpr ColoredPiece moverKing = coloredPiece<TURN, Piece::KING>();

  thinker->cache.prefetch(thread->pos.hash_);

  if (depthRemaining >= kThreadingDepth) {
    thinker->stopThinkingLock.lock();
    const bool shouldStopThinking = thinker->stopThinkingCondition->should_stop_thinking(*thinker);
    thinker->stopThinkingLock.unlock();
    if (shouldStopThinking) {
      if (IS_PRINT_NODE) {
        std::cout << pad(plyFromRoot) << "end a " << thread->pos.hash_ << " interrupted" << std::endl;
      }
      return SearchResult<TURN>(0, kNullMove, false);
    }
  }

  if (std::popcount(thread->pos.pieceBitboards_[coloredPiece<TURN, Piece::KING>()]) == 0) {
    if (IS_PRINT_NODE) {
      std::cout << pad(plyFromRoot) << "end b " << thread->pos.hash_ << " no king" << std::endl;
    }
    return SearchResult<TURN>(kMissingKing, kNullMove);
  }

  if (thread->pos.is_3fold_repetition(plyFromRoot)) {
    if (IS_PRINT_NODE) {
      std::cout << pad(plyFromRoot) << "end c " << thread->pos.hash_ << " 3fold draw" << std::endl;
    }
    return SearchResult<TURN>(Evaluation(0), kNullMove);
  }

  if (SEARCH_TYPE != SearchTypeRoot && thread->evaluator.is_material_draw(thread->pos)) {
    if (IS_PRINT_NODE) {
      std::cout << pad(plyFromRoot) << "end d " << thread->pos.hash_ << " material draw" << std::endl;
    }
    return SearchResult<TURN>(Evaluation(0), kNullMove);
  }

  CacheResult cr = thinker->cache.find<IS_PARALLEL>(thread->pos.hash_);
  // Short-circuiting due to a cached result.
  // 0.0729 ± 0.0109 after 512 games at 50,000 nodes/move
  if (!(SEARCH_TYPE == SearchTypeRoot && thread->id == 0)) {
    // It's important that the root node of thread 0 not short-circuit here
    // so that thinker->variations is properly set.
    if (!isNullCacheResult(cr) && cr.depthRemaining >= depthRemaining) {
      if (cr.nodeType == NodeTypePV || cr.lowerbound() >= beta || cr.upperbound() <= alpha) {
        if (IS_PRINT_NODE) {
          std::cout << pad(plyFromRoot) << "end e " << thread->pos.hash_ << " cached " << cr << std::endl;
        }
        return SearchResult<TURN>(std::max(alpha, std::min(beta, cr.eval)), cr.bestMove);
      }

      // Adjust alpha/beta based on cached value. This helps us achieve more cuts and avoid returning
      // scores we know are wrong. We don't want to do this for root nodes since a partially-searched
      // root node could return a terrible move.
      // 0.1553 ± 0.0128 after 512 games at 50,000 nodes/move
      alpha = std::max(alpha, cr.lowerbound());
      beta = std::min(beta, cr.upperbound());
    }
  }

  if (depthRemaining <= 0) {
    #if SIMPLE_SEARCH
    {
      SearchResult<TURN> r = qsearch<TURN>(thinker, thread, 0, plyFromRoot, alpha, beta);
      NodeType nodeType = NodeTypePV;
      if (r.score >= beta) {
        nodeType = NodeTypeCut_LowerBound;
      } else if (r.score <= alpha) {
        nodeType = NodeTypeAll_UpperBound;
      }
      CacheResult newCr = thinker->cache.create_cache_result(
        thread->pos.hash_,
        depthRemaining,
        std::max(originalAlpha, std::min(originalBeta, r.score)),
        r.move,
        nodeType,
        distFromPV
      );
      thinker->cache.insert<IS_PARALLEL>(newCr);
      if (IS_PRINT_NODE) {
        std::cout << pad(plyFromRoot) << "  insert " << newCr << std::endl;
        std::cout << pad(plyFromRoot) << "end f " << thread->pos.hash_ << " qsearch " << r << std::endl;
      }
      return r;
    }
    #else
    {
      // Quiescence Search
      // 0.4814 ± 0.0035 after 512 games at 50,000 nodes/move
      SearchResult<TURN> r = qsearch<TURN>(thinker, thread, 0, plyFromRoot, alpha, beta);

      // // Search Extensions
      // // 0.0293 ± 0.0116 after 512 games at 50,000 nodes/move
      // if (SEARCH_TYPE == SearchTypeNormal && r.score > alpha && r.score < beta) {
      //   r = search<TURN, SearchTypeExtended, IS_PARALLEL>(thinker, thread, 2, plyFromRoot, alpha, beta, recommendedMoves, distFromPV);
      // }

      if (IS_PRINT_NODE) {
        std::cout << pad(plyFromRoot) << "end g " << thread->pos.hash_ << " qsearch " << r << std::endl;
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
        std::max(originalAlpha, std::min(originalBeta, r.score)),
        r.move,
        nodeType,
        distFromPV
      );
      if (IS_PRINT_NODE) {
        std::cout << pad(plyFromRoot) << "  insert " << cr << std::endl;
      }
      thinker->cache.insert<IS_PARALLEL>(cr);
      return r;
    }
    #endif
  }

  const bool inCheck = can_enemy_attack<TURN>(thread->pos, lsb(thread->pos.pieceBitboards_[moverKing]));

  #if COMPLEX_SEARCH
  if (depthRemaining == 1) {
    const int32_t futilityThreshold = 150;
    if (!isNullCacheResult(cr)) {
      // 0.3633 ± 0.0294 after 64 games at 100,000 nodes/move
      if (int32_t(cr.lowerbound()) >= beta + futilityThreshold || int32_t(cr.upperbound()) <= alpha - futilityThreshold) {
        if (IS_PRINT_NODE) {
          std::cout << pad(plyFromRoot) << "end h " << thread->pos.hash_ << " futile1" << std::endl;
        }
        return SearchResult<TURN>(std::max<int32_t>(alpha, std::min<int32_t>(beta, cr.eval)), cr.bestMove);
      }
    } else {
      SearchResult<TURN> r = qsearch<TURN>(thinker, thread, 0, plyFromRoot, alpha, beta);
      if (int32_t(r.score) >= beta + futilityThreshold || int32_t(r.score) <= alpha - futilityThreshold) {
        return SearchResult<TURN>(std::max<int32_t>(alpha, std::min<int32_t>(beta, r.score)), r.move);
      }
    }
  }
  #endif  // COMPLEX_SEARCH

  Move lastFoundBestMove = (isNullCacheResult(cr) ? kNullMove : cr.bestMove);

  ExtMove moves[kMaxNumMoves];
  ExtMove *movesEnd;
  if (SEARCH_TYPE == SearchTypeRoot) {
    movesEnd = compute_legal_moves<TURN>(&thread->pos, moves);
  } else {
    movesEnd = compute_moves<TURN, MoveGenType::ALL_MOVES>(thread->pos, moves);
  }

  if (movesEnd - moves == 0) {
    if (inCheck) {
      if (IS_PRINT_NODE) {
        std::cout << pad(plyFromRoot) << "end i " << thread->pos.hash_ << " checkmate" << std::endl;
      }
      const CacheResult cr = thinker->cache.create_cache_result(
        thread->pos.hash_,
        99,
        kCheckmate,
        kNullMove,
        NodeTypePV,
        distFromPV
      );
      thinker->cache.insert<IS_PARALLEL>(cr);
      return SearchResult<TURN>(kCheckmate, kNullMove);
    } else {
      if (IS_PRINT_NODE) {
        std::cout << pad(plyFromRoot) << "end j " << thread->pos.hash_ << " stalemate" << std::endl;
      }
      const CacheResult cr = thinker->cache.create_cache_result(
        thread->pos.hash_,
        99,
        0,
        kNullMove,
        NodeTypePV,
        distFromPV
      );
      thinker->cache.insert<IS_PARALLEL>(cr);
      return SearchResult<TURN>(Evaluation(0), kNullMove);
    }
  }

  // We need to check this *after* we do the checkmate test above.
  if (thread->pos.is_fifty_move_rule()) {
    if (IS_PRINT_NODE) {
      std::cout << pad(plyFromRoot) << "end k " << thread->pos.hash_ << " 50 move draw" << std::endl;
    }
    return SearchResult<TURN>(Evaluation(0), kNullMove);
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

    #if !SIMPLE_SEARCH
      // Bonus if siblings like a move.
      // (+0.0343 ± 0.0117) after 512 games at 50,000 nodes/move
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
    #endif
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

  // // (+0.0107 ± 0.0121) Null-move pruning.
  // const bool scaredOfZugzwang = std::popcount(thread->pos.colorBitboards_[TURN] & ~thread->pos.pieceBitboards_[coloredPiece<TURN, Piece::PAWN>()]) <= 3;
  // if (!inCheck && !scaredOfZugzwang && depthRemaining >= 2 && SEARCH_TYPE != SearchTypeExtended) {
  //   const bool isExpectedCutNode = !isNullCacheResult(cr) && cr.nodeType == NodeTypeCut_LowerBound;
  //   if (isExpectedCutNode && cr.lowerbound() >= originalBeta) {
  //     make_nullmove<TURN>(&thread->pos);
  //     SearchResult<TURN> a = flip(search<opposingColor, SearchTypeExtended, IS_PARALLEL>(thinker, thread, depthRemaining - 2, plyFromRoot + 1, -beta, -(beta - 1), recommendationsForChildren, distFromPV));
  //     undo_nullmove<TURN>(&thread->pos);
  //     if (a.score >= originalBeta) {
  //       a.score = originalBeta;
  //       a.move = kNullMove;
  //       return a;
  //     }
  //   }
  // }

  // Should be optimized away if SEARCH_TYPE != SearchTypeRoot.
  std::vector<VariationHead<TURN>> children;
  size_t numValidMoves = 0;
  for (int isDeferred = 0; isDeferred <= 1; ++isDeferred) {
    ExtMove *start = (isDeferred ? deferredMoves : &moves[0]);
    ExtMove *end = (isDeferred ? deferredMovesEnd : movesEnd);
    for (ExtMove *extMove = start; extMove < end; ++extMove) {
      if (SEARCH_TYPE == SearchTypeRoot) {
        if (!thread->moves.contains(extMove->uci())) {
          continue;
        }
      }

      if (IS_PRINT_NODE) {
        std::cout << pad(plyFromRoot) << "> move " << extMove->move << std::endl;
      }

      make_move<TURN>(&thread->pos, extMove->move);

      // Don't move into check.
      if (can_enemy_attack<TURN>(thread->pos, lsb(thread->pos.pieceBitboards_[moverKing]))) {
        undo<TURN>(&thread->pos);
        continue;
      }

      if (IS_PARALLEL) {
        if (depthRemaining > 1 && isDeferred == 0) {
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
      #if !SIMPLE_SEARCH
        if (extMove == moves) {
          a = flip(search<opposingColor, kChildSearchType, IS_PARALLEL>(thinker, thread, depthRemaining - 1, plyFromRoot + 1, -beta, -alpha, recommendationsForChildren, distFromPV + (extMove != moves)));
        } else {
          a = flip(search<opposingColor, SearchTypeNullWindow, IS_PARALLEL>(thinker, thread, depthRemaining - 1, plyFromRoot + 1, -(alpha + 1), -alpha, recommendationsForChildren, distFromPV + (extMove != moves)));
          if (a.score > alpha) {
            a = flip(search<opposingColor, kChildSearchType, IS_PARALLEL>(thinker, thread, depthRemaining - 1, plyFromRoot + 1, -beta, -(alpha + 1), recommendationsForChildren, distFromPV + (extMove != moves)));
          }
        }
      #else
        a = flip(search<opposingColor, kChildSearchType, IS_PARALLEL>(thinker, thread, depthRemaining - 1, plyFromRoot + 1, -beta, -alpha, recommendationsForChildren, distFromPV + (extMove != moves)));
      #endif

      if (IS_PRINT_NODE) {
        std::cout << pad(plyFromRoot) << ": a " << a << std::endl;
      }

      a.score -= (a.score > -kQLongestForcedMate);
      a.score += (a.score <  kQLongestForcedMate);

      if (IS_PARALLEL) {
        thinker->_manager.finished_searching(thread->pos.hash_);
      }
      undo<TURN>(&thread->pos);

      if (IS_PRINT_NODE) {
        std::cout << pad(plyFromRoot) << "< move " << extMove->move << " " << a << std::endl;
      }

      // We don't bother to break here, since all of our children will automatically return a low-cost,
      // good-faith estimate (unless our depth is 4, in which case thinker will be true for the depth above
      // us).
      r.analysisComplete &= a.analysisComplete;

      if (SEARCH_TYPE == SearchTypeRoot) {
        if (children.size() < thinker->multiPV || children.back().score < a.score) {
          children.push_back(VariationHead<TURN>(a.score, extMove->move, a.move));
          a.move = extMove->move;
          std::stable_sort(
            children.begin(),
            children.end(),
            [](VariationHead<TURN> a, VariationHead<TURN> b) -> bool {
              return a.score > b.score;
          });
          if (children.size() > thinker->multiPV) {
            children.pop_back();
          }
          if (a.score > r.score) {
            r.score = a.score;
            r.move = extMove->move;
            recommendationsForChildren.add(a.move);
          }
          if (children.size() >= thinker->multiPV) {
            alpha = std::max(alpha, children[thinker->multiPV - 1].score);
          }
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

  if (SEARCH_TYPE == SearchTypeRoot && thread->id == 0) {
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

  if (depthRemaining >= kSyncDepth) {
    if (IS_PARALLEL) {
      thinker->stopThinkingLock.lock();
      thinker->nodeCounter += thread->nodeCounter;
      r.analysisComplete = !thinker->stopThinkingCondition->should_stop_thinking(*thinker);
      thinker->stopThinkingLock.unlock();
      thread->nodeCounter = 0;
    } else {
      thinker->nodeCounter += thread->nodeCounter;
      r.analysisComplete = !thinker->stopThinkingCondition->should_stop_thinking(*thinker);
    }
  }

  #if SIMPLE_SEARCH
  r.score = std::max(originalAlpha, std::min(originalBeta, r.score));
  #endif

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
    if (IS_PRINT_NODE) {
      std::cout << pad(plyFromRoot) << "  insert " << cr << std::endl;
    }
    thinker->cache.insert<IS_PARALLEL>(cr);
  }

  if (IS_PRINT_NODE) {
    std::cout << pad(plyFromRoot) << "end l " << thread->pos.hash_ << " return " << r << std::endl;
  }

  return r;
}

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
      search<TURN, SearchTypeRoot, true>,
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

template<Color TURN>
bool _isCheckmate(Position *pos) {
  ExtMove moves[256];
  ExtMove *end = compute_legal_moves<TURN>(pos, moves);
  if (moves != end) {
    return false;
  }
  constexpr ColoredPiece moverKing = coloredPiece<TURN, Piece::KING>();
  const bool inCheck = can_enemy_attack<TURN>(*pos, lsb(pos->pieceBitboards_[moverKing]));
  return inCheck;
}

bool isCheckmate(Position *pos) {
  if (pos->turn_ == Color::WHITE) {
    return _isCheckmate<Color::WHITE>(pos);
  } else {
    return _isCheckmate<Color::BLACK>(pos);
  }
}

template<Color TURN>
bool _isStalemate(Position* pos, const Evaluator& evaluator) {
  if (_isCheckmate<TURN>(pos)) {
    return false;
  }
  if (pos->is_fifty_move_rule()) {
    return true;
  }
  if (pos->is_3fold_repetition(0)) {
    return true;
  }
  if (evaluator.is_material_draw(*pos)) {
    return true;
  }

  ExtMove moves[kMaxNumMoves];
  ExtMove *end;
  if (pos->turn_ == Color::WHITE) {
    end = compute_legal_moves<Color::WHITE>(pos, moves);
  } else {
    end = compute_legal_moves<Color::BLACK>(pos, moves);
  }
  if (moves == end) {
    return true;
  }
  return false;
}

bool isStalemate(Position *pos, const Evaluator& evaluator) {
  if (pos->turn_ == Color::WHITE) {
    return _isStalemate<Color::WHITE>(pos, evaluator);
  } else {
    return _isStalemate<Color::BLACK>(pos, evaluator);
  }
}

template<Color TURN>
static SearchResult<Color::WHITE> _search_fixed_depth(Thinker *thinker, const Position& pos, std::vector<Thread> *threadObjs, Depth depth) {
  if (_isCheckmate<TURN>(&((*threadObjs)[0].pos))) {
    return to_white(SearchResult<TURN>(kCheckmate, kNullMove));
  }
  if (pos.turn_ == Color::WHITE) {
    _search_with_aspiration_window<TURN>(thinker, threadObjs, depth);
  } else {
    _search_with_aspiration_window<TURN>(thinker, threadObjs, depth);
  }

  if (thinker->variations.size() == 0) {
    throw std::runtime_error("Null result from search");
  }
  VariationHead<Color::WHITE> pv = thinker->variations[0];


  return to_white(SearchResult<TURN>(pv.score, pv.move));
}

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
  std::chrono::time_point<std::chrono::system_clock> current_time() const {
    return std::chrono::system_clock::now();
  }
  bool should_stop_thinking(const Thinker& thinker) {
    std::chrono::duration<double> delta = this->current_time() - startTime;
    return std::chrono::duration_cast<std::chrono::milliseconds>(delta).count() > milliseconds;
  }
 private:
  std::chrono::time_point<std::chrono::system_clock> startTime;
  uint64_t milliseconds;
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

// TODO: making threads work with multiPV seems really nontrivial.
static SearchResult<Color::WHITE> search(Thinker *thinker, const GoCommand& command, std::shared_ptr<StopThinkingCondition> condition, std::function<void(Position *, VariationHead<Color::WHITE>, size_t, double)> callback) {

  if (!condition) {
    condition = std::make_shared<NeverStopThinkingCondition>();
  }

  thinker->nodeCounter = 0;

  uint64_t timeLimitMs = command.timeLimitMs;

  const uint64_t increment = command.pos.turn_ == Color::WHITE ? command.wIncrementMs : command.bIncrementMs;
  const uint64_t timeRemaining = (command.pos.turn_ == Color::WHITE ? command.wtimeMs : command.btimeMs) + increment;
  if (timeRemaining != 0) {
    timeLimitMs = increment + timeRemaining / 20;
  }

  thinker->stopThinkingCondition = std::make_unique<OrStopCondition>(
    std::make_shared<StopThinkingNodeCountCondition>(command.nodeLimit),
    std::make_shared<StopThinkingTimeCondition>(timeLimitMs),
    condition
  );

  std::chrono::time_point<std::chrono::steady_clock> tstart = std::chrono::steady_clock::now();

  Position copy(command.pos);
  // It's important to call this at the beginning of a search, since if we're sharing Position (e.g. selfplay.cpp) we
  // need to recompute piece map scores using our own weights.
  copy.set_piece_maps(thinker->pieceMaps);

  thinker->variations.clear();

  if (isCheckmate(&copy)) {
    VariationHead<Color::WHITE> variation(kCheckmate, kNullMove, kNullMove);
    callback(&copy, variation, 0, 0.0);
    if (copy.turn_ == Color::WHITE) {
      return SearchResult<Color::WHITE>(kCheckmate, kNullMove);
    } else {
      return SearchResult<Color::WHITE>(-kCheckmate, kNullMove);
    }
  }
  if (isStalemate(&copy, thinker->evaluator)) {
    VariationHead<Color::WHITE> variation(0, kNullMove, kNullMove);
    callback(&copy, variation, 0, 0.0);
    return SearchResult<Color::WHITE>(0, kNullMove);
  }

  thinker->stopThinkingCondition->start_thinking(*thinker);
  thinker->clear_history_heuristic();
  thinker->cache.starting_search(copy.hash_);

  std::vector<Thread> threadObjs;
  for (size_t i = 0; i < std::max<size_t>(1, thinker->numThreads); ++i) {
    threadObjs.push_back(Thread(i, copy, thinker->evaluator, command.moves));
  }

  std::vector<VariationHead<Color::WHITE>> lastVar;

  size_t depth;
  bool stoppedEarly = false;
  for (depth = 1; depth <= command.depthLimit; ++depth) {
    lastVar = thinker->variations;
    if (copy.turn_ == Color::WHITE) {
      _search_fixed_depth<Color::WHITE>(thinker, copy, &threadObjs, Depth(depth));
    } else {
      _search_fixed_depth<Color::BLACK>(thinker, copy, &threadObjs, Depth(depth));
    }
    std::chrono::duration<double> delta = std::chrono::steady_clock::now() - tstart;
    const double secs = std::max(0.001, std::chrono::duration_cast<std::chrono::milliseconds>(delta).count() / 1000.0);
    if (thinker->stopThinkingCondition->should_stop_thinking(*thinker)) {
      stoppedEarly = true;
      break;
    }
    callback(&copy, thinker->variations[0], depth, secs);
  }

  // We simply fall back on our previous search results if we are interrupted.
  // (+0.3125 ± 0.0204) after 128 games at 100,000 nodes/move
  if (stoppedEarly) {
    thinker->variations = lastVar;
    std::chrono::duration<double> delta = std::chrono::steady_clock::now() - tstart;
    const double secs = std::max(0.001, std::chrono::duration_cast<std::chrono::milliseconds>(delta).count() / 1000.0);
    callback(&copy, thinker->variations[0], depth, secs);
  }

  return thinker->variations[0].to_search_result();
}

static SearchResult<Color::WHITE> search(Thinker *thinker, GoCommand command, std::shared_ptr<StopThinkingCondition> condition) {
  return search(thinker, command, condition, [](Position *position, VariationHead<Color::WHITE> results, size_t depth, double secs) {});
}

// TODO: there is a bug where
// "./a.out fen 8/8/8/1k6/3P4/8/8/3K4 w - - 0 1 depth 17"
// claims white is winning.

}  // namespace ChessEngine

#endif  // SEARCH_H
