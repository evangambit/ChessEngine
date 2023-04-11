#ifndef SEARCH_H
#define SEARCH_H

#include <cassert>
#include <cstdint>

#include <algorithm>
#include <unordered_map>

#include "geometry.h"
#include "utils.h"
#include "Position.h"
#include "movegen.h"
#include "movegen/sliding.h"
#include "Evaluator.h"

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

struct CacheResult {
  Depth depth;
  Evaluation eval;
  Move bestMove;
  NodeType nodeType;
  inline Evaluation lowerbound() const {
    return (nodeType == NodeTypeAll_UpperBound || nodeType == NodeTypeQ) ? kMinEval : eval;
  }
  inline Evaluation upperbound() const {
    return (nodeType == NodeTypeCut_LowerBound || nodeType == NodeTypeQ) ? kMaxEval : eval;
  }
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

template<Color PERSPECTIVE>
struct SearchResult {
  SearchResult() : score(0), move(kNullMove) {}
  SearchResult(Evaluation score, Move move) : score(score), move(move) {}
  Evaluation score;
  Move move;
};

template<Color color>
bool operator<(SearchResult<color> a, SearchResult<color> b) {
  if (color == Color::WHITE) {
    return a.score < b.score;
  } else {
    return a.score > b.score;
  }
}

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
  std::unordered_map<uint64_t, CacheResult> cache;
  uint32_t historyHeuristicTable[Color::NUM_COLORS][64][64];
  size_t multiPV;

  Thinker() {
    reset_stuff();
    multiPV = 1;
  }

  void make_move(Position* pos, Move move) {
    if (pos->turn_ == Color::WHITE) {
      make_move<Color::WHITE>(pos, move);
    } else {
      make_move<Color::BLACK>(pos, move);
    }
  }

  void undo(Position* pos) {
    if (pos->turn_ == Color::BLACK) {
      undo<Color::WHITE>(pos);
    } else {
      undo<Color::BLACK>(pos);
    }
  }

  void print_variation(Position* pos, Move move) {
    uint64_t foo = pos->hash_;
    this->make_move(pos, move);
    auto it = this->cache.find(pos->hash_);

    if (it == this->cache.end()) {
      return;
    }

    Evaluation eval = it->second.eval;
    if (pos->turn_ == Color::BLACK) {
      eval *= -1;
    }

    if (eval < 0) {
      std::cout << eval << " " << move;
    } else {
      std::cout << "+" << eval << " " << move;
    }

    size_t counter = 1;
    while (it != this->cache.end() && it->second.bestMove != kNullMove && counter < 10) {
      std::cout << " " << it->second.bestMove;
      this->make_move(pos, it->second.bestMove);
      ++counter;
      it = cache.find(pos->hash_);
    }
    std::cout << std::endl;
    while (counter > 0) {
      this->undo(pos);
      --counter;
    }
    uint64_t bar = pos->hash_;
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

    const Bitboard theirTargets = compute_my_targets_except_king<opposingColor>(*pos);

    for (ExtMove *move = moves; move < end; ++move) {
      move->score = kQSimplePieceValues[move->capture];
      move->score -= value_or_zero(((threats.badForOur[move->piece] & bb(move->move.to)) > 0) && ~((threats.badForOur[move->piece] & bb(move->move.from)) > 0), kQSimplePieceValues[move->piece]);
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

  void reset_stuff() {
    this->leafCounter = 0;
    this->nodeCounter = 0;
    cache.clear();
    std::fill_n(historyHeuristicTable[Color::WHITE][0], 64 * 64, 0);
    std::fill_n(historyHeuristicTable[Color::BLACK][0], 64 * 64, 0);
  }

  template<Color TURN, SearchType SEARCH_TYPE>
  SearchResult<TURN> search(
    Position* pos, const Depth depth,
    Evaluation alpha, const Evaluation beta,
    RecommendedMoves recommendedMoves,
    bool isPV) {

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

    if (std::popcount(pos->pieceBitboards_[coloredPiece<TURN, Piece::KING>()]) == 0) {
      return SearchResult<TURN>(kMissingKing, kNullMove);
    }

    if (depth <= 0) {
      ++this->leafCounter;
      // Quiescence Search (0.4334 ± 0.0053)
      SearchResult<TURN> r = qsearch<TURN>(pos, 0, alpha, beta);

      NodeType nodeType = NodeTypePV;
      if (r.score >= originalBeta) {
        nodeType = NodeTypeCut_LowerBound;
      } else if (r.score <= originalAlpha) {
        nodeType = NodeTypeAll_UpperBound;
      }

      const CacheResult cr = CacheResult{
        depth,
        r.score,
        r.move,
        NodeTypePV,
      };
      auto it = this->cache.find(pos->hash_);
      if (this->cache.find(pos->hash_) == this->cache.end()) {
        this->cache.insert(std::make_pair(pos->hash_, cr));
      }
      return r;
    }

    if (pos->is_draw() || this->evaluator.is_material_draw(*pos)) {
      SearchResult<TURN>(Evaluation(0), kNullMove);
    }

    auto it = this->cache.find(pos->hash_);
    if (it != this->cache.end() && it->second.depth >= depth) {
      const CacheResult& cr = it->second;
      if (cr.nodeType == NodeTypePV || cr.lowerbound() >= beta) {
        return SearchResult<TURN>(cr.eval, cr.bestMove);
      }
    }

    // Futility pruning (0.1437 ± 0.0091 or ~53 Elo)
    // If our depth is 2 or exceeds the transposition table by 2 then we ignore moves that
    // are sufficiently bad that they are unlikely to be improved by increasing depth by 2.
    const Evaluation futilityThreshold = 30;
    if (it != this->cache.end() && depth - it->second.depth <= 2) {
      const CacheResult& cr = it->second;
      if (cr.lowerbound() >= beta + futilityThreshold * (depth - it->second.depth) || cr.upperbound() <= alpha - futilityThreshold * (depth - it->second.depth)) {
        return SearchResult<TURN>(cr.eval, cr.bestMove);
      }
    }
    if (it == this->cache.end() && depth <= 2) {
      SearchResult<TURN> r = qsearch<TURN>(pos, 0, alpha, beta);
      if (r.score >= beta + futilityThreshold * depth || r.score <= alpha - futilityThreshold * depth) {
        return r;
      }
    }

    const bool inCheck = can_enemy_attack<TURN>(*pos, lsb(pos->pieceBitboards_[moverKing]));

    // Null move pruning doesn't seem to help.
    // const Bitboard ourPieces = pos->colorBitboards_[TURN] & ~pos->pieceBitboards_[coloredPiece<TURN, Piece::PAWN>()];
    // if (depth == 3 && std::popcount(ourPieces) > 4 && !inCheck && !isPV && (it != this->cache.end() && it->second.lowerbound() - 20 >= beta)) {
    //   make_nullmove<TURN>(pos);
    //   SearchResult<TURN> a = flip(search<opposingColor, SearchTypeNormal>(pos, depth - 3, -beta, -beta+1, RecommendedMoves(), false));
    //   if (a.score >= beta && a.move != kNullMove) {
    //     undo_nullmove<TURN>(pos);
    //     return SearchResult<TURN>(beta + 1, kNullMove);
    //   }
    //   undo_nullmove<TURN>(pos);
    // }

    // // Null-window search doesn't seem to help,
    // if (SEARCH_TYPE == SearchTypeNormal && !isPV && (it != this->cache.end() && it->second.upperbound() + 40 < alpha)) {
    //   // Null Window search.
    //   SearchResult<TURN> r = search<TURN, SearchTypeNullWindow>(pos, depth, alpha, alpha + 1, RecommendedMoves(), false);
    //   if (r.score <= alpha) {
    //     return r;
    //   }
    // }

    Move lastFoundBestMove = (it != this->cache.end() ? it->second.bestMove : kNullMove);

    SearchResult<TURN> r(
      kMinEval + 1,
      kNullMove
    );

    ExtMove moves[kMaxNumMoves];
    ExtMove *end = compute_moves<TURN, MoveGenType::ALL_MOVES>(*pos, moves);

    if (end - moves == 0) {
      if (inCheck) {
        return SearchResult<TURN>(kCheckmate, kNullMove);
      } else {
        return SearchResult<TURN>(Evaluation(0), kNullMove);
      }
    }

    const Move lastMove = pos->history_.size() > 0 ? pos->history_.back().move : kNullMove;
    size_t numCaptures = 0;
    for (ExtMove *move = moves; move < end; ++move) {
      move->score = 0;

      const bool isCapture = (move->capture != Piece::NO_PIECE);
      numCaptures += isCapture;

      // Bonus for capturing a piece.  (+0.136 ± 0.012)
      move->score += kMoveOrderPieceValues[move->capture];

      // Bonus if it was the last-found best move.  (0.048 ± 0.014)
      move->score += value_or_zero((move->move == lastFoundBestMove) && (depth == 1), 5000);
      move->score += value_or_zero((move->move == lastFoundBestMove) && (depth == 2), 5000);
      move->score += value_or_zero((move->move == lastFoundBestMove) && (depth >= 3), 5000);

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

    std::sort(moves, end, [](ExtMove a, ExtMove b) {
      return a.score > b.score;
    });

    bool foo = it != this->cache.end() && it->second.depth >= depth - 1;

    RecommendedMoves recommendationsForChildren;

    // Should be optimized away if SEARCH_TYPE != SearchTypeRoot.
    std::vector<SearchResult<TURN>> children;
    size_t numValidMoves = 0;
    for (ExtMove *extMove = moves; extMove < end; ++extMove) {

      make_move<TURN>(pos, extMove->move);

      // Don't move into check.
      if (can_enemy_attack<TURN>(*pos, lsb(pos->pieceBitboards_[moverKing]))) {
        undo<TURN>(pos);
        continue;
      }

      ++numValidMoves;

      // TODO: why does "./main multiPV 2 moves g1f3 e7e6 d2d4 c7c5 b1c3 g8f6 e2e4 d7d5 depth 8"
      // return terrible moves for its secondary variation?

      // This simple, very limited null-window search has negligible effect (-0.003 ± 0.003).
      // SearchResult<TURN> a;
      // if (SEARCH_TYPE != SearchTypeRoot && extMove != moves && foo && it->second.upperbound() + 50 < alpha) {
      //   a = flip(search<opposingColor, SearchTypeNullWindow>(pos, depth - 1, -(alpha + 1), -alpha, recommendationsForChildren, isPV && (extMove == moves)));
      //   if (a.score > alpha) {
      //     a = flip(search<opposingColor, SearchTypeNormal>(pos, depth - 1, -beta, -alpha, recommendationsForChildren, isPV && (extMove == moves)));
      //   }
      // } else {
      //   a = flip(search<opposingColor, SearchTypeNormal>(pos, depth - 1, -beta, -alpha, recommendationsForChildren, isPV && (extMove == moves)));
      // }
      SearchResult<TURN> a = flip(search<opposingColor, SearchTypeNormal>(pos, depth - 1, -beta, -alpha, recommendationsForChildren, isPV && (extMove == moves)));
      a.score -= (a.score > -kLongestForcedMate);
      a.score += (a.score < kLongestForcedMate);

      undo<TURN>(pos);

      if (SEARCH_TYPE == SearchTypeRoot) {
        children.push_back(a);
        std::sort(children.begin(), children.end());
        if (a.score > r.score) {
          r.score = a.score;
          r.move = extMove->move;
          recommendationsForChildren.add(a.move);
          if (children.size() >= multiPV) {
            if (r.score > alpha) {
              alpha = r.score;
            }
          }
        }
      } else {
        if (a.score > r.score) {
          r.score = a.score;
          r.move = extMove->move;
          recommendationsForChildren.add(a.move);
          if (r.score >= beta) {
            // NOTE: Unlike other engines, we include captures in our history heuristic, as this
            // order captures that are materially equal.
            this->historyHeuristicTable[TURN][r.move.from][r.move.to] += depth * depth;
            break;
          }
          if (r.score > alpha) {
            alpha = r.score;
          }
        }
      }
    }

    if (numValidMoves == 0) {
      r.score = inCheck ? kCheckmate : 0;
      r.move = kNullMove;
    }

    NodeType nodeType = NodeTypePV;
    if (r.score >= originalBeta) {
      nodeType = NodeTypeCut_LowerBound;
    } else if (r.score <= originalAlpha) {
      nodeType = NodeTypeAll_UpperBound;
    }

    {
      const CacheResult cr = CacheResult{
        depth,
        r.score,
        r.move,
        nodeType,
      };
      it = this->cache.find(pos->hash_);  // Need to re-search since the iterator may have changed when searching my children.
      if (it == this->cache.end()) {
        this->cache.insert(std::make_pair(pos->hash_, cr));
      } else if (depth >= it->second.depth) {
        // We use ">=" because otherwise if we fail the aspiration window search the table will have
        // stale results.
        it->second = cr;
      }
    }

    return r;
  }

  template<Color TURN>
  SearchResult<TURN> search_with_aspiration_window(Position* pos, Depth depth, SearchResult<TURN> lastResult) {
    // TODO: the aspiration window technique used here should probably be implemented for internal nodes too.
    // Even just using this at the root node gives my engine a +0.25 (n=100) score against itself.
    // Table of historical experiments (program with window vs program without)
    // 100: 0.099 ± 0.021
    //  75: 0.152 ± 0.021
    //  50: 0.105 ± 0.019
    if (multiPV != 1) {
      // Disable aspiration window if we are searching more than one principle variation.
      return search<TURN, SearchTypeRoot>(pos, depth, kMinEval, kMaxEval, RecommendedMoves(), true);
    }
    constexpr Evaluation kBuffer = 75;
    SearchResult<TURN> r = search<TURN, SearchTypeRoot>(pos, depth, lastResult.score - kBuffer, lastResult.score + kBuffer, RecommendedMoves(), true);
    if (r.score > lastResult.score - kBuffer && r.score < lastResult.score + kBuffer) {
      return r;
    }
    return search<TURN, SearchTypeRoot>(pos, depth, kMinEval, kMaxEval, RecommendedMoves(), true);
  }

  // Gives scores from white's perspective
  SearchResult<Color::WHITE> search(Position* pos, Depth depth, SearchResult<Color::WHITE> lastResult) {
    if (pos->turn_ == Color::WHITE) {
      return search_with_aspiration_window<Color::WHITE>(pos, depth, lastResult);
    } else {
      return flip(search_with_aspiration_window<Color::BLACK>(pos, depth, flip(lastResult)));
    }
  }

};


// TODO: there is a bug where
// "./a.out fen 8/8/8/1k6/3P4/8/8/3K4 w - - 0 1 depth 17"
// claims white is winning.

}  // namespace ChessEngine

#endif  // SEARCH_H