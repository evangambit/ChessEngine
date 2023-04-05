#ifndef SEARCH_H
#define SEARCH_H

#include <cassert>
#include <cstdint>

#include <algorithm>
#include <unordered_map>

#ifndef NDEBUG
#include <iostream>
#include <string>
#include <vector>
#endif

#include "geometry.h"
#include "utils.h"
#include "Position.h"
#include "movegen.h"
#include "movegen/sliding.h"
#include "Evaluator.h"

namespace ChessEngine {

typedef int8_t Depth;

#ifndef NDEBUG
std::vector<std::string> gStackDebug;
#endif

// PV-nodes ("principal variation" nodes) have a score that lies between alpha and beta; their scores are exact.

// Cut-nodes are nodes which contain a beta cutoff; score is a lower bound

// All-Nodes are nodes where no score exceeded alpha; score is an upper bound
enum NodeType {
  NodeTypeAll_UpperBound,
  NodeTypeCut_LowerBound,
  NodeTypePV,
};

struct CacheResult {
  Depth depth;
  Evaluation eval;
  Move bestMove;
  NodeType nodeType;
  #ifndef NDEBUG
  std::string fen;
  #endif
  inline Evaluation lowerbound() const {
    return nodeType == NodeTypeAll_UpperBound ? kMinEval : eval;
  }
  inline Evaluation upperbound() const {
    return nodeType == NodeTypeAll_UpperBound ? kMaxEval : eval;
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

    if (depth > 8) {
      Evaluation e = this->evaluator.score<TURN>(*pos);
      return SearchResult<TURN>(e, kNullMove);
    }

    constexpr Color opposingColor = opposite_color<TURN>();
    constexpr ColoredPiece moverKing = coloredPiece<TURN, Piece::KING>();
    const bool inCheck = can_enemy_attack<TURN>(*pos, lsb(pos->pieceBitboards_[moverKing]));

    ExtMove moves[kMaxNumMoves];
    ExtMove *end = compute_moves<TURN, MoveGenType::CHECKS_AND_CAPTURES>(*pos, moves);

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

  template<Color TURN, bool isRoot>
  SearchResult<TURN> search(
    Position* pos, const Depth depth,
    Evaluation alpha, const Evaluation beta,
    RecommendedMoves recommendedMoves,
    bool isPV) {

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
      std::cout << "a" << std::endl;
      return SearchResult<TURN>(kMissingKing, kNullMove);
    }

    if (depth <= 0) {
      ++this->leafCounter;
      // Quiescence Search (+0.47)
      SearchResult<TURN> r = qsearch<TURN>(pos, 0, alpha, beta);

      const CacheResult cr = CacheResult{
        depth,
        r.score,
        r.move,
        NodeTypePV,
        #ifndef NDEBUG
        pos->fen(),
        #endif
      };
      auto it = this->cache.find(pos->hash_);
      if (this->cache.find(pos->hash_) == this->cache.end()) {
        this->cache.insert(std::make_pair(pos->hash_, cr));
      }
      return r;
    }

    if (pos->is_draw()) {
      SearchResult<TURN>(Evaluation(0), kNullMove);
    }

    auto it = this->cache.find(pos->hash_);
    if (it != this->cache.end() && it->second.depth >= depth) {
      const CacheResult& cr = it->second;
      if (cr.nodeType == NodeTypePV || cr.lowerbound() >= beta) {
        return SearchResult<TURN>(cr.eval, cr.bestMove);
      }
    }

    // // Futility pruning (+0.10)
    // // If our depth is 1 or exceeds the transposition table by 1 then we ignore moves that
    // // are sufficiently bad that they are unlikely to be improved by increasing depth by 1.
    // const int numMenLeft = std::popcount(pos->colorBitboards_[Color::WHITE] | pos->colorBitboards_[Color::BLACK]);
    // const Evaluation futilityThreshold = 70;
    // if (it != this->cache.end() && depth - it->second.depth == 1) {
    //   const CacheResult& cr = it->second;
    //   if (cr.lowerbound() >= beta + futilityThreshold || cr.upperbound() <= alpha - futilityThreshold) {
    //     return SearchResult<TURN>(cr.eval, cr.bestMove);
    //   }
    // }
    // if (it == this->cache.end() && depth <= 2) {
    //   // We assume that you can make a move that improves your position, so comparing against alpha
    //   // gets a small bonus.
    //   const Evaluation tempoBonus = 20;
    //   SearchResult<TURN> r = qsearch<TURN>(pos, 0, alpha, beta);
    //   if (r.score >= beta + futilityThreshold || r.score + tempoBonus <= alpha - futilityThreshold) {
    //     return r;
    //   }
    // }

    Bitboard ourPieces = pos->colorBitboards_[TURN] & ~pos->pieceBitboards_[coloredPiece<TURN, Piece::PAWN>()];
    const bool inCheck = can_enemy_attack<TURN>(*pos, lsb(pos->pieceBitboards_[moverKing]));

    // // Null Move Pruning (+0.016)
    // if (depth >= 3 && std::popcount(ourPieces) > 1 && !inCheck && !isPV) {
    //   make_nullmove<TURN>(pos);
    //   SearchResult<TURN> a = flip(search<opposingColor, false>(pos, depth - 3, -beta, -alpha, RecommendedMoves(), false));
    //   if (a.score >= beta && a.move != kNullMove) {
    //     undo_nullmove<TURN>(pos);
    //     return SearchResult<TURN>(beta + 1, kNullMove);
    //   }
    //   undo_nullmove<TURN>(pos);
    // }

    Move lastFoundBestMove = (it != this->cache.end() ? it->second.bestMove : kNullMove);

    #ifndef NDEBUG
    std::string fen0 = pos->fen();
    #endif

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

    RecommendedMoves recommendationsForChildren;

    NodeType nodeType = NodeTypePV;

    // Should be optimized away if !isRoot.
    std::vector<SearchResult<TURN>> children;

    size_t numValidMoves = 0;
    for (ExtMove *extMove = moves; extMove < end; ++extMove) {

      #ifndef NDEBUG
      gStackDebug.push_back(extMove->uci());
      pos->assert_valid_state("a " + extMove->uci());
      #endif

      #ifndef NDEBUG
      const size_t h0 = pos->hash_;
      #endif

      make_move<TURN>(pos, extMove->move);

      // Don't move into check.
      if (can_enemy_attack<TURN>(*pos, lsb(pos->pieceBitboards_[moverKing]))) {
        undo<TURN>(pos);
        continue;
      }

      ++numValidMoves;

      SearchResult<TURN> a = flip(search<opposingColor, false>(pos, depth - 1, -beta, -alpha, recommendationsForChildren, isPV && (extMove == moves)));

      a.score -= (a.score > -kLongestForcedMate);
      a.score += (a.score < kLongestForcedMate);

      undo<TURN>(pos);

      #ifndef NDEBUG
      const size_t h1 = pos->hash_;
      if (h0 != h1) {
        throw std::runtime_error("h0 != h1");
      }
      #endif

      #ifndef NDEBUG
      gStackDebug.pop_back();
      if (pos->fen() != fen0) {
        std::cout << fen0 << std::endl;
        std::cout << pos->fen() << std::endl;
        throw std::runtime_error("fen != fen0");
      }
      pos->assert_valid_state("b " + extMove->uci());
      #endif

      if (isRoot) {
        children.push_back(a);
        std::sort(children.begin(), children.end());
        if (a.score > r.score) {
          r.score = a.score;
          r.move = extMove->move;
          recommendationsForChildren.add(a.move);
        }
        if (children.size() >= multiPV) {
          alpha = std::max(alpha, children[0].score);
        }
      } else {
        if (a.score > r.score) {
          r.score = a.score;
          r.move = extMove->move;
          recommendationsForChildren.add(a.move);
          if (r.score >= beta) {
            nodeType = NodeTypeCut_LowerBound;
            this->historyHeuristicTable[TURN][r.move.from][r.move.to] += depth * depth;
            break;
          }
          if (r.score > alpha) {
            alpha = r.score;
          }
        }
      }
    }

    if (r.score <= alpha) {
      nodeType = NodeTypeAll_UpperBound;
    }

    if (numValidMoves == 0) {
      r.score = inCheck ? kCheckmate : 0;
      r.move = kNullMove;
    }

    {
      const CacheResult cr = CacheResult{
        depth,
        r.score,
        r.move,
        nodeType,
        #ifndef NDEBUG
        pos->fen(),
        #endif
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
      return search<TURN, true>(pos, depth, kMinEval, kMaxEval, RecommendedMoves(), true);
    }
    constexpr Evaluation kBuffer = 75;
    SearchResult<TURN> r = search<TURN, true>(pos, depth, lastResult.score - kBuffer, lastResult.score + kBuffer, RecommendedMoves(), true);
    if (r.score > lastResult.score - kBuffer && r.score < lastResult.score + kBuffer) {
      return r;
    }
    return search<TURN, true>(pos, depth, kMinEval, kMaxEval, RecommendedMoves(), true);
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