#import "game/search.h"
#import "game/Position.h"
#import "game/movegen.h"
#import "game/utils.h"
#import "game/string_utils.h"

#include <condition_variable>
#include <deque>
#include <iostream>
#include <mutex>
#include <unordered_set>
#include <unordered_map>

using namespace ChessEngine;

inline bool isCacheEntryEmpty(const CacheResult& cr) {
  return cr.positionHash == 0;
}

void merge(CacheResult* a, const CacheResult& b) {
  // TODO: be smarter
  *a = b;
}

struct Bound {
  Evaluation lowerbound;
  Evaluation upperbound;
  Bound() {}
  Bound(Evaluation lowerbound, Evaluation upperbound)
  : lowerbound(lowerbound), upperbound(upperbound) {}
  Bound(const CacheResult& cr) {
    lowerbound = cr.lowerbound();
    upperbound = cr.upperbound();
  }
  Bound flip() const {
    return Bound(-upperbound, -lowerbound);
  }
};

std::ostream& operator<<(std::ostream& stream, const Bound& bound) {
  if (bound.lowerbound == bound.upperbound) {
    return stream << "[" << bound.lowerbound << "]";
  }
  return stream << "[" << bound.lowerbound << ", " << bound.upperbound << "]";
}

struct SimpleTranspositionTable {
  CacheResult kEmptyCacheEntry;
  SimpleTranspositionTable() {
    size_ = 1'000'000;
    data_ = new CacheResult[size_];
    kEmptyCacheEntry = CacheResult{
      0,
      -99,  // depth is -99 so this is always a useless result
      0, kNullMove, NodeTypePV, 0 // these values should never matter
    };
  }
  ~SimpleTranspositionTable() {
    delete[] data_;
  }
  SimpleTranspositionTable(const SimpleTranspositionTable&) = delete;
  SimpleTranspositionTable& operator=(const SimpleTranspositionTable&) = delete;
  void insert(const CacheResult& entry) {
    const size_t index = entry.positionHash % size_;
    CacheResult *it = data_ + index;
    if (isCacheEntryEmpty(*it)) {
      *it = entry;
      return;
    }
    if (it->positionHash == entry.positionHash) {
      merge(it, entry);
      return;
    }
    if (it->depthRemaining <= entry.depthRemaining) {
      data_[index] = entry;
    }
  }
  void clear() {
    std::memset(data_, 0, sizeof(CacheResult) * size_);
  }
  CacheResult *find(uint64_t hash) {
    const size_t index = hash % size_;
    CacheResult *it = data_ + index;
    if (it->positionHash == hash) {
      return it;
    }
    return this->end();
  }
  CacheResult *end() {
    return &kEmptyCacheEntry;
  }
  CacheResult *data_;
  size_t size_;
};

std::string pad(int depth) {
  std::string r = "";
  for (int i = 0; i < depth; ++i) {
    r += "  ";
  }
  return r;
}

struct SimpleThinker {
  SimpleTranspositionTable transpositionTable_;
  PieceMaps pieceMaps_;
  Evaluator evaluator_;
  int multiPV_;
  SimpleThinker() {
    // std::ifstream myfile;
    // myfile.open("simpleweights.txt");
    // if (!myfile.is_open()) {
    //   std::cout << "Error opening file" << std::endl;
    //   exit(0);
    // }
    // this->pieceMaps_.load_weights_from_file(myfile);
    // myfile.close();

    multiPV_ = 1;

    std::ifstream myfile;
    myfile.open("weights.txt");
    if (!myfile.is_open()) {
      std::cout << "Error opening file" << std::endl;
      exit(0);
    }
    this->evaluator_.load_weights_from_file(myfile);
    this->pieceMaps_.load_weights_from_file(myfile);
    myfile.close();
  }

  std::vector<Move> last_n_moves(const Position& position, size_t n) {
    if (position.history_.size() < n) {
      throw std::runtime_error("");
    }
    std::vector<Move> r;
    for (size_t i = position.history_.size() - n; i < n; ++i) {
      r.push_back(position.history_[i].move);
    }
    return r;
  }

  template<Color US>
  Evaluation _qsearch(
    Position *position,
    uint16_t plyFromLeaf,
    uint16_t plyFromRoot,
    Evaluation alpha,
    Evaluation beta
    ) {
    std::string pad = "";
    for (int i = 0; i < plyFromRoot; ++i) {
      pad += "  ";
    }
    if (plyFromRoot > 100) {
      throw std::runtime_error("");
    }
    // std::cout << pad << "<qstart> " << alpha << " " << beta << " " << last_n_moves(*position, plyFromRoot) << std::endl;
    if (position->pieceBitboards_[coloredPiece<US, Piece::KING>()] == kEmptyBitboard) {
      // std::cout << pad << "</qend> mate " << kMissingKing << " + " << plyFromRoot - 1 << std::endl;
      return std::max(alpha, std::min<Evaluation>(beta, kQMissingKing));
    }

    // NOTE: Kind of a waste of time to check for 3-repetition or 50-move rule in qsearch.
    // Since we're mostly checking captures these are unlikely to happen.

    // There is no way we can do better than checkmate.
    if (alpha == -kQCheckmate) {
      return alpha;
    }

    // Evaluation r = std::max(alpha, this->eval<US>(*position));
    Evaluation r = std::max(alpha, this->evaluator_.score<US>(*position));
    if (r >= beta) {
      // std::cout << pad << "</qend> beta " << r << std::endl;
      return beta;
    }

    ExtMove moves[kMaxNumMoves];
    ExtMove *end;
    if (plyFromLeaf < 2) {
      end = compute_moves<US, MoveGenType::CHECKS_AND_CAPTURES>(*position, moves);
    } else {
      end = compute_moves<US, MoveGenType::CAPTURES>(*position, moves);
    }

    Threats<US> threats(*position);
    for (ExtMove *move = moves; move < end; ++move) {
      move->score = kQSimplePieceValues[move->capture];
      move->score -= value_or_zero(
        ((threats.badForOur[move->piece] & bb(move->move.to)) > 0) && !((threats.badForOur[move->piece] & bb(move->move.from)) > 0),
        kQSimplePieceValues[move->piece]
      );
    }
    std::sort(moves, end, [](ExtMove a, ExtMove b) {
      return a.score > b.score;
    });

    constexpr Color THEM = opposite_color<US>();

    for (ExtMove *move = moves; move < end; ++move) {
      if (move->score < 0) {
        break;
      }
      make_move<US>(position, move->move);
      Evaluation child = -this->_qsearch<THEM>(
        position, plyFromLeaf + 1, plyFromRoot + 1, -beta, -r);
      undo<US>(position);
      if (child < kQLongestForcedMate) {
        child += 1;
      } else if (-child > -kQLongestForcedMate) {
        child -= 1;
      }
      // std::cout << pad << move << " " << child << std::endl;
      if (child > r) {
        r = child;
        if (r >= beta) {
          break;
        }
      }
    }

    // std::cout << pad << "</qend> FIN " << r << std::endl;

    return std::max(alpha, std::min(beta, r));
  }

  template<Color US, bool IS_ROOT>
  Evaluation _search(
    Position *position,
    uint16_t plyFromRoot,
    uint16_t depthRemaining,
    Evaluation alpha,
    Evaluation beta
    ) {
    std::string pad = "";
    for (int i = 0; i < plyFromRoot; ++i) {
      pad += "  ";
    }
    // std::cout << pad << "<start> " << alpha << " " << beta << std::endl;
    if (position->pieceBitboards_[coloredPiece<US, Piece::KING>()] == kEmptyBitboard) {
      // std::cout << pad << "</end> mate " << kMissingKing << " + " << plyFromRoot - 1 << std::endl;
      return std::max(alpha, std::min<Evaluation>(beta, kMissingKing));
    }
    if (position->is_draw_assuming_no_checkmate()) {
      // std::cout << pad << "</end> draw " << std::endl;
      return std::max(alpha, std::min(beta, Evaluation(0)));
    }

    // There is no way we can do better than checkmate.
    if (alpha == -kCheckmate) {
      return alpha;
    }

    auto it = transpositionTable_.find(position->hash_);
    {
      if (it != transpositionTable_.end() && it->depthRemaining >= depthRemaining) {
        if (it->lowerbound() >= beta) {
          // std::cout << pad << "</end> tt beta " << it->eval << std::endl;
          return beta;
        }
        if (it->upperbound() <= alpha) {
          // std::cout << pad << "</end> tt alpha " << it->eval << std::endl;
          return alpha;
        }
        if (it->nodeType == NodeTypePV) {
          // std::cout << pad << "</end> tt pv " << it->eval << std::endl;
          return std::max(alpha, std::min(beta, it->eval));
        }
      }
    }

    if (depthRemaining == 0) {
      Evaluation r = this->_qsearch<US>(position, 0, plyFromRoot, alpha, beta);
      transpositionTable_.insert(CacheResult{
        position->hash_,
        0,
        r,
        kNullMove,
        NodeTypePV,
        0,  // todo; priority
        0   // todo; rootCounter
      });
      // std::cout << pad << "</end> leaf " << r << std::endl;
      return std::max(alpha, std::min(beta, r));
    }

    constexpr Color THEM = opposite_color<US>();

    // Compute moves.
    ExtMove moves[kMaxNumMoves];
    ExtMove *end = compute_legal_moves<US>(position, moves);

    if (moves == end) {
      constexpr ColoredPiece moverKing = coloredPiece<US, Piece::KING>();
      const bool inCheck = can_enemy_attack<US>(*position, safe_lsb(position->pieceBitboards_[moverKing]));
      if (inCheck) {
        Evaluation e = kCheckmate;
        // std::cout << pad << "</end> checkmate " << e << std::endl;
        return std::max(alpha, std::min(beta, e));
      }
      // std::cout << pad << "</end> draw" << std::endl;
      return std::max(alpha, std::min(beta, Evaluation(0)));
    }


    // Sort moves.
    for (ExtMove *move = moves; move != end; ++move) {
      move->score = kMoveOrderPieceValues[move->capture];
      move->score += value_or_zero(move->move == it->bestMove, 1000);
    }
    std::sort(moves, end, [](ExtMove a, ExtMove b) {
      return a.score > b.score;
    });

    std::vector<std::pair<Evaluation, Move>> evaluations;

    Evaluation r = std::max<Evaluation>(alpha, kCheckmate + plyFromRoot - 1); // "+0" ?
    Move bestMove = kNullMove;
    for (ExtMove *move = moves; move != end; ++move) {
      make_move<US>(position, move->move);
      // std::cout << pad << "move " << move->move << " (" << r << ")" << std::endl;
      Evaluation child = -this->_search<THEM, false>(
        position,
        plyFromRoot + 1,
        depthRemaining - 1,
        -beta,
        -r
      );
      undo<US>(position);
      if (child < kQLongestForcedMate) {
        child += 1;
      } else if (-child > -kQLongestForcedMate) {
        child -= 1;
      }
      if (IS_ROOT) {
        evaluations.push_back(std::make_pair(child, move->move));
        std::sort(evaluations.begin(), evaluations.end(), [](const std::pair<Evaluation, Move>& A, const std::pair<Evaluation, Move>& B) {
          return A.first > B.first;
        });
        if (evaluations.size() > multiPV_) {
          evaluations.pop_back();
        }
        if (evaluations.size() == multiPV_) {
          r = evaluations.back().first;
        }
      } else {
        if (child > r) {
          bestMove = move->move;
          r = child;
          if (r >= beta) {
            break;
          }
        }
      }
    }

    if (IS_ROOT) {
      bestMove = evaluations[0].second;
    }

    // std::cout << pad << "</end> FIN " << r << " " << bestMove << std::endl;

    r = std::max(alpha, std::min(beta, r));

    NodeType nodeType = NodeTypePV;
    if (r <= alpha) {
      nodeType = NodeTypeAll_UpperBound;
    } else if (r >= beta) {
      nodeType = NodeTypeCut_LowerBound;
    }

    transpositionTable_.insert(CacheResult{
      position->hash_,
      Depth(depthRemaining),
      r,
      bestMove,
      nodeType,
      0,  // todo; priority
      0   // todo; rootCounter
    });

    return r;
  }

  CacheResult search(Position *position, uint16_t depthRemaining) {
    transpositionTable_.clear();
    position->set_piece_maps(pieceMaps_);
    if (position->turn_ == Color::WHITE) {
      this->templated_search<Color::WHITE>(position, depthRemaining);
    } else {
      this->templated_search<Color::BLACK>(position, depthRemaining);
    }

    auto it = transpositionTable_.find(position->hash_);
    if (isCacheEntryEmpty(*it)) {
      if (position->is_draw_assuming_no_checkmate()) {
        return CacheResult{position->hash_, 0, 0, kNullMove, NodeTypePV, 0, 0};
      }
      throw std::runtime_error("empty");
    }
    if (it->nodeType != NodeTypePV) {
      throw std::runtime_error("not pv");
    }
    CacheResult r = *it;
    if (position->turn_ == Color::BLACK) {
      r.eval *= -1;
    }
    return r;
  }

  template<Color US>
  void templated_search(Position *position, uint16_t depthRemaining) {
    Evaluation e = this->_search<US, true>(position, 0, 1, kMinEval, kMaxEval);

    for (uint16_t depth = 2; depth <= depthRemaining; ++depth) {
      if (e <= kQLongestForcedMate || e >= -kQLongestForcedMate) {
        // No aspiration window when mates are involved.
        e = this->_search<US, true>(position, 0, depth, kMinEval, kMaxEval);
      } else {
        // Aspiration window
        Evaluation alpha = std::max<int32_t>(kMinEval, int32_t(e) - 300);
        Evaluation beta = std::min<int32_t>(kMaxEval, int32_t(e) + 300);

        e = this->_search<US, true>(position, 0, depth, alpha, beta);
        if (e <= alpha || e >= beta) {
          // We cannot only extend one time, because for some reason (todo: what reason?)
          // this can result in no pv entry in the TT.
          e = this->_search<US, true>(position, 0, depth, kMinEval, kMaxEval);
        }
      }

      std::deque<std::deque<std::pair<Move, Bound>>> primary_variations;
      ExtMove moves[kMaxNumMoves];
      ExtMove *end = compute_legal_moves<US>(position, moves);
      for (ExtMove *move = moves; move < end; ++move) {
        make_move<US>(position, move->move);
        std::deque<std::pair<Move, Bound>> variation;
        this->get_pv(position, &variation);
        undo<US>(position);
        if (variation.size() == 0) {
          continue;
        }
        if (variation[0].second.lowerbound == variation[0].second.upperbound) {
          variation.push_front(std::make_pair(
            move->move,
            variation[0].second
          ));
          primary_variations.push_back(variation);
        }
      }

      std::sort(primary_variations.begin(), primary_variations.end(), [](const std::deque<std::pair<Move, Bound>>& A, const std::deque<std::pair<Move, Bound>>& B) {
        return A[0].second.lowerbound > B[0].second.lowerbound;
      });
      while (primary_variations.size() > multiPV_) {
        primary_variations.pop_back();
      }
      for (const auto& variation : primary_variations) {
        std::cout << depth << " : " << variation << std::endl;
      }
    }
  }

  template<Color US>
  Evaluation eval(const Position& position) {
    constexpr Color THEM = opposite_color<US>();

    const Bitboard ourPawns = position.pieceBitboards_[coloredPiece<US, Piece::PAWN>()];
    const Bitboard ourKnights = position.pieceBitboards_[coloredPiece<US, Piece::KNIGHT>()];
    const Bitboard ourBishops = position.pieceBitboards_[coloredPiece<US, Piece::BISHOP>()];
    const Bitboard ourRooks = position.pieceBitboards_[coloredPiece<US, Piece::ROOK>()];
    const Bitboard ourQueens = position.pieceBitboards_[coloredPiece<US, Piece::QUEEN>()];
    const Bitboard ourKings = position.pieceBitboards_[coloredPiece<US, Piece::KING>()];

    const Bitboard theirPawns = position.pieceBitboards_[coloredPiece<THEM, Piece::PAWN>()];
    const Bitboard theirKnights = position.pieceBitboards_[coloredPiece<THEM, Piece::KNIGHT>()];
    const Bitboard theirBishops = position.pieceBitboards_[coloredPiece<THEM, Piece::BISHOP>()];
    const Bitboard theirRooks = position.pieceBitboards_[coloredPiece<THEM, Piece::ROOK>()];
    const Bitboard theirQueens = position.pieceBitboards_[coloredPiece<THEM, Piece::QUEEN>()];
    const Bitboard theirKings = position.pieceBitboards_[coloredPiece<THEM, Piece::KING>()];

    int32_t time = 24 - (
      std::popcount(ourKnights | theirKnights | ourBishops | theirBishops)
    + std::popcount(ourRooks | theirRooks) * 2
    + std::popcount(ourQueens | theirQueens) * 4);

    Evaluation r = 0;
    r += (std::popcount(ourPawns) - std::popcount(theirPawns)) * (100 * (24 - time) + 140 * time) / 24;
    r += (std::popcount(ourKnights) - std::popcount(theirKnights)) * (300 * (24 - time) + 300 * time) / 24;
    r += (std::popcount(ourBishops) - std::popcount(theirBishops)) * (300 * (24 - time) + 400 * time) / 24;
    r += (std::popcount(ourRooks) - std::popcount(theirRooks)) * (500 * (24 - time) + 700 * time) / 24;
    r += (std::popcount(ourQueens) - std::popcount(theirQueens)) * (900 * (24 - time) + 1200 * time) / 24;

    if (US == Color::WHITE) {
      r += int32_t(position.pieceMapScores[PieceMapType::PieceMapTypeEarly]) * (24 - time) / 24;
      r += int32_t(position.pieceMapScores[PieceMapType::PieceMapTypeLate]) * time / 24;
    } else {
      r -= int32_t(position.pieceMapScores[PieceMapType::PieceMapTypeEarly]) * (24 - time) / 24;
      r -= int32_t(position.pieceMapScores[PieceMapType::PieceMapTypeLate]) * time / 24;
    }

    return r;
  }

  template<Color US>
  void _get_pv(Position *position, std::deque<std::pair<Move, Bound>> *result) {
    auto it = transpositionTable_.find(position->hash_);
    if (it == transpositionTable_.end() || it->bestMove == kNullMove) {
      return;
    }
    if (it->eval == kCheckmate || it->eval == -kCheckmate) {
      return;
    }
    if (US == Color::WHITE) {
      result->push_back(std::make_pair(it->bestMove, Bound(*it)));
    } else {
      result->push_back(std::make_pair(it->bestMove, Bound(*it).flip()));
    }
    if (position->is_draw_assuming_no_checkmate()) {
      return;
    }
    make_move<US>(position, it->bestMove);
    _get_pv<opposite_color<US>()>(position, result);
    undo<US>(position);
  }

  void get_pv(Position *position, std::deque<std::pair<Move, Bound>> *result) {
    if (position->turn_ == Color::WHITE) {
      _get_pv<Color::WHITE>(position, result);
    } else {
      _get_pv<Color::BLACK>(position, result);
    }
  }
};

template<class F>
void stdin_loop(const F& f) {
  while (true) {
    if (std::cin.eof()) {
      break;
    }
    std::string line;
    getline(std::cin, line);
    if (line == "quit") {
      break;
    }

    // Skip empty lines.
    if(line.find_first_not_of(' ') == std::string::npos) {
      continue;
    }

    f(line);
  }
}

int main(int argc, char *argv[]) {
  initialize_geometry();
  initialize_zorbrist();
  initialize_movegen();
  SimpleThinker t;

  t.multiPV_ = 3;

  stdin_loop([&t](std::string line) {
    Position position(line);
    CacheResult r = t.search(&position, 10);
    std::cout << r.bestMove << std::endl;
  });

  // Position position("8/k7/3p4/p2P1p2/P2P1P2/8/8/K7 w - - 1 1");
  // std::cout << t.search(&position, 30) << std::endl;
  // ExtMove moves[kMaxNumMoves];
  // ExtMove *end;
  // if (position.turn_ == Color::WHITE) {
  //   end = compute_legal_moves<Color::WHITE>(&position, moves);
  // } else {
  //   end = compute_legal_moves<Color::BLACK>(&position, moves);
  // }
  // for (ExtMove *move = moves; move < end; ++move) {
  //   ez_make_move(&position, move->move);
  //   std::deque<std::pair<Move, Bound>> pv;
  //   t.get_pv(&position, &pv);
  //   std::cout << *move << " " << pv << std::endl;
  //   ez_undo(&position);
  // }

  // auto t0 = std::chrono::steady_clock::now();

  // Position position = Position::init();
  // for (size_t i = 0; i < 210; ++i) {
  //   auto result = t.search(&position, 7);
  //   // std::cout << " " << result.bestMove << " {" << result.eval << "}" << std::flush;
  //   std::cout << " " << result.bestMove << std::flush;
  //   if (result.eval == kCheckmate || result.eval == -kCheckmate || result.bestMove == kNullMove) {
  //     break;
  //   }
  //   if (position.turn_ == Color::WHITE) {
  //     make_move<Color::WHITE>(&position, result.bestMove);
  //   } else {
  //     make_move<Color::BLACK>(&position, result.bestMove);
  //   }
  //   if (position.is_draw_assuming_no_checkmate()) {
  //     break;
  //   }
  // }
  // std::cout << std::endl;

  // auto t1 = std::chrono::steady_clock::now();

  // const double secs = std::max(0.001, std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() / 1000.0);
  // std::cout << secs << std::endl;
}

