#ifndef SEARCH_RESULT_H
#define SEARCH_RESULT_H

#include "Position.h"
#include "utils.h"

namespace ChessEngine {

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
  return stream << "(" << sr.move << " " << sr.score << ")";
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

}  // namespace ChessEngine

#endif // SEARCH_RESULT_H
