#ifndef STOP_THINKING_CONDITION_H
#define STOP_THINKING_CONDITION_H

namespace ChessEngine {

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

}  // namespace ChessEngine

#endif