#ifndef STOP_THINKING_CONDITION_H
#define STOP_THINKING_CONDITION_H

namespace ChessEngine {

struct ThinkerInterface;
struct StopThinkingCondition {
  virtual void start_thinking(const ThinkerInterface& thinker) = 0;
  virtual bool should_stop_thinking(const ThinkerInterface&) = 0;
  virtual ~StopThinkingCondition() = default;
};

struct NeverStopThinkingCondition : public StopThinkingCondition {
  void start_thinking(const ThinkerInterface& thinker) {}
  bool should_stop_thinking(const ThinkerInterface&) {
    return false;
  }
};

}  // namespace ChessEngine

#endif