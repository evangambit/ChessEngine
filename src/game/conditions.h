#ifndef GAME_CONDITIONS_H_
#define GAME_CONDITIONS_H_

#include <memory>

#include "StopThinkingCondition.h"
#include "Thinker.h"

namespace ChessEngine {

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

}  // namespace ChessEngine

#endif  // GAME_CONDITIONS_H_