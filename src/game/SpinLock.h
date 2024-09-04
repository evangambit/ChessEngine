#ifndef SPINLOCK_H
#define SPINLOCK_H

#include <atomic>

namespace ChessEngine {

struct SpinLock {
  std::atomic<bool> lock_ = {false};
  void lock() { while(lock_.exchange(true)); }
  void unlock() { lock_.store(false); }
};

}  // namespace ChessEngine

#endif