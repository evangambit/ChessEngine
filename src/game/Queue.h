#ifndef QUEUE_H
#define QUEUE_H

#include <queue>
#include <mutex>
#include <condition_variable>

template <class T>
class Queue {
 public:
  void push(std::shared_ptr<T> item) {
    std::lock_guard<std::mutex> lock(mutex);
    queue.push(item);
    cond.notify_one();
  }

  std::shared_ptr<T> pop() {
  	this->wait();
    std::shared_ptr<T> result = queue.front();
    queue.pop();
    return result;
  }

  void wait() {
    std::unique_lock<std::mutex> lock(mutex);
    while(queue.empty()) {
      cond.wait(lock);
    }
  }

private:
  std::queue<std::shared_ptr<T>> queue;
  mutable std::mutex mutex;
  std::condition_variable cond;
};

#endif