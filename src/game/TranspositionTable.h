#ifndef TRANSPOSITION_TABLE_H
#define TRANSPOSITION_TABLE_H

#include "utils.h"
#include "Position.h"
#include "SpinLock.h"

namespace ChessEngine {

typedef int8_t Depth;

// PV-nodes ("principal variation" nodes) have a score that lies between alpha and beta; their scores are exact.
// Cut-nodes are nodes which contain a beta cutoff; score is a lower bound
// All-Nodes are nodes where no score exceeded alpha; score is an upper bound
enum NodeType {
  NodeTypeAll_UpperBound,
  NodeTypeCut_LowerBound,
  NodeTypePV,
};

std::string nodeTypeToString(NodeType nodeType) {
  switch (nodeType) {
    case NodeTypeAll_UpperBound:
      return "all";
    case NodeTypeCut_LowerBound:
      return "cut";
    case NodeTypePV:
      return "pv";
    default:
      return "??";
  }
}

const int32_t kMaxCachePriority = 8191;
const int32_t kNumRootCounters = 8;

struct CacheResult {  // 16 bytes
  uint64_t positionHash;  // 8 bytes
  Depth depthRemaining;   // 1 byte
  Evaluation eval;        // 2 bytes
  Move bestMove;          // 2 bytes
  NodeType nodeType;      // 1 byte
  unsigned priority : 13;      // 2 bytes; kMaxCachePriority if primary variation
  unsigned rootCounter : 3;
  inline Evaluation lowerbound() const {
    return (nodeType == NodeTypeAll_UpperBound) ? kMinEval : eval;
  }
  inline Evaluation upperbound() const {
    return (nodeType == NodeTypeCut_LowerBound) ? kMaxEval : eval;
  }
  static NodeType flip(NodeType nodeType) {
    switch (nodeType) {
      case NodeTypeAll_UpperBound:
        return NodeTypeCut_LowerBound;
      case NodeTypeCut_LowerBound:
        return NodeTypeAll_UpperBound;
      default:
        return NodeTypePV;
    }
  }
  CacheResult flip() const {
    return CacheResult{
      positionHash,
      depthRemaining,
      Evaluation(-eval),
      bestMove,
      flip(nodeType),
      priority,
      rootCounter,
    };
  }
};

// Transposition table is guaranteed to be a multiple of this.
constexpr int64_t kTranspositionTableFactor = 1024;
const CacheResult kMissingCacheResult = CacheResult{
  uint64_t(0),
  -99,  // depth is -99 so this is always a useless result
  0, kNullMove, NodeTypePV, 0 // these values should never matter
};
inline bool isNullCacheResult(const CacheResult& cr) {
  return cr.positionHash == 0;
}

std::ostream& operator<<(std::ostream& stream, CacheResult cr) {
  if (isNullCacheResult(cr)) {
    return stream << "[NULL]";
  }
  return stream << "[ hash:" << cr.positionHash << " depth:" << uint16_t(cr.depthRemaining) << " eval:[" << eval2str(cr.lowerbound()) << ", " << eval2str(cr.upperbound()) << "] move:" << cr.bestMove << " nodeType:" << nodeTypeToString(cr.nodeType) << " priority:" << unsigned(cr.priority) << " rootCounter:" << unsigned(cr.rootCounter) << " ]";
}

// TODO: it's possible for the same position to occur multiple times in this table.
// 1) Insert P1
// 2) Insert P2, which collides with P1, so it goes to it's second spot
// 3) make a move
// 4) Insert P2; it's first spot is free so it is inserted
//
// The saving grace is that it's second spot is still marked as obsolete, so it should be cleared eventually,
// but this kind of behavior seems really scary. I should a *least* write some tests for the transposition table,
// and possible refactor how it handles duplicates.

constexpr size_t kTranspositionTableMaxSteps = 3;
struct TranspositionTable {
  TranspositionTable(size_t kilobytes) : rootCounter(0), currentRootHash(0) {
    if (kilobytes < 1) {
      kilobytes = 1;
    }
    data = nullptr;
    this->set_cache_size(kilobytes);
  }
  TranspositionTable(const TranspositionTable&);  // Not implemented.
  TranspositionTable& operator=(const TranspositionTable& table);  // Not implemented.
  ~TranspositionTable() {
    delete[] data;
  }

  void starting_search(uint64_t rootHash) {
    if (rootHash != this->currentRootHash) {
      this->currentRootHash = rootHash;
      this->rootCounter = (this->rootCounter + 1) % kNumRootCounters;
    }
  }

  // For testing.
  void set_size_to_one() {
    size = 1;
    delete[] data;
    data = new CacheResult[size];
    this->clear();
  }

  void set_cache_size(size_t kilobytes) {
    size = kilobytes * 1024 / sizeof(CacheResult);
    size = size / kTranspositionTableFactor;
    if (size == 0) {
      size = 1;
    }
    size *= kTranspositionTableFactor;
    delete[] data;
    data = new CacheResult[size];
    this->clear();
  }

  uint64_t kb_size() const {
    return size * sizeof(CacheResult) / 1024;
  }

  SpinLock spinLocks[kTranspositionTableFactor];

  inline int age_of_entry(CacheResult *entry) {
    return (this->rootCounter - entry->rootCounter + kNumRootCounters) % kNumRootCounters;
  }

  template<bool IS_PARALLEL>
  void insert(const CacheResult& cr) {
    size_t idx = cr.positionHash % size;
    const size_t delta = (cr.positionHash >> 32) % 16;

    for (size_t i = 0; i < kTranspositionTableMaxSteps; ++i) {
      if (IS_PARALLEL) {
        spinLocks[idx % kTranspositionTableFactor].lock();
      }
      CacheResult *it = &data[idx];
      if (age_of_entry(it) > 0) {
        *it = cr;
      } else if (cr.positionHash == it->positionHash) {
        if (cr.depthRemaining > it->depthRemaining || (cr.nodeType == NodeTypePV && it->nodeType != NodeTypePV)) {
          *it = cr;
        }
        if (IS_PARALLEL) {
          spinLocks[idx % kTranspositionTableFactor].unlock();
        }
        return;
      } else if (
        // Forget any entry that wasn't used in the last search.
        age_of_entry(it) > 0
        // Replace shallow entries with deeper entries with a small penalty for older results.
        || cr.depthRemaining > it->depthRemaining
        // If entries are otherwise identical, prefer the one closer to the PV.
        || (cr.depthRemaining == it->depthRemaining && cr.priority > it->priority)) {
        *it = cr;
        if (IS_PARALLEL) {
          spinLocks[idx % kTranspositionTableFactor].unlock();
        }
        return;
      }
      if (IS_PARALLEL) {
        spinLocks[idx % kTranspositionTableFactor].unlock();
      }
      idx = (idx + delta) % size;
    }
  }
  void prefetch(uint64_t hash) {
    // TODO
    // _mm_prefetch((char *)&data[hash & size], _MM_HINT_NTA);
  }
  void clear() {
    std::fill_n((uint8_t *)data, sizeof(CacheResult) * size, 0);
    rootCounter = 1;  // Set this to one so all zeroed-out entries are considered old.
  }
  template<bool IS_PARALLEL>
  CacheResult find(uint64_t hash) {
    size_t idx = hash % size;
    const size_t delta = (hash >> 32) % 16;
    for (size_t i = 0; i < kTranspositionTableMaxSteps; ++i) {
      if (IS_PARALLEL) {
        spinLocks[idx % kTranspositionTableFactor].lock();
      }
      CacheResult *cr = &data[idx];
      if (cr->positionHash == hash) {
        cr->rootCounter = this->rootCounter;
        if (IS_PARALLEL) {
          spinLocks[idx % kTranspositionTableFactor].unlock();
        }
        return *cr;
      }
      if (IS_PARALLEL) {
        spinLocks[idx % kTranspositionTableFactor].unlock();
      }
      idx = (idx + delta) % size;
    }
    return kMissingCacheResult;
  }
  CacheResult unsafe_find(uint64_t hash) const {
    size_t idx = hash % size;
    const size_t delta = (hash >> 32) % 16;
    for (size_t i = 0; i < kTranspositionTableMaxSteps; ++i) {
      CacheResult *cr = &data[idx];
      if (cr->positionHash == hash) {
        return *cr;
      }
      idx = (idx + delta) % size;
    }
    return kMissingCacheResult;
  }

  CacheResult create_cache_result(
    uint64_t hash,
    Depth depthRemaining,
    Evaluation eval,
    Move bestMove,
    NodeType nodeType,
    int32_t distFromPV) {
    return CacheResult{
      hash,
      depthRemaining,
      eval,
      bestMove,
      nodeType,
      uint8_t(std::max(0, kMaxCachePriority - distFromPV)),
      this->rootCounter,
    };
  }
 private:
  uint64_t currentRootHash;
  unsigned rootCounter;
  CacheResult *data;
  size_t size;
};

}  // namespace ChessEngine

#endif  // TRANSPOSITION_TABLE_H