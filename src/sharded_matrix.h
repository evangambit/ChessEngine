#include <cstdint>

#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

namespace ShardedMatrix {

namespace {

std::string path2shardname(const std::string& path, int i) {
  std::string n = std::to_string(i);
  return path + "-" + std::string(5 - n.size(), '0') + n + ".sm";
}

uint32_t product(const std::vector<uint32_t>& A) {
  uint32_t p = 1;
  for (uint32_t d : A) {
    p *= d;
  }
  return p;
}

/** Targeted shard size (25 MB). */
const uint32_t kShardSizeBytes = 25'000'000;

/**
 * An internal class with a 1-to-1 relationship with a shard file.
 */
template<class T>
class ShardWriter {
public:
  ShardWriter(const std::string shardpath, const std::vector<uint32_t>& dims)
  : path_(shardpath), dims_(dims), rowCounter_(0),
  bytesPerRow(product(dims) * sizeof(T)),
  rowsPerShard(kShardSizeBytes / (product(dims) * sizeof(T))) {
    if (this->rowsPerShard == 0) {
      throw std::runtime_error("Shard size is too small");
    }
    this->_init_file();
  }

  ShardWriter(const ShardWriter&) = delete;
  ShardWriter& operator=(const ShardWriter&) = delete;
  ~ShardWriter() {
    this->_close();
  }

  /**
   * Write a row to the shard.
   * 
   * @param row The row to write.
   * @return Is the shard is full?
   */
  bool write_row(const T *const row) {
    this->_write_row(row);
    return ++rowCounter_ >= this->rowsPerShard;
  }

private:
  void _init_file() {
    file_.open(path_, std::ios::binary | std::ios::out);
    const std::string typeCode = ShardWriter<T>::_type_code();
    const uint32_t numDims = dims_.size() + 1;
    file_.write((char *)(typeCode.data()), 4);
    file_.write((char *)(&numDims), sizeof(uint32_t));
    file_.write((char *)(&rowCounter_), sizeof(uint32_t));
    file_.write((char *)(dims_.data()), dims_.size() * sizeof(uint32_t));
  }

  void _write_row(const T *const row) {
    file_.write((char *)(row), this->bytesPerRow);
  }

  /** Returns the unique 4-byte string that identifies the type "T" */
  static std::string _type_code();

  void _close() {
    file_.seekp(8);
    file_.write((const char*)(&rowCounter_), sizeof(uint32_t));
    file_.close();
  }

  const std::string path_;
  const std::vector<uint32_t> dims_;
  const size_t bytesPerRow;
  const size_t rowsPerShard;
  uint32_t rowCounter_;
  std::ofstream file_;
};

/** We override some boolean methods since we'd like to pack 8 bools in a byte. */
template<>
ShardWriter<bool>::ShardWriter(const std::string path, const std::vector<uint32_t>& dims)
  : path_(path), dims_(dims), rowCounter_(0),
  bytesPerRow(product(dims) / 8),
  rowsPerShard(kShardSizeBytes / (product(dims) / 8)) {
  if (this->rowsPerShard == 0) {
    throw std::runtime_error("Shard size is too small");
  }
  if (product(dims) % 8 != 0) {
    throw std::runtime_error("Boolean array size must be a multiple of 8");
  }
  this->_init_file();
}

template<>
void ShardWriter<bool>::_write_row(const bool *const row) {
  // Write to file
  int8_t *A = new int8_t[this->bytesPerRow];
  std::fill_n(A, this->bytesPerRow, 0);
  for (int i = 0; i < this->bytesPerRow; ++i) {
    A[i] = row[8 * i + 0] << 0 | row[8 * i + 1] << 1 | row[8 * i + 2] << 2 | row[8 * i + 3] << 3 | row[8 * i + 4] << 4 | row[8 * i + 5] << 5 | row[8 * i + 6] << 6 | row[8 * i + 7] << 7;
  }
  file_.write((const char*)A, this->bytesPerRow);
  delete[] A;
}


template<>
std::string ShardWriter<float>::_type_code() {
  return "f32 ";
}

template<>
std::string ShardWriter<int8_t>::_type_code() {
  return "i8  ";
}

template<>
std::string ShardWriter<int16_t>::_type_code() {
  return "i16  ";
}

template<>
std::string ShardWriter<bool>::_type_code() {
  return "bool";
}

template<class T>
class WriterDelegate {
public:
  WriterDelegate(const std::string& filename) : filename_(filename), shardCounter_(-1) {}

  std::string new_shard_path() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::string result = path2shardname(filename_, ++shardCounter_);
    return result;
  }
private:
  std::string filename_;
  uint32_t shardCounter_;
  std::mutex mutex_;
};

}  // namespace

template<class T>
class WriterManager;

/**
 * An external class that manages the writing to a sharded matrix.
 * 
 * When one shard is filled, Writer requests a new shard automatically.
 */
template<class T>
class Writer {
public:
  friend WriterManager<T>;

  /** Public constructor that can be used in single-threaded contexts. */
  Writer(const std::string filename, const std::vector<uint32_t>& dims)
  : dims_(dims), doesOwnDelegate_(true) {
    delegate_ = new WriterDelegate<T>(filename);
    internalWriter_ = std::make_unique<ShardWriter<T>>(delegate_->new_shard_path(), dims);
  }

  void write_row(const T *const row) {
    if (this->internalWriter_->write_row(row)) {
      // Create new writer
      this->internalWriter_.reset(new ShardWriter<T>(delegate_->new_shard_path(), dims_));
    }
  }

  ~Writer() {
    if (doesOwnDelegate_) {
      delete delegate_;
    }
  }

private:
  /** Private constructor used by WriterManager */
  Writer(const std::vector<uint32_t>& dims, WriterDelegate<T> *delegate)
  : dims_(dims), internalWriter_(new ShardWriter<T>(delegate->new_shard_path(), dims)), delegate_(delegate), doesOwnDelegate_(false) {}

  std::vector<uint32_t> dims_;
  std::unique_ptr<ShardWriter<T>> internalWriter_;
  WriterDelegate<T> *delegate_;
  bool doesOwnDelegate_;
};

/**
 * Manages multiple Writers for a single sharded matrix. Useful when there are
 * multiple threads.
 */
template<class T>
class WriterManager {
public:
  WriterManager(const std::string& filename, const std::vector<uint32_t>& dims)
  : writerDelegate_(filename), dims_(dims), shardCounter_(-1) {}

  /** Each thread should call this to get its own writer. */
  std::shared_ptr<Writer<T>> new_shard_writer() {
    return std::shared_ptr<Writer<T>>(new Writer<T>(dims_, &(this->writerDelegate_)));
  }
private:
  const std::vector<uint32_t> dims_;
  uint32_t shardCounter_;
  WriterDelegate<T> writerDelegate_;
};

}  // namespace ShardedMatrix