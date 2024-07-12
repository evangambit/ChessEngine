/**
 * Header to write sharded matrices. See "sharded_matrix.py"
 * Source: https://github.com/evangambit/sharded-matrix
 */

#include <iostream>
#include <fstream>
#include <cstdint>
#include <vector>

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

const uint32_t kShardSizeBytes = 25'000'000;

template<class T>
class ShardedWriter {
public:
  const std::string filename;
  const std::vector<uint32_t> dims;
  const size_t bytesPerRow;
  const size_t rowsPerShard;
  uint32_t shardCounter;
  uint32_t rowCounter;
  std::ofstream file;
  ShardedWriter(const std::string& filename, const std::vector<uint32_t>& dims)
  : filename(filename), dims(dims),
    shardCounter(-1), rowCounter(0),
    bytesPerRow(product(dims) * sizeof(T)),
    rowsPerShard(kShardSizeBytes / (product(dims) * sizeof(T))) {
    if (this->rowsPerShard == 0) {
      throw std::runtime_error("Shard size is too small");
    }
    this->_init_file();
  }

  static std::string type_code();
  
  void write_row(const T *const row) {
    this->_write_row(row);
    if (++rowCounter >= this->rowsPerShard) {
      this->_close();
      rowCounter = 0;
      this->_init_file();
    }
  }

  void _init_file() {
    std::string path = path2shardname(filename, ++shardCounter);
    file.open(path, std::ios::binary | std::ios::out);
    const std::string typeCode = ShardedWriter<T>::type_code();
    const uint32_t numDims = dims.size() + 1;
    file.write((char *)(typeCode.data()), 4);
    file.write((char *)(&numDims), sizeof(uint32_t));
    file.write((char *)(&rowCounter), sizeof(uint32_t));
    file.write((char *)(dims.data()), dims.size() * sizeof(uint32_t));
  }

  ~ShardedWriter() {
    this->_close();
  }

  void _close() {
    file.seekp(8);
    file.write((const char*)(&rowCounter), sizeof(uint32_t));
    file.close();
    rowCounter = 0;
  }

  void _write_row(const T *const row) {
    file.write((char *)(row), this->bytesPerRow);
  }

  void close() {
    // Update row count
    this->_close();
  }
};

template<>
std::string ShardedWriter<float>::type_code() {
  return "f32 ";
}

template<>
std::string ShardedWriter<int8_t>::type_code() {
  return "i8  ";
}

template<>
std::string ShardedWriter<int16_t>::type_code() {
  return "i16  ";
}

template<>
std::string ShardedWriter<bool>::type_code() {
  return "bool";
}

template<>
ShardedWriter<bool>::ShardedWriter(const std::string& filename, const std::vector<uint32_t>& dims)
  : filename(filename), dims(dims),
  shardCounter(-1), rowCounter(0),
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
void ShardedWriter<bool>::_write_row(const bool *const row) {
  // Write to file
  int8_t *A = new int8_t[this->bytesPerRow];
  std::fill_n(A, this->bytesPerRow, 0);
  for (int i = 0; i < this->bytesPerRow; ++i) {
    A[i] = row[8 * i + 0] << 0 | row[8 * i + 1] << 1 | row[8 * i + 2] << 2 | row[8 * i + 3] << 3 | row[8 * i + 4] << 4 | row[8 * i + 5] << 5 | row[8 * i + 6] << 6 | row[8 * i + 7] << 7;
  }
  file.write((const char*)A, this->bytesPerRow);
}
