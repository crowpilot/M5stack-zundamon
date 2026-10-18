#pragma once
#include <cstddef>
#include <cstdint>
namespace fs {
struct File {
  explicit operator bool() const { return false; }
  void seek(size_t) {}
  size_t read(uint8_t*, size_t) { return 0; }
};
struct FS { File open(const char*, const char*) { return File(); } };
}
