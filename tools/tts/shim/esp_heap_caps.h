#pragma once
#include <cstdlib>
#define MALLOC_CAP_SPIRAM 1
#define MALLOC_CAP_8BIT 2
extern size_t host_allocated_bytes;
inline void* heap_caps_malloc(size_t n, unsigned) {
  void* p = std::malloc(n);
  if (p) host_allocated_bytes += n;
  return p;
}
