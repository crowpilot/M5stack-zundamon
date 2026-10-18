#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdarg>
using std::min;
#define PI 3.1415926535897932384626433832795
struct HostSerial {
  void println(const char* s) { std::fprintf(stderr, "%s\n", s); }
  void printf(const char* fmt, ...) {
    va_list ap; va_start(ap, fmt); std::vfprintf(stderr, fmt, ap); va_end(ap);
  }
};
extern HostSerial Serial;
