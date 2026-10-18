#pragma once
#include <cstddef>
struct esp_partition_t {};
#define ESP_OK 0
inline int esp_partition_read(const esp_partition_t*, size_t, void*, size_t) { return -1; }
