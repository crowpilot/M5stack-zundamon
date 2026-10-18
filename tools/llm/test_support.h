#pragma once
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "llm_core.h"

static void *read_binary(const char *path, size_t *len) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    assert(fseek(f, 0, SEEK_END) == 0);
    long size = ftell(f);
    assert(size > 0);
    *len = (size_t)size;
    rewind(f);
    void *buf = malloc(*len);
    assert(buf && fread(buf, 1, *len, f) == *len);
    fclose(f);
    return buf;
}

static void init_test_llm(Llm *m, const void *model, size_t mn,
                          const void *tok, size_t tn, int steps) {
    assert(llm_init(m, model, mn, tok, tn, steps, malloc) == 0);
}
