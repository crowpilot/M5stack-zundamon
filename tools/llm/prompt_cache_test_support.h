#pragma once
// Same-host exact comparison; target-generated snapshots need a separate target check.
#include "test_support.h"
#include "zunda_brain.h"

typedef struct {
    unsigned char bytes[262144];
    size_t len;
    int vocab, prepared, delivered;
    char pending[512];
} Trace;

static void prepare(const char *piece, void *ctx) {
    Trace *t = ctx;
    assert(t->prepared == t->delivered);
    assert(strlen(piece) < sizeof(t->pending));
    strcpy(t->pending, piece);
    t->prepared++;
}

static void piece(const char *text, const float *logits, void *ctx) {
    Trace *t = ctx;
    if (t->prepared) {
        assert(t->prepared == t->delivered + 1);
        assert(strcmp(text, t->pending) == 0);
    }
    size_t text_n = strlen(text) + 1, logits_n = (size_t)t->vocab * sizeof(float);
    assert(t->len + text_n + logits_n <= sizeof(t->bytes));
    memcpy(t->bytes + t->len, text, text_n); t->len += text_n;
    memcpy(t->bytes + t->len, logits, logits_n); t->len += logits_n;
    t->delivered++;
}

static void replies_equal(Llm *ref, Llm *cached, const char *heard, int source) {
    static Trace a, b;
    memset(&a, 0, sizeof(a)); memset(&b, 0, sizeof(b));
    a.vocab = b.vocab = ref->config.vocab_size;
    char ra[256], rb[256]; int fa, fb;
    int na = zunda_brain_reply(ref, heard, ra, sizeof(ra), &fa, piece, &a);
    int nb = zunda_brain_reply_ex(cached, heard, rb, sizeof(rb), &fb, prepare, piece, &b);
    assert(na == nb && fa == fb && strcmp(ra, rb) == 0);
    assert(ref->rng == cached->rng && a.len == b.len);
    assert(memcmp(a.bytes, b.bytes, a.len) == 0);
    assert(a.delivered == b.delivered && b.prepared == b.delivered);
    assert(llm_last_prefill_source(cached) == source);
}

static void prefill_equal(Llm *ref, Llm *cached, const int *tokens, int n, int source) {
    float *a = llm_prefill(ref, tokens, n, NULL);
    float *b = llm_prefill(cached, tokens, n, NULL);
    assert(a && b && memcmp(a, b, (size_t)ref->config.vocab_size * sizeof(float)) == 0);
    assert(llm_last_prefill_source(cached) == source);
    int kv = ref->config.dim * ref->config.n_kv_heads / ref->config.n_heads;
    for (int l = 0; l < ref->config.n_layers; l++) {
        size_t off = (size_t)l * ref->max_steps * kv, len = (size_t)n * kv * sizeof(float);
        assert(memcmp(ref->state.key_cache + off, cached->state.key_cache + off, len) == 0);
        assert(memcmp(ref->state.value_cache + off, cached->state.value_cache + off, len) == 0);
    }
}
