#include "prompt_cache_test_support.h"

typedef struct { unsigned char *data; size_t cap, len; } Sink;
static int write_bytes(const void *data, size_t n, void *ctx) {
    Sink *s = ctx;
    if (n > s->cap - s->len) return -1;
    memcpy(s->data + s->len, data, n); s->len += n;
    return 0;
}

static LlmPromptSnapshot exported(Llm *m, const char *heard, int skip, const unsigned char identity[32]) {
    size_t size = zunda_brain_prompt_snapshot_size(m, heard, skip);
    assert(size > 96);
    Sink s = {malloc(size), size, 0}; assert(s.data);
    unsigned long long rng = m->rng;
    assert(zunda_brain_export_prompt(m, heard, skip, identity, write_bytes, &s) == 0);
    assert(s.len == size && m->rng == rng);
    return (LlmPromptSnapshot){s.data, size};
}

int main(int argc, char **argv) {
    assert(argc == 3);
    size_t mn, tn; void *model = read_binary(argv[1], &mn), *tok = read_binary(argv[2], &tn);
    Llm ref, snapshot, producer;
    init_test_llm(&ref, model, mn, tok, tn, 64);
    init_test_llm(&snapshot, model, mn, tok, tn, 64);
    init_test_llm(&producer, model, mn, tok, tn, 256);
    assert(zunda_brain_init(&ref) == 0 && zunda_brain_init(&snapshot) == 0 && zunda_brain_init(&producer) == 0);
    const unsigned char identity[32] = {1, 2, 3, 4}, other[32] = {9};
    const char *hello = "こんにちは。";
    LlmPromptSnapshot full = exported(&producer, hello, 0, identity);
    LlmPromptSnapshot suffix = exported(&producer, hello, 2, identity);
    assert(full.size - suffix.size == 2 * 2 * 6 * 48 * 4);
    int toks[256], n; llm_encode(&ref, "Ａ：こんにちは。Ｂ：", 1, 0, toks, &n);
    assert(llm_prompt_snapshot_import(&snapshot, &full, toks, n, other) == -1);
    int changed[256]; memcpy(changed, toks, (size_t)n * sizeof(int)); changed[n - 1]++;
    assert(llm_prompt_snapshot_import(&snapshot, &full, changed, n, identity) == 0);
    llm_invalidate_prompt_cache(&snapshot);
    assert(llm_prompt_snapshot_import(&snapshot, &suffix, toks, n, identity) == 0);
    assert(llm_prompt_snapshot_import(&snapshot, &full, toks, n, identity) == 1);
    assert(llm_prompt_snapshot_import(&snapshot, &suffix, toks, n, identity) == 1);
    // Public import verifies CRC before mutating state; unaligned Flash data is safe.
    unsigned char *bad = malloc(full.size + 1); assert(bad);
    memcpy(bad + 1, full.data, full.size);
    LlmPromptSnapshot damaged = {bad + 1, full.size};
    assert(llm_prompt_snapshot_import(&snapshot, &damaged, toks, n, identity) == 1);
    bad[full.size] ^= 1;
    assert(llm_prompt_snapshot_import(&snapshot, &damaged, toks, n, identity) == -1);
    assert(llm_set_prompt_snapshots(&snapshot, &damaged, 1, identity, 1) == -1);
    assert(!snapshot.snapshots_enabled);
    memcpy(bad + 1, full.data, full.size); bad[1] ^= 1;
    assert(llm_prompt_snapshot_import(&snapshot, &damaged, toks, n, identity) == -1);
    damaged.size = 95;
    assert(llm_prompt_snapshot_import(&snapshot, &damaged, toks, n, identity) == -1);
    assert(llm_prompt_snapshot_size(&snapshot, n, 1) == 0);
    assert(llm_prompt_snapshot_size(&snapshot, 65, 0) == 0);
    Sink too_small = {bad, 1, 0};
    assert(zunda_brain_export_prompt(&producer, hello, 2, identity, write_bytes, &too_small) == -2);
    assert(llm_set_prompt_snapshots(&snapshot, &suffix, 1, identity, 1) == 0);
    llm_invalidate_prompt_cache(&snapshot);
    prefill_equal(&ref, &snapshot, toks, n, 0);  // no shared prefix -> normal compute
    prefill_equal(&ref, &snapshot, toks, n, 2);  // shared prefix now exists
    assert(llm_set_prompt_cache(&snapshot, 1) == 0);
    prefill_equal(&ref, &snapshot, toks, n, 2);
    prefill_equal(&ref, &snapshot, toks, n, 1);  // last-input cache has priority
    assert(llm_set_prompt_cache(&snapshot, 0) == 0);
    assert(llm_set_prompt_snapshots(&snapshot, &suffix, 1, identity, 0) == 0);
    prefill_equal(&ref, &snapshot, toks, n, 0);

    LlmPromptSnapshot entries[64]; char inputs[64][1024]; int count = 0;
    // Use matching max_steps for prompt truncation when producing long-input snapshots.
    init_test_llm(&producer, model, mn, tok, tn, 64);
    assert(zunda_brain_init(&producer) == 0);
    while (count < 64 && fgets(inputs[count], sizeof(inputs[count]), stdin)) {
        inputs[count][strcspn(inputs[count], "\n")] = 0;
        entries[count] = exported(&producer, inputs[count], 2, identity);
        count++;
    }
    assert(count > 0 && llm_set_prompt_snapshots(&snapshot, entries, count, identity, 1) == 0);
    for (int temp = 0; temp < 2; temp++) {
        ref.temperature = snapshot.temperature = temp ? .8f : 0;
        for (int i = 0; i < count; i++) replies_equal(&ref, &snapshot, inputs[i], 2);
        for (int i = count - 1; i >= 0; i--) replies_equal(&ref, &snapshot, inputs[i], 2);
    }
    // OFF returns to current prefill with no RNG rollback.
    assert(llm_set_prompt_snapshots(&snapshot, NULL, 0, NULL, 0) == 0);
    replies_equal(&ref, &snapshot, hello, 0);
    printf("snapshots: %d inputs, temperature 0/0.8, full/suffix restore and callbacks/RNG equal; identity/CRC/fallback passed\n", count);
}
