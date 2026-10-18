// Run via check_prefill.py. Compare cached prefill against complete forward calls.
#include "test_support.h"

static void compare(Llm *ref, Llm *fast, const int *tokens, int n, int expected_reused) {
    float *expected = NULL;
    for (int i = 0; i < n; i++) expected = llm_forward(ref, tokens[i], i);
    int reused = -1;
    float *actual = llm_prefill(fast, tokens, n, &reused);
    assert(actual && reused == expected_reused);
    assert(memcmp(expected, actual, ref->config.vocab_size * sizeof(float)) == 0);
    int kv = ref->config.dim * ref->config.n_kv_heads / ref->config.n_heads;
    for (int layer = 0; layer < ref->config.n_layers; layer++) {
        int off = layer * ref->max_steps * kv;
        assert(memcmp(ref->state.key_cache + off, fast->state.key_cache + off,
                      n * kv * sizeof(float)) == 0);
        assert(memcmp(ref->state.value_cache + off, fast->state.value_cache + off,
                      n * kv * sizeof(float)) == 0);
    }
}

int main(int argc, char **argv) {
    assert(argc == 3);
    size_t mn, tn;
    void *model = read_binary(argv[1], &mn), *tok = read_binary(argv[2], &tn);
    Llm ref, fast, separate;
    init_test_llm(&ref, model, mn, tok, tn, 64);
    init_test_llm(&fast, model, mn, tok, tn, 64);
    init_test_llm(&separate, model, mn, tok, tn, 64);
    int tokens[256], n;
    llm_encode(&ref, "Ａ：こんにちは。Ｂ：", 1, 0, tokens, &n);
    assert(n > 2 && n < 64);
    compare(&ref, &fast, tokens, n, 0);  // cold
    compare(&ref, &fast, tokens, n, 2);  // repeated
    compare(&ref, &separate, tokens, n, 0);  // per-instance ownership
    compare(&ref, &fast, tokens, 2, 0);  // final logits must be recomputed
    compare(&ref, &fast, tokens, n, 2);
    compare(&ref, &fast, tokens, 1, 0);
    compare(&ref, &fast, tokens, n, 0);
    int changed[256]; memcpy(changed, tokens, n * sizeof(int));
    changed[1] = (changed[1] + 1) % ref.config.vocab_size;
    compare(&ref, &fast, changed, n, 0);  // new tokenizer/prefix IDs
    compare(&ref, &fast, changed, n, 2);
    compare(&ref, &fast, tokens, n, 0);
    llm_forward(&fast, changed[1], 1);  // public API overwrites cached KV
    compare(&ref, &fast, tokens, n, 0);
    llm_forward(&fast, changed[1], 0);
    compare(&ref, &fast, tokens, n, 0);
    llm_forward(&fast, tokens[2], 2);  // generation leaves prefix intact
    compare(&ref, &fast, tokens, n, 2);
    int reused = -1, bad = -1;
    assert(!llm_prefill(&fast, tokens, 0, &reused) && reused == 0);
    assert(!llm_prefill(&fast, tokens, 65, NULL));
    assert(!llm_prefill(&fast, NULL, n, NULL));
    assert(!llm_prefill(&fast, &bad, 1, NULL));
    compare(&ref, &fast, tokens, n, 2);
    puts("prefill logits/KV: bitwise equal; cache lifetime and fallback: passed");
    return 0;
}
