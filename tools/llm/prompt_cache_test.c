#include "prompt_cache_test_support.h"

static void *fail_alloc(size_t n) { (void)n; return NULL; }

int main(int argc, char **argv) {
    assert(argc == 3);
    size_t mn, tn; void *model = read_binary(argv[1], &mn), *tok = read_binary(argv[2], &tn);
    Llm ref, cached, failed;
    init_test_llm(&ref, model, mn, tok, tn, 64);
    init_test_llm(&cached, model, mn, tok, tn, 64);
    init_test_llm(&failed, model, mn, tok, tn, 64);
    assert(zunda_brain_init(&ref) == 0 && zunda_brain_init(&cached) == 0);
    assert(llm_set_prompt_cache(&cached, 1) == 0);
    failed.alloc = fail_alloc;
    assert(llm_set_prompt_cache(&failed, 1) == -1 && !failed.prompt_cache_enabled);
    int tokens[256], changed[256], n;
    llm_encode(&ref, "Ａ：こんにちは。Ｂ：", 1, 0, tokens, &n);
    prefill_equal(&ref, &cached, tokens, n, 0);
    prefill_equal(&ref, &cached, tokens, n, 1);
    // Sampling mutates live logits; the saved raw logits and RNG must survive reuse.
    ref.temperature = cached.temperature = .8f;
    assert(llm_sample(&ref, ref.state.logits) == llm_sample(&cached, cached.state.logits));
    prefill_equal(&ref, &cached, tokens, n, 1);
    assert(ref.rng == cached.rng);
    llm_forward(&cached, tokens[0], n);  // decode appends; the prompt is still intact
    prefill_equal(&ref, &cached, tokens, n, 1);
    llm_forward(&cached, tokens[0], n - 1);  // a prompt KV row was overwritten
    prefill_equal(&ref, &cached, tokens, n, 0);
    memcpy(changed, tokens, (size_t)n * sizeof(int)); changed[2] = (changed[2] + 1) % ref.config.vocab_size;
    prefill_equal(&ref, &cached, changed, n, 0);
    prefill_equal(&ref, &cached, tokens, n, 0);
    prefill_equal(&ref, &cached, tokens, 2, 0);
    prefill_equal(&ref, &cached, tokens, 2, 1);
    prefill_equal(&ref, &cached, tokens, 1, 0);
    prefill_equal(&ref, &cached, tokens, 1, 1);
    llm_invalidate_prompt_cache(&cached);
    prefill_equal(&ref, &cached, tokens, n, 0);
    assert(llm_set_prompt_cache(&cached, 0) == 0);
    prefill_equal(&ref, &cached, tokens, n, 0);
    assert(llm_set_prompt_cache(&cached, 1) == 0);
    prefill_equal(&ref, &cached, tokens, n, 0);
    int bad = -1;
    assert(!llm_prefill(&cached, &bad, 1, NULL));
    prefill_equal(&ref, &cached, tokens, n, 1);
    char heard[1024]; int count = 0;
    while (fgets(heard, sizeof(heard), stdin)) {
        heard[strcspn(heard, "\n")] = 0;
        assert(llm_set_prompt_cache(&cached, 1) == 0);
        for (int temp = 0; temp < 2; temp++) {
            ref.temperature = cached.temperature = temp ? .8f : 0;
            replies_equal(&ref, &cached, heard, temp ? 1 : 0);
            replies_equal(&ref, &cached, heard, 1);
        }
        count++;
    }
    printf("prompt cache: %d inputs at temperature 0/0.8, reply/logits/RNG/prepare order equal; invalidation passed\n", count);
}
