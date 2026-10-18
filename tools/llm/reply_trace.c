// Binary trace, compiled separately against v0.0.3 and the trial sources.
// Includes every callback's next-position logits used for voice selection.
#include "test_support.h"
#include "zunda_brain.h"

static void emit(const void *data, size_t size) {
    assert(fwrite(data, 1, size, stdout) == size);
}
static void piece(const char *text, const float *logits, void *ctx) {
    Llm *m = ctx;
    emit(text, strlen(text) + 1);
    emit(logits, m->config.vocab_size * sizeof(float));
}
int main(int argc, char **argv) {
    assert(argc == 5);
    size_t mn, tn;
    void *model = read_binary(argv[1], &mn), *tok = read_binary(argv[2], &tn);
    Llm m;
    init_test_llm(&m, model, mn, tok, tn, atoi(argv[3]));
    assert(zunda_brain_init(&m) == 0);
    m.temperature = (float)atof(argv[4]);
    m.rng = 42;
    char heard[2048], out[256];
    while (fgets(heard, sizeof(heard), stdin)) {
        heard[strcspn(heard, "\r\n")] = 0;
        int feel, steps = zunda_brain_reply(&m, heard, out, sizeof(out), &feel, piece, &m);
        int prompt_tokens = zunda_brain_last_prompt_tokens();
        emit(out, strlen(out) + 1);
        emit(&feel, sizeof(feel)); emit(&steps, sizeof(steps));
        emit(&prompt_tokens, sizeof(prompt_tokens)); emit(&m.rng, sizeof(m.rng));
    }
    assert(!ferror(stdin));
    return 0;
}
