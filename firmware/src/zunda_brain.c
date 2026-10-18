#include "zunda_brain.h"

#include <stdio.h>
#include <string.h>

static const char *const kFeel[ZUNDA_N_FEEL] = {
    "ふつう", "うれしい", "ドヤ", "びっくり", "かなしい", "ぷんぷん", "こまった", "ねむい",
};
static int feel_id[ZUNDA_N_FEEL];
static int id_a = -1, id_b = -1;
static int last_prompt_tokens = 0;
static int last_cached_prompt_tokens = 0;

int zunda_brain_last_prompt_tokens(void) { return last_prompt_tokens; }
int zunda_brain_last_cached_prompt_tokens(void) { return last_cached_prompt_tokens; }

static int find_piece(Llm *m, const char *s) {
    for (int i = 0; i < m->tok.vocab_size; i++)
        if (strcmp(m->tok.vocab[i], s) == 0) return i;
    return -1;
}

int zunda_brain_init(Llm *m) {
    char buf[32];
    int ok = 0;
    for (int i = 0; i < ZUNDA_N_FEEL; i++) {
        snprintf(buf, sizeof(buf), "〈%s〉", kFeel[i]);
        feel_id[i] = find_piece(m, buf);
        if (feel_id[i] < 0) ok = -1;
    }
    id_a = find_piece(m, "Ａ：");
    id_b = find_piece(m, "Ｂ：");
    if (id_a < 0 || id_b < 0) ok = -2;
#if ZUNDA_FAST_PREFILL
    if (ok == 0) {
        // 語彙差し替えに備えて番号を決め打ちしない。実プロンプトでも一致を確認する。
        const char prefix[] = "Ａ：";
        int tokens[sizeof(prefix) + 3], n;
        llm_encode(m, prefix, 1, 0, tokens, &n);
        if (!llm_prefill(m, tokens, n, NULL)) return -3;
    }
#endif
    return ok;
}

const char *zunda_feel_name(int feel) { return feel >= 0 && feel < ZUNDA_N_FEEL ? kFeel[feel] : "?"; }

static int feel_of(int token) {
    for (int i = 0; i < ZUNDA_N_FEEL; i++)
        if (feel_id[i] >= 0 && token == feel_id[i]) return i;
    return -1;
}

int zunda_feel_from_logits(const float *logits) {
    int best = 0;
    for (int i = 1; i < ZUNDA_N_FEEL; i++)
        if (feel_id[i] >= 0 && (feel_id[best] < 0 || logits[feel_id[i]] > logits[feel_id[best]])) best = i;
    return best;
}

static int argmax(const float *x, int n) {
    int b = 0;
    for (int i = 1; i < n; i++)
        if (x[i] > x[b]) b = i;
    return b;
}

static int encode_prompt(Llm *m, const char *heard, int *toks) {
    static char prompt[400];
    // 長すぎる話しかけは切る（返事の分のKVキャッシュを残す）
    snprintf(prompt, sizeof(prompt), "Ａ：%.300sＢ：", heard);
    int n = 0;
    llm_encode(m, prompt, 1, 0, toks, &n);
    int limit = m->max_steps - ZUNDA_MAX_REPLY - 1;
    if (n > limit) {  // 末尾の「Ｂ：」は残す
        toks[limit - 1] = toks[n - 1];
        n = limit;
    }
    return n;
}

static float *prefill_prompt(Llm *m, const int *toks, int n) {
    last_prompt_tokens = n;
    last_cached_prompt_tokens = 0;
    float *logits = NULL;
#if ZUNDA_FAST_PREFILL
    logits = llm_prefill(m, toks, n, &last_cached_prompt_tokens);
#else
    for (int i = 0; i < n; i++) logits = llm_forward(m, toks[i], i);
#endif
    return logits;
}

float *zunda_brain_prefill_prompt(Llm *m, const char *heard, int *n_tokens) {
    int toks[403];
    int n = encode_prompt(m, heard, toks);
    if (n_tokens) *n_tokens = n;
    return prefill_prompt(m, toks, n);
}

size_t zunda_brain_prompt_snapshot_size(Llm *m, const char *heard, int skip_prefix) {
    int toks[403];
    return llm_prompt_snapshot_size(m, encode_prompt(m, heard, toks), skip_prefix);
}

int zunda_brain_export_prompt(Llm *m, const char *heard, int skip_prefix,
                              const unsigned char identity[32], llm_snapshot_write_fn writer, void *ctx) {
    int toks[403];
    int n = encode_prompt(m, heard, toks);
    return llm_prompt_snapshot_export(m, toks, n, skip_prefix, identity, writer, ctx);
}

int zunda_brain_reply(Llm *m, const char *heard, char *out, int out_size, int *feel, zunda_piece_fn on_piece,
                      void *ctx) {
    return zunda_brain_reply_ex(m, heard, out, out_size, feel, NULL, on_piece, ctx);
}

int zunda_brain_reply_ex(Llm *m, const char *heard, char *out, int out_size, int *feel,
                        zunda_prepare_fn on_prepare, zunda_piece_fn on_piece, void *ctx) {
    static int toks[403];
    int n = encode_prompt(m, heard, toks);
    float *logits = prefill_prompt(m, toks, n);
    int pos = n;

    int prev = toks[n - 1], len = 0, steps = 0;
    out[0] = 0;
    *feel = -1;
    for (; steps < ZUNDA_MAX_REPLY; steps++) {
        // greedy（温度0）か、温度>0なら llm_sample（logits を確率に書き換えるが、大小の順は変わらない）
        int next = m->temperature == 0.0f ? argmax(logits, m->config.vocab_size) : llm_sample(m, logits);
        int f = feel_of(next);
        if (f >= 0) {
            *feel = f;
            break;
        }
        if (next == id_a || next == id_b || next == 1 || next == 2) break;
        const char *piece = llm_decode(m, prev, next);
        int pl = (int)strlen(piece);
        if (len + pl < out_size) {
            memcpy(out + len, piece, pl + 1);
            len += pl;
        }
        if (on_prepare) on_prepare(piece, ctx);
        logits = llm_forward(m, next, pos++);
        if (on_piece) on_piece(piece, logits, ctx);
        prev = next;
    }
    if (*feel < 0) *feel = zunda_feel_from_logits(logits);
    return steps;
}
