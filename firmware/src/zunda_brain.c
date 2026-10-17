#include "zunda_brain.h"

#include <stdio.h>
#include <string.h>

static const char *const kFeel[ZUNDA_N_FEEL] = {
    "ふつう", "うれしい", "ドヤ", "びっくり", "かなしい", "ぷんぷん", "こまった", "ねむい",
};
static int feel_id[ZUNDA_N_FEEL];
static int id_a = -1, id_b = -1;

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

int zunda_brain_reply(Llm *m, const char *heard, char *out, int out_size, int *feel, zunda_piece_fn on_piece,
                      void *ctx) {
    static char prompt[400];
    static int toks[sizeof(prompt) + 3];
    // 長すぎる話しかけは切る（返事の分のKVキャッシュを残す）
    snprintf(prompt, sizeof(prompt), "Ａ：%.300sＢ：", heard);
    int n = 0;
    llm_encode(m, prompt, 1, 0, toks, &n);
    int limit = m->max_steps - ZUNDA_MAX_REPLY - 1;
    if (n > limit) {  // 末尾の「Ｂ：」は残す
        toks[limit - 1] = toks[n - 1];
        n = limit;
    }
    float *logits = NULL;
    int pos = 0;
    for (int i = 0; i < n; i++) logits = llm_forward(m, toks[i], pos++);

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
        logits = llm_forward(m, next, pos++);
        if (on_piece) on_piece(piece, logits, ctx);
        prev = next;
    }
    if (*feel < 0) *feel = zunda_feel_from_logits(logits);
    return steps;
}
