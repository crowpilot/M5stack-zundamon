// llm_core.c — llama2.c の run.c / runq.c (MIT License, (c) 2023 Andrej Karpathy) を
// メモリ上のモデル読み込み・アロケータ差し替え・KVキャッシュ長制限・int8(Q8)対応に改変したもの。
#include "llm_core.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

static void *xcalloc(llm_alloc_fn alloc, size_t n) {
    void *p = alloc ? alloc(n) : malloc(n);
    if (p) memset(p, 0, n);
    return p;
}

static int alloc_state(Llm *m, llm_alloc_fn alloc) {
    LlmConfig *p = &m->config;
    LlmState *s = &m->state;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    s->x = xcalloc(alloc, p->dim * sizeof(float));
    s->xb = xcalloc(alloc, p->dim * sizeof(float));
    s->xb2 = xcalloc(alloc, p->dim * sizeof(float));
    s->hb = xcalloc(alloc, p->hidden_dim * sizeof(float));
    s->hb2 = xcalloc(alloc, p->hidden_dim * sizeof(float));
    s->q = xcalloc(alloc, p->dim * sizeof(float));
    s->key_cache = xcalloc(alloc, (size_t)p->n_layers * m->max_steps * kv_dim * sizeof(float));
    s->value_cache = xcalloc(alloc, (size_t)p->n_layers * m->max_steps * kv_dim * sizeof(float));
    s->att = xcalloc(alloc, (size_t)p->n_heads * m->max_steps * sizeof(float));
    s->logits = xcalloc(alloc, p->vocab_size * sizeof(float));
    m->probindex = xcalloc(alloc, p->vocab_size * sizeof(LlmProbIndex));
    if (m->gs > 0) {
        int nmax = p->hidden_dim > p->dim ? p->hidden_dim : p->dim;
        s->xq = xcalloc(alloc, nmax);
        s->xs = xcalloc(alloc, (nmax / m->gs + 1) * sizeof(float));
        if (!s->xq || !s->xs) return -1;
    }
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q || !s->key_cache ||
        !s->value_cache || !s->att || !s->logits || !m->probindex) return -1;
    return 0;
}

static int alloc_mats(LlmWeights *w, int L, llm_alloc_fn alloc) {
    LlmMat **mats[] = { &w->wq, &w->wk, &w->wv, &w->wo, &w->w1, &w->w2, &w->w3 };
    for (int i = 0; i < 7; i++) {
        *mats[i] = xcalloc(alloc, L * sizeof(LlmMat));
        if (!*mats[i]) return -1;
    }
    return 0;
}

// llama2.c 形式（fp32）
static void map_weights_f32(LlmWeights *w, LlmConfig *p, const float *ptr, int shared) {
    int head_size = p->dim / p->n_heads, L = p->n_layers;
    size_t d = p->dim, hd = p->hidden_dim, kv = p->n_kv_heads * head_size, qd = p->n_heads * head_size;
    w->tok_emb.f = ptr; ptr += (size_t)p->vocab_size * d;
    w->rms_att = ptr;   ptr += L * d;
    for (int l = 0; l < L; l++) w->wq[l].f = ptr + l * d * qd;
    ptr += L * d * qd;
    for (int l = 0; l < L; l++) w->wk[l].f = ptr + l * d * kv;
    ptr += L * d * kv;
    for (int l = 0; l < L; l++) w->wv[l].f = ptr + l * d * kv;
    ptr += L * d * kv;
    for (int l = 0; l < L; l++) w->wo[l].f = ptr + l * qd * d;
    ptr += L * qd * d;
    w->rms_ffn = ptr;   ptr += L * d;
    for (int l = 0; l < L; l++) w->w1[l].f = ptr + l * d * hd;
    ptr += L * d * hd;
    for (int l = 0; l < L; l++) w->w2[l].f = ptr + l * hd * d;
    ptr += L * hd * d;
    for (int l = 0; l < L; l++) w->w3[l].f = ptr + l * d * hd;
    ptr += L * d * hd;
    w->rms_final = ptr; ptr += d;
    ptr += p->seq_len * head_size / 2;  // freq_cis_real（未使用）
    ptr += p->seq_len * head_size / 2;  // freq_cis_imag（未使用）
    w->wcls.f = shared ? w->tok_emb.f : ptr;
}

// Q8 形式: fp32 の正規化重み → int8 テンソル（int8[n] の直後に float[n/gs]）
static const char *take_q8(LlmMat *m, const char *ptr, size_t n, int gs) {
    m->q = (const int8_t *)ptr; ptr += n;
    m->s = (const float *)ptr;  ptr += (n / gs) * sizeof(float);
    return ptr;
}

static size_t map_weights_q8(LlmWeights *w, LlmConfig *p, const char *ptr, int gs) {
    const char *start = ptr;
    int head_size = p->dim / p->n_heads, L = p->n_layers;
    size_t d = p->dim, hd = p->hidden_dim, kv = p->n_kv_heads * head_size, qd = p->n_heads * head_size;
    w->rms_att = (const float *)ptr;   ptr += L * d * sizeof(float);
    w->rms_ffn = (const float *)ptr;   ptr += L * d * sizeof(float);
    w->rms_final = (const float *)ptr; ptr += d * sizeof(float);
    ptr = take_q8(&w->tok_emb, ptr, (size_t)p->vocab_size * d, gs);
    w->wcls = w->tok_emb;
    for (int l = 0; l < L; l++) ptr = take_q8(&w->wq[l], ptr, d * qd, gs);
    for (int l = 0; l < L; l++) ptr = take_q8(&w->wk[l], ptr, d * kv, gs);
    for (int l = 0; l < L; l++) ptr = take_q8(&w->wv[l], ptr, d * kv, gs);
    for (int l = 0; l < L; l++) ptr = take_q8(&w->wo[l], ptr, qd * d, gs);
    for (int l = 0; l < L; l++) ptr = take_q8(&w->w1[l], ptr, d * hd, gs);
    for (int l = 0; l < L; l++) ptr = take_q8(&w->w2[l], ptr, hd * d, gs);
    for (int l = 0; l < L; l++) ptr = take_q8(&w->w3[l], ptr, d * hd, gs);
    return (size_t)(ptr - start);
}

static int load_tokenizer(LlmTokenizer *t, const unsigned char *buf, size_t len,
                          int vocab_size, llm_alloc_fn alloc) {
    t->vocab_size = vocab_size;
    t->vocab = xcalloc(alloc, vocab_size * sizeof(char *));
    t->vocab_scores = xcalloc(alloc, vocab_size * sizeof(float));
    t->sorted_vocab = NULL;
    for (int i = 0; i < 256; i++) { t->byte_pieces[i * 2] = (unsigned char)i; t->byte_pieces[i * 2 + 1] = 0; }
    // 文字列を1つの塊に詰める（ESP32での細切れmalloc回避）
    char *pool = xcalloc(alloc, len + vocab_size);
    if (!t->vocab || !t->vocab_scores || !pool) return -1;
    size_t off = 0;
    memcpy(&t->max_token_length, buf + off, 4); off += 4;
    for (int i = 0; i < vocab_size; i++) {
        int l;
        if (off + 8 > len) return -2;
        memcpy(&t->vocab_scores[i], buf + off, 4); off += 4;
        memcpy(&l, buf + off, 4); off += 4;
        if (l < 0 || off + l > len) return -2;
        memcpy(pool, buf + off, l); off += l;
        pool[l] = 0;
        t->vocab[i] = pool;
        pool += l + 1;
    }
    return 0;
}

int llm_init(Llm *m, const void *model, size_t model_len,
             const void *tok, size_t tok_len, int max_steps, llm_alloc_fn alloc) {
    memset(m, 0, sizeof(*m));
    const char *base = (const char *)model;
    int magic;
    if (model_len < 40) return -1;
    memcpy(&magic, base, 4);
    int shared = 1;
    if (magic == LLM_Q8_MAGIC) {
        memcpy(&m->config, base + 8, sizeof(LlmConfig));
        memcpy(&m->gs, base + 36, 4);
        if (m->gs <= 0 || m->config.dim % m->gs || m->config.hidden_dim % m->gs) return -2;
    } else {
        memcpy(&m->config, base, sizeof(LlmConfig));
        shared = m->config.vocab_size > 0;
        m->config.vocab_size = abs(m->config.vocab_size);
    }
    m->max_steps = (max_steps > 0 && max_steps < m->config.seq_len) ? max_steps : m->config.seq_len;
    if (alloc_mats(&m->weights, m->config.n_layers, alloc)) return -3;
    if (m->gs > 0) {
        size_t used = map_weights_q8(&m->weights, &m->config, base + 40, m->gs);
        if (40 + used != model_len) return -2;
    } else {
        map_weights_f32(&m->weights, &m->config, (const float *)(base + sizeof(LlmConfig)), shared);
    }
    if (alloc_state(m, alloc)) return -3;
    if (load_tokenizer(&m->tok, tok, tok_len, m->config.vocab_size, alloc)) return -4;
    m->temperature = 0.8f;
    m->topp = 0.9f;
    m->rng = 42;
    return 0;
}

// ---------------------------------------------------------------- forward
static void rmsnorm(float *o, const float *x, const float *weight, int size) {
    float ss = 0.0f;
    for (int j = 0; j < size; j++) ss += x[j] * x[j];
    ss /= size; ss += 1e-5f; ss = 1.0f / sqrtf(ss);
    for (int j = 0; j < size; j++) o[j] = weight[j] * (ss * x[j]);
}

static void softmax(float *x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) if (x[i] > max_val) max_val = x[i];
    float sum = 0.0f;
    for (int i = 0; i < size; i++) { x[i] = expf(x[i] - max_val); sum += x[i]; }
    for (int i = 0; i < size; i++) x[i] /= sum;
}

static void matmul_f32(float *xout, const float *x, const float *w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    for (int i = 0; i < d; i++) {
        float val = 0.0f;
        const float *row = w + (size_t)i * n;
        for (int j = 0; j < n; j++) val += row[j] * x[j];
        xout[i] = val;
    }
}

// W (d,n) @ x (n,)。Q8 なら x を gs 個ずつ int8 に丸め、整数の積和に倍率を掛ける
static void matmul(Llm *m, float *xout, const float *x, LlmMat w, int n, int d) {
    if (!w.q) { matmul_f32(xout, x, w.f, n, d); return; }
    int gs = m->gs, ng = n / gs;
    int8_t *xq = m->state.xq;
    float *xs = m->state.xs;
    for (int g = 0; g < ng; g++) {
        float amax = 0.0f;
        for (int j = 0; j < gs; j++) { float v = fabsf(x[g * gs + j]); if (v > amax) amax = v; }
        float sc = amax > 0 ? amax / 127.0f : 1.0f;
        xs[g] = sc;
        for (int j = 0; j < gs; j++) xq[g * gs + j] = (int8_t)roundf(x[g * gs + j] / sc);
    }
    for (int i = 0; i < d; i++) {
        const int8_t *row = w.q + (size_t)i * n;
        const float *rs = w.s + (size_t)i * ng;
        float val = 0.0f;
        for (int g = 0; g < ng; g++) {
            int32_t acc = 0;
            const int8_t *a = row + g * gs, *b = xq + g * gs;
            for (int j = 0; j < gs; j++) acc += (int32_t)a[j] * (int32_t)b[j];
            val += (float)acc * rs[g] * xs[g];
        }
        xout[i] = val;
    }
}

static void embed(Llm *m, float *x, int token) {
    LlmMat e = m->weights.tok_emb;
    int dim = m->config.dim;
    if (!e.q) { memcpy(x, e.f + (size_t)token * dim, dim * sizeof(float)); return; }
    size_t off = (size_t)token * dim;
    for (int j = 0; j < dim; j++) x[j] = e.q[off + j] * e.s[(off + j) / m->gs];
}

float *llm_forward(Llm *m, int token, int pos) {
    LlmConfig *p = &m->config;
    LlmWeights *w = &m->weights;
    LlmState *s = &m->state;
    float *x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads;
    int hidden_dim = p->hidden_dim;
    int head_size = dim / p->n_heads;
    int T = m->max_steps;

    embed(m, x, token);

    for (unsigned long long l = 0; l < (unsigned long long)p->n_layers; l++) {
        rmsnorm(s->xb, x, w->rms_att + l * dim, dim);
        int loff = l * T * kv_dim;
        float *k = s->key_cache + loff + pos * kv_dim;
        float *v = s->value_cache + loff + pos * kv_dim;
        matmul(m, s->q, s->xb, w->wq[l], dim, dim);
        matmul(m, k, s->xb, w->wk[l], dim, kv_dim);
        matmul(m, v, s->xb, w->wv[l], dim, kv_dim);

        // RoPE（隣接ペアを回転する llama2.c 方式）
        for (int i = 0; i < dim; i += 2) {
            int head_dim = i % head_size;
            float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
            float val = pos * freq;
            float fcr = cosf(val), fci = sinf(val);
            int rotn = i < kv_dim ? 2 : 1;
            for (int vi = 0; vi < rotn; vi++) {
                float *vec = vi == 0 ? s->q : k;
                float v0 = vec[i], v1 = vec[i + 1];
                vec[i] = v0 * fcr - v1 * fci;
                vec[i + 1] = v0 * fci + v1 * fcr;
            }
        }

        for (int h = 0; h < p->n_heads; h++) {
            float *q = s->q + h * head_size;
            float *att = s->att + h * T;
            for (int t = 0; t <= pos; t++) {
                float *kk = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) score += q[i] * kk[i];
                att[t] = score / sqrtf((float)head_size);
            }
            softmax(att, pos + 1);
            float *xb = s->xb + h * head_size;
            memset(xb, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                float *vv = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                float a = att[t];
                for (int i = 0; i < head_size; i++) xb[i] += a * vv[i];
            }
        }
        matmul(m, s->xb2, s->xb, w->wo[l], dim, dim);
        for (int i = 0; i < dim; i++) x[i] += s->xb2[i];

        rmsnorm(s->xb, x, w->rms_ffn + l * dim, dim);
        matmul(m, s->hb, s->xb, w->w1[l], dim, hidden_dim);
        matmul(m, s->hb2, s->xb, w->w3[l], dim, hidden_dim);
        for (int i = 0; i < hidden_dim; i++) {
            float val = s->hb[i];
            val *= (1.0f / (1.0f + expf(-val)));   // SiLU
            s->hb[i] = val * s->hb2[i];
        }
        matmul(m, s->xb, s->hb, w->w2[l], hidden_dim, dim);
        for (int i = 0; i < dim; i++) x[i] += s->xb[i];
    }
    rmsnorm(x, x, w->rms_final, dim);
    matmul(m, s->logits, x, w->wcls, p->dim, p->vocab_size);
    return s->logits;
}

// ---------------------------------------------------------------- tokenizer
const char *llm_decode(Llm *m, int prev_token, int token) {
    LlmTokenizer *t = &m->tok;
    char *piece = t->vocab[token];
    if (prev_token == 1 && piece[0] == ' ') piece++;
    // "<0xAB>" 形式のバイトトークン（sscanfの%hhXに頼らず自前で解釈）
    if (piece[0] == '<' && piece[1] == '0' && piece[2] == 'x' && piece[5] == '>' && piece[6] == '\0') {
        int v = 0;
        for (int i = 3; i < 5; i++) {
            char c = piece[i];
            int d = (c >= '0' && c <= '9') ? c - '0' : (c >= 'A' && c <= 'F') ? c - 'A' + 10 :
                    (c >= 'a' && c <= 'f') ? c - 'a' + 10 : -1;
            if (d < 0) return piece;
            v = v * 16 + d;
        }
        piece = (char *)t->byte_pieces + v * 2;
    }
    return piece;
}

static int compare_tokens(const void *a, const void *b) {
    return strcmp(((const LlmTokenIndex *)a)->str, ((const LlmTokenIndex *)b)->str);
}

static int str_lookup(const char *str, LlmTokenIndex *sorted_vocab, int vocab_size) {
    LlmTokenIndex tok = { .str = (char *)str };
    LlmTokenIndex *res = bsearch(&tok, sorted_vocab, vocab_size, sizeof(LlmTokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

void llm_encode(Llm *m, const char *text, int bos, int eos, int *tokens, int *n_tokens) {
    LlmTokenizer *t = &m->tok;
    if (t->sorted_vocab == NULL) {
        t->sorted_vocab = malloc(t->vocab_size * sizeof(LlmTokenIndex));
        for (int i = 0; i < t->vocab_size; i++) { t->sorted_vocab[i].str = t->vocab[i]; t->sorted_vocab[i].id = i; }
        qsort(t->sorted_vocab, t->vocab_size, sizeof(LlmTokenIndex), compare_tokens);
    }
    char *str_buffer = malloc((t->max_token_length * 2 + 1 + 2) * sizeof(char));
    size_t str_len = 0;
    *n_tokens = 0;
    if (bos) tokens[(*n_tokens)++] = 1;
    // add_dummy_prefix（英語モデル用）。日本語モデルは学習側で無効化しているので語彙に" "が無ければ飛ばす
    // 話者マーカー「Ａ：」を持つ語彙＝このリポジトリの日本語モデル（add_dummy_prefix=False で学習）
    int ja_model = str_lookup("Ａ：", t->sorted_vocab, t->vocab_size) >= 0;
    if (text[0] != '\0' && !ja_model) {
        int dummy_prefix = str_lookup(" ", t->sorted_vocab, t->vocab_size);
        if (dummy_prefix >= 0) tokens[(*n_tokens)++] = dummy_prefix;
    }
    static const char *markers[] = { "Ａ：", "Ｂ：" };   // 学習時の user_defined_symbols
    const char *c = text;
    while (*c != '\0') {
        // 話者マーカーは丸ごと1トークン（BPEの結合では作られないため先に拾う）
        int matched = 0;
        for (int s = 0; s < 2 && ja_model; s++) {
            size_t ml = strlen(markers[s]);
            if (strncmp(c, markers[s], ml) == 0) {
                int id = str_lookup(markers[s], t->sorted_vocab, t->vocab_size);
                if (id >= 0) { tokens[(*n_tokens)++] = id; c += ml; matched = 1; break; }
            }
        }
        if (matched) continue;
        // UTF-8 1文字分
        unsigned char lead = (unsigned char)*c;
        size_t L = lead < 0x80 ? 1 : (lead >> 5) == 0x6 ? 2 : (lead >> 4) == 0xE ? 3 : (lead >> 3) == 0x1E ? 4 : 1;
        for (str_len = 0; str_len < L && c[str_len] != '\0'; str_len++) str_buffer[str_len] = c[str_len];
        str_buffer[str_len] = '\0';
        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
        if (id != -1) {
            tokens[(*n_tokens)++] = id;
        } else {
            // バイト片は "<0xAB>" を引く（語彙の並びが llama2 と違っても正しく動くように）
            for (size_t i = 0; i < str_len; i++) {
                char hex[8];
                snprintf(hex, sizeof(hex), "<0x%02X>", (unsigned char)str_buffer[i]);
                int bid = str_lookup(hex, t->sorted_vocab, t->vocab_size);
                tokens[(*n_tokens)++] = bid >= 0 ? bid : (unsigned char)str_buffer[i] + 3;
            }
        }
        c += str_len;
    }
    while (1) {
        float best_score = -1e10f;
        int best_id = -1, best_idx = -1;
        for (int i = 0; i < (*n_tokens - 1); i++) {
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i + 1]]);
            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score) { best_score = t->vocab_scores[id]; best_id = id; best_idx = i; }
        }
        if (best_idx == -1) break;
        tokens[best_idx] = best_id;
        for (int i = best_idx + 1; i < (*n_tokens - 1); i++) tokens[i] = tokens[i + 1];
        (*n_tokens)--;
    }
    if (eos) tokens[(*n_tokens)++] = 2;
    free(str_buffer);
}

// ---------------------------------------------------------------- sampler
static unsigned int random_u32(unsigned long long *state) {
    *state ^= *state >> 12; *state ^= *state << 25; *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
static float random_f32(unsigned long long *state) { return (random_u32(state) >> 8) / 16777216.0f; }

static int sample_argmax(const float *p, int n) {
    int max_i = 0; float max_p = p[0];
    for (int i = 1; i < n; i++) if (p[i] > max_p) { max_i = i; max_p = p[i]; }
    return max_i;
}
static int sample_mult(const float *p, int n, float coin) {
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) { cdf += p[i]; if (coin < cdf) return i; }
    return n - 1;
}
static int compare_prob(const void *a, const void *b) {
    const LlmProbIndex *x = a, *y = b;
    if (x->prob > y->prob) return -1;
    if (x->prob < y->prob) return 1;
    return 0;
}
static int sample_topp(const float *p, int n, float topp, LlmProbIndex *probindex, float coin) {
    int n0 = 0;
    const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) if (p[i] >= cutoff) { probindex[n0].index = i; probindex[n0].prob = p[i]; n0++; }
    qsort(probindex, n0, sizeof(LlmProbIndex), compare_prob);
    float cum = 0.0f; int last = n0 - 1;
    for (int i = 0; i < n0; i++) { cum += probindex[i].prob; if (cum > topp) { last = i; break; } }
    float r = coin * cum, cdf = 0.0f;
    for (int i = 0; i <= last; i++) { cdf += probindex[i].prob; if (r < cdf) return probindex[i].index; }
    return probindex[last].index;
}

int llm_sample(Llm *m, float *logits) {
    int n = m->config.vocab_size;
    if (m->temperature == 0.0f) return sample_argmax(logits, n);
    for (int q = 0; q < n; q++) logits[q] /= m->temperature;
    softmax(logits, n);
    float coin = random_f32(&m->rng);
    if (m->topp <= 0 || m->topp >= 1) return sample_mult(logits, n, coin);
    return sample_topp(logits, n, m->topp, m->probindex, coin);
}

int llm_topk(Llm *m, const float *logits, int k, LlmProbIndex *out) {
    int n = m->config.vocab_size;
    float mx = logits[0];
    for (int i = 1; i < n; i++) if (logits[i] > mx) mx = logits[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += expf(logits[i] - mx);
    int filled = 0;
    for (int i = 0; i < n; i++) {
        float pr = expf(logits[i] - mx) / sum;
        if (filled < k) { out[filled].prob = pr; out[filled].index = i; filled++; }
        else if (pr > out[k - 1].prob) { out[k - 1].prob = pr; out[k - 1].index = i; }
        else continue;
        for (int j = filled - 1; j > 0 && out[j].prob > out[j - 1].prob; j--) {
            LlmProbIndex tmp = out[j]; out[j] = out[j - 1]; out[j - 1] = tmp;
        }
    }
    return filled;
}
