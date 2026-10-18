// かな出力CTC音声認識コア。定義は training/frontend.py・model.py と一致させること。
#include "asr_core.h"

#include <math.h>
#include <string.h>

#define MAGIC 0x5253414Bu
#define VERSION 1u
#define TILE 16

typedef struct {
  const float *w_ih, *w_hh, *b_ih, *b_hh;
  int in;
} gru_dir_t;

// コアごとの作業領域（2コアで並列に計算するため2組持つ）
typedef struct {
  float *tile, *whh, *h, *gh, *gin;
  float *fre, *fim, *power;
} work_t;

static struct {
  int ready;
  // ハイパーパラメータ
  int n_mels, win, hop, nfft, nbins, ch, k, stride, hid, layers, vocab;
  float preemph, log_eps, norm_eps;
  // 重み（blobを直接指す）
  const float *window, *melfb, *c1w, *c1b, *c2w, *c2b, *fcw, *fcb;
  gru_dir_t gru[8][2];
  const char* vocab_str;  // vocab × 4バイト
  // 作業領域
  int max_samples, max_t, max_t1, max_t2;
  float *feat, *c1, *c2, *ga, *gb, *logits, *lse;
  int last_t2;  // 直前の推論の出力フレーム数（フレーズ照合用）
  work_t w[2];
  float *cosv, *sinv;
  uint16_t* bitrev;
  uint16_t *mel_lo, *mel_hi;  // 各メルフィルタの非ゼロ範囲 [lo, hi)
  int tile_in;
  const int16_t* frontend_pcm;
  int frontend_frames, frontend_samples, reused_frames;
} S;

static unsigned g_options;
void asr_set_options(unsigned options) {
  g_options = options & (ASR_OPT_SKIP_UNUSED_CONFIDENCE | ASR_OPT_RAW_MATCH);
}
unsigned asr_options(void) { return g_options; }

void asr_frontend_reset(void) {
  S.frontend_pcm = NULL;
  S.frontend_frames = S.frontend_samples = 0;
}
void asr_frontend_begin(const int16_t* pcm) {
  asr_frontend_reset();
  S.frontend_pcm = pcm;
}
int asr_frontend_reused_frames(void) { return S.reused_frames; }
void asr_release_scratch(void) {
  asr_frontend_reset();
  S.last_t2 = 0;
}

static const char* g_err = "";
const char* asr_error(void) { return g_err; }
int asr_vocab_size(void) { return S.vocab; }

// ---- 計測 ---------------------------------------------------------------------
static uint32_t (*g_now)(void);
static uint32_t g_prof[ASR_PROF_N], g_mark;
void asr_set_clock(uint32_t (*now_us)(void)) { g_now = now_us; }
const uint32_t* asr_profile(void) { return g_prof; }
static uint32_t g_gru_ih_us;
static void prof_start(void) {
  memset(g_prof, 0, sizeof(g_prof));
  g_gru_ih_us = 0;
  if (g_now) g_mark = g_now();
}
static void prof_add(int stage) {  // 前回の印からの時間を stage に足す
  if (!g_now) return;
  uint32_t t = g_now();
  g_prof[stage] += t - g_mark;
  g_mark = t;
}

// ---- 2コア並列 -----------------------------------------------------------------
static asr_run2_fn g_run2;
void asr_set_parallel(asr_run2_fn run2) { g_run2 = run2; }
static void run2(asr_job_fn fn, void* a, void* b) {
  if (g_run2) {
    g_run2(fn, a, b);
  } else {
    fn(a);
    fn(b);
  }
}

static int fail(const char* msg) {
  g_err = msg;
  return -1;
}

static int num_frames(int n) { return n < S.win ? 1 : 1 + (n - S.win) / S.hop; }
static int conv_out(int n) { return (n + 2 * (S.k / 2) - S.k) / S.stride + 1; }

static float u32f(uint32_t u) {
  float f;
  memcpy(&f, &u, 4);
  return f;
}

size_t asr_scratch_bytes(const void* blob, size_t blob_bytes, int max_samples) {
  if (!blob || ((uintptr_t)blob & 3u) || (blob_bytes && blob_bytes < 64) || max_samples <= 0) return 0;
  const uint32_t* hd = (const uint32_t*)blob;
  if (hd[0] != MAGIC || hd[1] != VERSION || !hd[2] || !hd[3] || !hd[4] || !hd[6] ||
      !hd[7] || !hd[8] || !hd[9] || !hd[11]) return 0;
  uint64_t T = (uint64_t)max_samples < hd[3] ? 1 : 1 + ((uint64_t)max_samples - hd[3]) / hd[4];
  uint64_t T1 = (T + 2 * (hd[7] / 2) - hd[7]) / hd[8] + 1;
  uint64_t T2 = (T1 + 2 * (hd[7] / 2) - hd[7]) / hd[8] + 1;
  uint64_t count = T * hd[2] + (T1 + T2) * hd[6] + T2 * (10 * (uint64_t)hd[9] + hd[11] + 1);
  return count > SIZE_MAX / sizeof(float) ? 0 : (size_t)count * sizeof(float);
}

static int init_impl(const void* blob, size_t blob_bytes, int max_samples, asr_alloc_fn alloc,
                     void* scratch, size_t scratch_bytes) {
  S.ready = 0;
  asr_release_scratch();
  S.reused_frames = 0;
  if (!blob || !alloc || (blob_bytes && blob_bytes < 64) || max_samples <= 0) return fail("bad init arguments");
  const uint32_t* hd = (const uint32_t*)blob;
  if (((uintptr_t)blob & 3u) != 0) return fail("blob is not 4-byte aligned");
  if (hd[0] != MAGIC) return fail("bad magic");
  if (hd[1] != VERSION) return fail("unsupported version");
  S.n_mels = (int)hd[2]; S.win = (int)hd[3]; S.hop = (int)hd[4]; S.nfft = (int)hd[5];
  S.ch = (int)hd[6]; S.k = (int)hd[7]; S.stride = (int)hd[8]; S.hid = (int)hd[9];
  S.layers = (int)hd[10]; S.vocab = (int)hd[11];
  S.preemph = u32f(hd[12]); S.log_eps = u32f(hd[13]); S.norm_eps = u32f(hd[14]);
  S.nbins = S.nfft / 2 + 1;
  if (S.layers < 1 || S.layers > 8) return fail("bad layer count");
  if ((S.nfft & (S.nfft - 1)) != 0 || S.win > S.nfft) return fail("bad fft size");

  const float* p = (const float*)(hd + 16);
  int H = S.hid, H3 = 3 * S.hid;
  S.window = p; p += S.win;
  S.melfb = p; p += S.n_mels * S.nbins;
  S.c1w = p; p += S.ch * S.n_mels * S.k;
  S.c1b = p; p += S.ch;
  S.c2w = p; p += S.ch * S.ch * S.k;
  S.c2b = p; p += S.ch;
  for (int l = 0; l < S.layers; l++) {
    int in = l == 0 ? S.ch : 2 * H;
    for (int d = 0; d < 2; d++) {
      gru_dir_t* g = &S.gru[l][d];
      g->in = in;
      g->w_ih = p; p += H3 * in;
      g->w_hh = p; p += H3 * H;
      g->b_ih = p; p += H3;
      g->b_hh = p; p += H3;
    }
  }
  S.fcw = p; p += S.vocab * 2 * H;
  S.fcb = p; p += S.vocab;
  S.vocab_str = (const char*)p;
  size_t need = (size_t)((const char*)p - (const char*)blob) + (size_t)S.vocab * 4;
  if (blob_bytes && blob_bytes != need) return fail("blob size mismatch");

  // 作業領域
  S.max_samples = max_samples;
  S.max_t = num_frames(max_samples);
  S.max_t1 = conv_out(S.max_t);
  S.max_t2 = conv_out(S.max_t1);
  int ti = S.n_mels * S.k;
  if (S.ch * S.k > ti) ti = S.ch * S.k;
  if (2 * H > ti) ti = 2 * H;
  S.tile_in = ti;
  size_t arena_need = asr_scratch_bytes(blob, blob_bytes, max_samples), arena_offset = 0;
  if (!arena_need) return fail("invalid scratch dimensions");
  if (scratch && (((uintptr_t)scratch & 3u) || scratch_bytes < arena_need)) return fail("scratch arena too small or unaligned");
#define A(ptr, n, fast) do { ptr = alloc(sizeof(*ptr) * (size_t)(n), fast); if (!ptr) return fail("out of memory: " #ptr); } while (0)
#define L(ptr, n) do { \
  if (scratch) { ptr = (float*)((unsigned char*)scratch + arena_offset); arena_offset += sizeof(*ptr) * (size_t)(n); } \
  else { A(ptr, n, 0); } \
} while (0)
  L(S.feat, S.max_t * S.n_mels);
  L(S.c1, S.max_t1 * S.ch);
  L(S.c2, S.max_t2 * S.ch);
  L(S.ga, S.max_t2 * 2 * H);
  L(S.gb, S.max_t2 * 2 * H);
  L(S.logits, S.max_t2 * S.vocab);
  L(S.lse, S.max_t2);
  for (int c = 0; c < 2; c++) {
    work_t* w = &S.w[c];
    L(w->gin, S.max_t2 * H3);
    A(w->whh, H3 * H, 1);
    A(w->h, H, 1);
    A(w->gh, H3, 1);
    A(w->tile, TILE * ti, 1);
    A(w->fre, S.nfft, 1);
    A(w->fim, S.nfft, 1);
    A(w->power, S.nbins, 1);
  }
  A(S.cosv, S.nfft / 2, 1);
  A(S.sinv, S.nfft / 2, 1);
  A(S.bitrev, S.nfft, 1);
  A(S.mel_lo, S.n_mels, 1);
  A(S.mel_hi, S.n_mels, 1);
#undef A
#undef L
  for (int m = 0; m < S.n_mels; m++) {  // メルフィルタはほぼゼロなので非ゼロ範囲だけ計算する
    const float* w = S.melfb + m * S.nbins;
    int lo = 0, hi = S.nbins;
    while (lo < hi && w[lo] == 0.0f) lo++;
    while (hi > lo && w[hi - 1] == 0.0f) hi--;
    S.mel_lo[m] = (uint16_t)lo;
    S.mel_hi[m] = (uint16_t)hi;
  }
  int bits = 0;
  while ((1 << bits) < S.nfft) bits++;
  for (int i = 0; i < S.nfft; i++) {
    int r = 0;
    for (int b = 0; b < bits; b++) r |= ((i >> b) & 1) << (bits - 1 - b);
    S.bitrev[i] = (uint16_t)r;
  }
  for (int i = 0; i < S.nfft / 2; i++) {
    double a = -2.0 * 3.14159265358979323846 * i / S.nfft;
    S.cosv[i] = (float)cos(a);
    S.sinv[i] = (float)sin(a);
  }
  S.ready = 1;
  return 0;
}

int asr_init(const void* blob, size_t blob_bytes, int max_samples, asr_alloc_fn alloc) {
  return init_impl(blob, blob_bytes, max_samples, alloc, NULL, 0);
}

int asr_init_with_scratch(const void* blob, size_t blob_bytes, int max_samples, asr_alloc_fn alloc,
                          void* scratch, size_t scratch_bytes) {
  if (!scratch) return fail("missing scratch arena");
  return init_impl(blob, blob_bytes, max_samples, alloc, scratch, scratch_bytes);
}

// ---- 特徴量 ---------------------------------------------------------------

static void fft_inplace(float* re, float* im) {
  int n = S.nfft;
  for (int i = 0; i < n; i++) {
    int j = S.bitrev[i];
    if (j > i) {
      float t = re[i]; re[i] = re[j]; re[j] = t;
      t = im[i]; im[i] = im[j]; im[j] = t;
    }
  }
  for (int len = 2; len <= n; len <<= 1) {
    int half = len >> 1, step = n / len;
    for (int s = 0; s < n; s += len) {
      for (int j = 0; j < half; j++) {
        float wr = S.cosv[j * step], wi = S.sinv[j * step];
        int a = s + j, b = a + half;
        float xr = re[b] * wr - im[b] * wi;
        float xi = re[b] * wi + im[b] * wr;
        re[b] = re[a] - xr; im[b] = im[a] - xi;
        re[a] += xr; im[a] += xi;
      }
    }
  }
}

typedef struct {
  const int16_t* pcm;
  int n, t0, t1;
  work_t* w;
} feat_job_t;

static void feat_job(void* arg) {
  const feat_job_t* jb = (const feat_job_t*)arg;
  const int16_t* pcm = jb->pcm;
  int n = jb->n;
  float *fre = jb->w->fre, *fim = jb->w->fim, *power = jb->w->power;
  const float inv = 1.0f / 32768.0f;
  for (int t = jb->t0; t < jb->t1; t++) {
    int s0 = t * S.hop;
    for (int j = 0; j < S.nfft; j++) {
      float v = 0.0f;
      if (j < S.win) {
        int s = s0 + j;
        float x = s < n ? pcm[s] * inv : 0.0f;
        float xp = (s > 0 && s - 1 < n) ? pcm[s - 1] * inv : 0.0f;
        float y = s == 0 ? x : x - S.preemph * xp;
        v = y * S.window[j];
      }
      fre[j] = v;
      fim[j] = 0.0f;
    }
    fft_inplace(fre, fim);
    for (int b = 0; b < S.nbins; b++) power[b] = fre[b] * fre[b] + fim[b] * fim[b];
    float* f = S.feat + t * S.n_mels;
    for (int m = 0; m < S.n_mels; m++) {
      const float* w = S.melfb + m * S.nbins;
      float acc = 0.0f;
      for (int b = S.mel_lo[m]; b < S.mel_hi[m]; b++) acc += w[b] * power[b];
      f[m] = logf(acc + S.log_eps);
    }
  }
}

int asr_frontend_push(int n_samples) {
  if (!S.ready || !S.frontend_pcm) return fail("frontend not started");
  if (n_samples < S.frontend_samples || n_samples > S.max_samples) return fail("frontend PCM length is not append-only");
  int T = n_samples < S.win ? 0 : num_frames(n_samples);
  int added = T - S.frontend_frames;
  if (added > 0) {
    // Short chunks avoid waking the second core; every FFT uses the same arithmetic as batch mode.
    feat_job_t job = {S.frontend_pcm, n_samples, S.frontend_frames, T, &S.w[0]};
    feat_job(&job);
  }
  S.frontend_frames = T;
  S.frontend_samples = n_samples;
  return added;
}

static int compute_features(const int16_t* pcm, int n) {
  int T = num_frames(n);
  int start = pcm == S.frontend_pcm && n >= S.win ? S.frontend_frames : 0;
  if (start > T) start = T;  // the caller may trim the final silence
  S.reused_frames = start;
  if (start < T) {
    int mid = start + (T - start) / 2;
    feat_job_t a = {pcm, n, start, mid, &S.w[0]}, b = {pcm, n, mid, T, &S.w[1]};
    run2(feat_job, &a, &b);
  }
  // CMVN writes over the raw features: a later inference must not reuse them a second time.
  asr_frontend_reset();
  // 発話単位CMVN（母分散）
  for (int m = 0; m < S.n_mels; m++) {
    double mu = 0.0, var = 0.0;
    for (int t = 0; t < T; t++) mu += S.feat[t * S.n_mels + m];
    mu /= T;
    for (int t = 0; t < T; t++) {
      double d = S.feat[t * S.n_mels + m] - mu;
      var += d * d;
    }
    float sd = (float)sqrt(var / T);
    float fm = (float)mu, sc = 1.0f / (sd + S.norm_eps);
    for (int t = 0; t < T; t++) S.feat[t * S.n_mels + m] = (S.feat[t * S.n_mels + m] - fm) * sc;
  }
  return T;
}

// ---- ネットワーク -----------------------------------------------------------

// 重み4行 W[0..3][in] と x の内積を a[0..3] に足し込む。x を1回読んで4行に使い、
// 独立した4つの和で浮動小数点演算の待ちを隠す（各行の足し算の順序は1行ずつのときと同じ）
static inline void dot4(const float* restrict W, int in, const float* restrict x, float* a) {
  const float *w0 = W, *w1 = W + in, *w2 = W + 2 * in, *w3 = W + 3 * in;
  float a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3];
  for (int i = 0; i < in; i++) {
    float xi = x[i];
    a0 += w0[i] * xi;
    a1 += w1[i] * xi;
    a2 += w2[i] * xi;
    a3 += w3[i] * xi;
  }
  a[0] = a0; a[1] = a1; a[2] = a2; a[3] = a3;
}

static inline float dot1(const float* restrict w, int in, const float* restrict x, float acc) {
  for (int i = 0; i < in; i++) acc += w[i] * x[i];
  return acc;
}

// tile[nt][in] に対して Y[t][r] = b[r] + W[r]·tile[t]（r0〜r1行。重みを4行ずつ使い回す順序）
static void gemm_tile(const float* W, const float* b, int r0, int r1, int in, int nt, const float* tile,
                      float* Y, int ldy) {
  int r = r0;
  for (; r + 4 <= r1; r += 4) {
    const float* w = W + (size_t)r * in;
    for (int t = 0; t < nt; t++) {
      float a[4] = {b[r], b[r + 1], b[r + 2], b[r + 3]};
      dot4(w, in, tile + t * in, a);
      float* y = Y + (size_t)t * ldy + r;
      y[0] = a[0]; y[1] = a[1]; y[2] = a[2]; y[3] = a[3];
    }
  }
  for (; r < r1; r++) {
    const float* w = W + (size_t)r * in;
    for (int t = 0; t < nt; t++) Y[(size_t)t * ldy + r] = dot1(w, in, tile + t * in, b[r]);
  }
}

typedef struct {
  const float *W, *b, *tile;
  int r0, r1, in, nt, ldy;
  float* Y;
} gemm_job_t;

static void gemm_job(void* arg) {
  const gemm_job_t* jb = (const gemm_job_t*)arg;
  gemm_tile(jb->W, jb->b, jb->r0, jb->r1, jb->in, jb->nt, jb->tile, jb->Y, jb->ldy);
}

// 同じタイルを2コアで読み、出力の行を前半・後半に分けて計算する
static void gemm_tile2(const float* W, const float* b, int rows, int in, int nt, const float* tile, float* Y,
                       int ldy) {
  int mid = (rows / 2 + 3) & ~3;
  if (mid > rows) mid = rows;
  gemm_job_t a = {W, b, tile, 0, mid, in, nt, ldy, Y}, c = {W, b, tile, mid, rows, in, nt, ldy, Y};
  run2(gemm_job, &a, &c);
}

// parallel=1 なら行を2コアで分担（w のタイルを使う）
static void dense(const float* W, const float* b, int rows, int in, const float* X, int T, float* Y, work_t* w,
                  int parallel) {
  for (int t0 = 0; t0 < T; t0 += TILE) {
    int nt = T - t0 < TILE ? T - t0 : TILE;
    memcpy(w->tile, X + (size_t)t0 * in, sizeof(float) * (size_t)nt * in);
    if (parallel)
      gemm_tile2(W, b, rows, in, nt, w->tile, Y + (size_t)t0 * rows, rows);
    else
      gemm_tile(W, b, 0, rows, in, nt, w->tile, Y + (size_t)t0 * rows, rows);
  }
}

// X[Tin][cin] → ReLU(conv) → Y[Tout][cout]（PyTorch Conv1d, padding=k/2）
static int conv_relu(const float* W, const float* b, int cin, int cout, const float* X, int Tin, float* Y) {
  int Tout = conv_out(Tin), K = S.k, P = S.k / 2, in = cin * K;
  for (int o0 = 0; o0 < Tout; o0 += TILE) {
    int nt = Tout - o0 < TILE ? Tout - o0 : TILE;
    for (int tt = 0; tt < nt; tt++) {
      float* col = S.w[0].tile + tt * in;
      int base = (o0 + tt) * S.stride - P;
      for (int c = 0; c < cin; c++)
        for (int j = 0; j < K; j++) {
          int s = base + j;
          col[c * K + j] = (s >= 0 && s < Tin) ? X[(size_t)s * cin + c] : 0.0f;
        }
    }
    gemm_tile2(W, b, cout, in, nt, S.w[0].tile, Y + (size_t)o0 * cout, cout);
  }
  for (int i = 0; i < Tout * cout; i++)
    if (Y[i] < 0.0f) Y[i] = 0.0f;
  return Tout;
}

static float sigm(float x) { return 1.0f / (1.0f + expf(-x)); }


typedef struct {
  int l, d, T;
  const float* X;
  float* Y;
} gru_job_t;

// 双方向GRU1層の片方向。前向き・後向きは互いに独立なので別々のコアで計算できる
static void gru_dir_job(void* arg) {
  const gru_job_t* jb = (const gru_job_t*)arg;
  int H = S.hid, H3 = 3 * H, d = jb->d, T = jb->T;
  const gru_dir_t* g = &S.gru[jb->l][d];
  work_t* w = &S.w[d];
  float* Y = jb->Y;
  {
    uint32_t t0 = g_now ? g_now() : 0;
    dense(g->w_ih, g->b_ih, H3, g->in, jb->X, T, w->gin, w, 0);
    if (g_now && d == 0) g_gru_ih_us += g_now() - t0;
    memcpy(w->whh, g->w_hh, sizeof(float) * (size_t)H3 * H);  // 再帰重みは内部SRAMへ
    float *h = w->h, *gh = w->gh;
    const float *whh = w->whh, *bhh = g->b_hh;
    for (int i = 0; i < H; i++) h[i] = 0.0f;
    for (int s = 0; s < T; s++) {
      int t = d == 0 ? s : T - 1 - s;
      int r = 0;
      for (; r + 4 <= H3; r += 4) {
        float a[4] = {bhh[r], bhh[r + 1], bhh[r + 2], bhh[r + 3]};
        dot4(whh + r * H, H, h, a);
        gh[r] = a[0]; gh[r + 1] = a[1]; gh[r + 2] = a[2]; gh[r + 3] = a[3];
      }
      for (; r < H3; r++) gh[r] = dot1(whh + r * H, H, h, bhh[r]);
      const float* gi = w->gin + (size_t)t * H3;
      float* y = Y + (size_t)t * 2 * H + d * H;
      for (int i = 0; i < H; i++) {
        float rr = sigm(gi[i] + gh[i]);
        float zz = sigm(gi[H + i] + gh[H + i]);
        float nn = tanhf(gi[2 * H + i] + rr * gh[2 * H + i]);
        h[i] = (1.0f - zz) * nn + zz * h[i];
        y[i] = h[i];
      }
    }
  }
}

// 双方向GRU1層: X[T][in] → Y[T][2H]（前向き, 後向き の順に連結）
static void gru_layer(int l, const float* X, int T, float* Y) {
  gru_job_t f = {l, 0, T, X, Y}, b = {l, 1, T, X, Y};
  run2(gru_dir_job, &f, &b);
}

static int clamp_len(int n) { return n > S.max_samples ? S.max_samples : n; }

int asr_features(const int16_t* pcm, int n, const float** feat, int* n_frames) {
  if (!S.ready) return fail("not initialized");
  if (!pcm || n < 0) return fail("bad PCM arguments");
  n = clamp_len(n);
  *n_frames = compute_features(pcm, n);
  *feat = S.feat;
  return 0;
}

int asr_logits(const int16_t* pcm, int n, const float** logits, int* n_out) {
  if (!S.ready) return fail("not initialized");
  if (!pcm || n < 0) return fail("bad PCM arguments");
  n = clamp_len(n);
  prof_start();
  int T = compute_features(pcm, n);
  prof_add(ASR_PROF_FEAT);
  int T1 = conv_relu(S.c1w, S.c1b, S.n_mels, S.ch, S.feat, T, S.c1);
  prof_add(ASR_PROF_CONV1);
  int T2 = conv_relu(S.c2w, S.c2b, S.ch, S.ch, S.c1, T1, S.c2);
  prof_add(ASR_PROF_CONV2);
  const float* x = S.c2;
  float* bufs[2] = {S.ga, S.gb};
  for (int l = 0; l < S.layers; l++) {
    gru_layer(l, x, T2, bufs[l & 1]);
    x = bufs[l & 1];
  }
  prof_add(ASR_PROF_GRU_HH);  // GRU全体。うち前向き方向の入力側を GRU_IH に分ける
  g_prof[ASR_PROF_GRU_HH] -= g_gru_ih_us;
  g_prof[ASR_PROF_GRU_IH] += g_gru_ih_us;
  dense(S.fcw, S.fcb, S.vocab, 2 * S.hid, x, T2, S.logits, &S.w[0], 1);
  prof_add(ASR_PROF_FC);
  *logits = S.logits;
  *n_out = T2;
  S.last_t2 = T2;
  return 0;
}

int asr_recognize(const int16_t* pcm, int n, char* out, int out_size, float* confidence) {
  const float* lg;
  int T;
  if (asr_logits(pcm, n, &lg, &T) < 0) return -1;
  int len = 0, chars = 0, prev = 0, emitted = 0;
  double conf = 0.0;
  if (out_size > 0) out[0] = '\0';
  for (int t = 0; t < T; t++) {
    const float* v = lg + (size_t)t * S.vocab;
    int best = 0;
    for (int i = 1; i < S.vocab; i++)
      if (v[i] > v[best]) best = i;
    if (best != 0) {
      if (confidence || !(g_options & ASR_OPT_SKIP_UNUSED_CONFIDENCE)) {
        double z = 0.0;
        for (int i = 0; i < S.vocab; i++) z += exp((double)(v[i] - v[best]));
        conf += 1.0 / z;
        emitted++;
      }
      if (best != prev) {
        const char* s = S.vocab_str + best * 4;
        int sl = 0;
        while (sl < 4 && s[sl]) sl++;
        if (len + sl < out_size) {
          memcpy(out + len, s, (size_t)sl);
          len += sl;
          out[len] = '\0';
          chars++;
        }
      }
    }
    prev = best;
  }
  if (confidence) *confidence = emitted ? (float)(conf / emitted) : 0.0f;
  prof_add(ASR_PROF_DECODE);
  return chars;
}

// ---- フレーズ照合 ---------------------------------------------------------------

#define MATCH_MAX_CHARS 64
#define NEG_INF (-1e30f)

// UTF-8のかな1文字を語彙番号に。見つからなければ -1。*len に文字のバイト数
static int vocab_id(const char* s, int* len) {
  int sl = (unsigned char)s[0] < 0x80 ? 1 : ((unsigned char)s[0] < 0xE0 ? 2 : ((unsigned char)s[0] < 0xF0 ? 3 : 4));
  *len = sl;
  for (int i = 1; i < S.vocab; i++) {
    const char* v = S.vocab_str + i * 4;
    int vl = 0;
    while (vl < 4 && v[vl]) vl++;
    if (vl == sl && memcmp(v, s, (size_t)sl) == 0) return i;
  }
  return -1;
}

// CTCの最良経路（Viterbi）の対数確率。lab は blank を含まないラベル列
static float ctc_viterbi(const int* lab, int L, int T) {
  static float v[2 * MATCH_MAX_CHARS + 1], nv[2 * MATCH_MAX_CHARS + 1];
  int SN = 2 * L + 1;
  for (int s = 0; s < SN; s++) v[s] = NEG_INF;
  const float* lg = S.logits;
  v[0] = lg[0] - S.lse[0];
  if (L > 0) v[1] = lg[lab[0]] - S.lse[0];
  for (int t = 1; t < T; t++) {
    const float* row = lg + (size_t)t * S.vocab;
    for (int s = 0; s < SN; s++) {
      int k = (s & 1) ? lab[s >> 1] : 0;
      float best = v[s];
      if (s >= 1 && v[s - 1] > best) best = v[s - 1];
      if ((s & 1) && s >= 2 && lab[s >> 1] != lab[(s >> 1) - 1] && v[s - 2] > best) best = v[s - 2];
      nv[s] = best <= NEG_INF ? NEG_INF : best + row[k] - S.lse[t];
    }
    memcpy(v, nv, sizeof(float) * (size_t)SN);
  }
  float e = v[SN - 1];
  if (SN >= 2 && v[SN - 2] > e) e = v[SN - 2];
  return e;
}

int asr_match(const char* const* phrases, int n, float* gaps) {
  if (!S.ready || n <= 0 || S.last_t2 <= 0) return -1;
  int T = S.last_t2, V = S.vocab;
  double free_lp = 0.0;  // 制約なし最良経路 = 各フレームの最大値の和
  for (int t = 0; t < T; t++) {
    const float* row = S.logits + (size_t)t * V;
    float mx = row[0];
    for (int i = 1; i < V; i++)
      if (row[i] > mx) mx = row[i];
    if (g_options & ASR_OPT_RAW_MATCH) {
      // Any common per-frame offset cancels between the free and constrained paths.
      // Subtract max(logit), rather than logsumexp, to avoid exponentials AND large raw sums.
      S.lse[t] = mx;
    } else {
      double z = 0.0;
      for (int i = 0; i < V; i++) z += exp((double)(row[i] - mx));
      S.lse[t] = mx + (float)log(z);
      free_lp += mx - S.lse[t];
    }
  }
  int best = -1;
  float best_gap = 0.0f;
  for (int p = 0; p < n; p++) {
    int lab[MATCH_MAX_CHARS], L = 0, ok = 1;
    for (const char* c = phrases[p]; *c && ok;) {
      int len, id = vocab_id(c, &len);
      if (id < 0 || L >= MATCH_MAX_CHARS) ok = 0;
      else lab[L++] = id;
      c += len;
    }
    float gap = 1e9f;
    if (ok && L > 0) {
      float lp = ctc_viterbi(lab, L, T);
      if (lp > NEG_INF / 2) gap = (float)((free_lp - lp) / L);
    }
    if (gaps) gaps[p] = gap;
    if (best < 0 || gap < best_gap) {
      best = p;
      best_gap = gap;
    }
  }
  return best;
}
