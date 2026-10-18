// llm_core: llama2.c (Andrej Karpathy, MIT License) をベースにした
// ファイルI/Oなし・メモリ上のモデルを直接読む極小推論コア。
// Core2(ESP32)とPCの両方で同じコードが動く。
#pragma once
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len;
} LlmConfig;

// 重み行列1枚。fp32 なら f、int8(Q8) なら q と s（gs 個ごとの倍率）
typedef struct { const float *f; const int8_t *q; const float *s; } LlmMat;

typedef struct {
    const float *rms_att, *rms_ffn, *rms_final;   // rms_att/rms_ffn は層ごとに dim 個ずつ連続
    LlmMat tok_emb, wcls;
    LlmMat *wq, *wk, *wv, *wo, *w1, *w2, *w3;     // 層の数だけの配列
} LlmWeights;

typedef struct {
    float *x, *xb, *xb2, *hb, *hb2, *q, *att, *logits, *key_cache, *value_cache;
    int8_t *xq; float *xs;                        // Q8 用: 入力ベクトルの量子化バッファ
} LlmState;

typedef struct { char *str; int id; } LlmTokenIndex;

typedef struct {
    char **vocab;
    float *vocab_scores;
    LlmTokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512];
} LlmTokenizer;

typedef struct { float prob; int index; } LlmProbIndex;

typedef struct {
    LlmConfig config;
    LlmWeights weights;
    LlmState state;
    LlmTokenizer tok;
    int max_steps;            // 実際に確保したKVキャッシュ長（<= seq_len）
    int gs;                   // 0 = fp32（llama2.c形式）、>0 = int8(Q8) のグループサイズ
    float temperature, topp;
    unsigned long long rng;
    LlmProbIndex *probindex;
    // 既存KVの位置0,1を再利用する。llm_init/先頭へのllm_forwardで無効になる。
    int prefix_tokens[2], prefix_len;
} Llm;

// メモリ確保関数を差し替え可能に（ESP32ではPSRAMを使う）
typedef void *(*llm_alloc_fn)(size_t);

#define LLM_Q8_MAGIC 0x38515454
// model: llama2.c形式(v0) または Q8形式（train/llama2c_format.py の write_q8）の .bin 全体（4バイト境界に置くこと）
// tok:   llama2.c形式の tokenizer .bin 全体
// max_steps: KVキャッシュ長。0ならseq_len。小さくするとメモリ節約。
int llm_init(Llm *m, const void *model, size_t model_len,
             const void *tok, size_t tok_len, int max_steps, llm_alloc_fn alloc);

float *llm_forward(Llm *m, int token, int pos);
// 位置0からのプロンプト。中間位置の分類層を省き、最後のlogitsを返す。
// 同じ先頭2トークンのKVが残っていれば再利用する（追加KV領域なし）。
// reused は省いたトークン数（NULL可）。nは1..max_steps、不正な入力はNULL。
// 重み・KVを直接書き換える場合は事前に prefix_len=0 にすること。
float *llm_prefill(Llm *m, const int *tokens, int n, int *reused);
int llm_sample(Llm *m, float *logits);
const char *llm_decode(Llm *m, int prev_token, int token);
// text を BOS付きでトークン化。tokens は strlen(text)+3 以上の長さが必要
void llm_encode(Llm *m, const char *text, int bos, int eos, int *tokens, int *n_tokens);
// 上位k候補(確率つき)を取り出す（「迷い」の可視化用）。logitsはsoftmax前でよい
int llm_topk(Llm *m, const float *logits, int k, LlmProbIndex *out);

#ifdef __cplusplus
}
#endif
