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

// メモリ確保関数を差し替え可能に（ESP32ではPSRAMを使う）
typedef void *(*llm_alloc_fn)(size_t);

typedef struct { const void *data; size_t size; } LlmPromptSnapshot;
// 0=書き込み成功、非0=中断。dataは呼出し中だけ有効。
typedef int (*llm_snapshot_write_fn)(const void *data, size_t size, void *ctx);

// forward内のFP演算順・近似・量子化を変える場合は更新する。UIやcache形式変更では更新しない。
#define LLM_MATH_ABI 1
#define LLM_SNAPSHOT_FORMAT 1

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
    llm_alloc_fn alloc;
    // 直前promptのKVは既存領域に残す。raw logitsのみ別保存（sampleの破壊から守る）。
    int prompt_cache_enabled, prompt_cache_n;
    int *prompt_cache_tokens;
    float *prompt_cache_logits;
    int prefill_source;       // 0=計算（固定先頭の再利用を含む）、1=直前入力、2=Flash snapshot
    const LlmPromptSnapshot *snapshots;
    int snapshot_count, snapshots_enabled;
    unsigned char snapshot_identity[32];
} Llm;

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
// KVを直接書き換える場合は事前に llm_invalidate_prompt_cache() を呼ぶこと。
// 重み/語彙を交換するときはllm_initで再初期化し、対応するidentityのsnapshotだけ登録する。
float *llm_prefill(Llm *m, const int *tokens, int n, int *reused);
// 初回ON時に約(max_steps+vocab_size)*4Bを確保。OFFで無効化、確保済み領域は再利用する。
// 0=成功、-1=確保失敗（無効のまま）。llm_init時はOFF。乱数状態は変更しない。
int llm_set_prompt_cache(Llm *m, int enabled);
// KVをAPI外で変更したときに呼ぶ。固定先頭と直前入力を無効化。登録済snapshotは保持する。
void llm_invalidate_prompt_cache(Llm *m);
int llm_last_prefill_source(const Llm *m);
// identity: model/tokenizer/推論演算ABI/target float環境を識別するSHA256。
// データとentriesの寿命は登録解除まで必要。登録時CRC等を検証し、以後immutableとして読む。
// OFF/空配列=0、ONかつ全entry有効=0、無効entry=-1（全体をOFFにする）。
int llm_set_prompt_snapshots(Llm *m, const LlmPromptSnapshot *entries, int count,
                             const unsigned char identity[32], int enabled);
// blob: little-endian uint32 header(96B), tokens[n], 層順のK/V suffix, fp32 logits。
// skip_prefixは0または2。2なら復元時に先頭2tokenの既存KVが一致する場合のみ適用。
size_t llm_prompt_snapshot_size(const Llm *m, int n, int skip_prefix);
// prefillを実施してraw結果を順次出力。RNG不変。0=成功、負=不正引数/出力失敗。
int llm_prompt_snapshot_export(Llm *m, const int *tokens, int n, int skip_prefix,
                               const unsigned char identity[32], llm_snapshot_write_fn writer, void *ctx);
// 単独blobの検証/復元: 1=復元、0=入力/先頭が不一致、-1=blobが不正。RNG不変。
int llm_prompt_snapshot_import(Llm *m, const LlmPromptSnapshot *entry, const int *tokens, int n,
                               const unsigned char identity[32]);
int llm_sample(Llm *m, float *logits);
const char *llm_decode(Llm *m, int prev_token, int token);
// text を BOS付きでトークン化。tokens は strlen(text)+3 以上の長さが必要
void llm_encode(Llm *m, const char *text, int bos, int eos, int *tokens, int *n_tokens);
// 上位k候補(確率つき)を取り出す（「迷い」の可視化用）。logitsはsoftmax前でよい
int llm_topk(Llm *m, const float *logits, int k, LlmProbIndex *out);

#ifdef __cplusplus
}
#endif
