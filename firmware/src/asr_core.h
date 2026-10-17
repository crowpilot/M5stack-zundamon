// かな出力CTC音声認識コア（C99、ESP32/ホスト共通。Arduino非依存）
//
//   asr_init(asr_model_blob, sizeof(asr_model_blob), 16000 * 5, my_alloc);
//   char text[256]; float conf;
//   asr_recognize(pcm, n, text, sizeof(text), &conf);   // → "こんにちわ" など
//
// 単一インスタンス（内部状態は静的変数）。スレッドセーフではない。
#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// fast=1 は内部SRAM優先（小さく頻繁に触るバッファ）、fast=0 は大容量（PSRAM可）
typedef void* (*asr_alloc_fn)(size_t bytes, int fast);

// blob: export.py が書き出したモデル（4バイト境界に置くこと）
// blob_bytes: 0なら大きさの検査を省略
// max_samples: 1回に認識させる最大サンプル数（16kHz）
// 戻り値: 0=成功, 負=失敗（asr_error()で理由）
int asr_init(const void* blob, size_t blob_bytes, int max_samples, asr_alloc_fn alloc);

// 16kHz mono int16 → ひらがな（UTF-8）。confidence は 0〜1 の目安。
// 戻り値: 出力した文字数（0=何も認識せず）, 負=失敗
int asr_recognize(const int16_t* pcm, int n_samples, char* out, int out_size, float* confidence);

// フレーズ照合: 直前の asr_recognize/asr_logits の出力に、登録フレーズ（発音ひらがな）を当てはめる。
// gaps[i] = （制約なしの最良経路の対数確率 − フレーズiの最良経路の対数確率）÷ フレーズの文字数。
// 小さいほどそのフレーズらしい（0以上。語彙外の文字を含む・音声が短すぎる場合は 1e9）。
// 戻り値: gaps が最小のフレーズ番号（n<=0 なら -1）。gaps は NULL でもよい
int asr_match(const char* const* phrases_kana, int n, float* gaps);

// 以下は検証・デバッグ用
int asr_features(const int16_t* pcm, int n_samples, const float** feat, int* n_frames);  // [T][n_mels]
int asr_logits(const int16_t* pcm, int n_samples, const float** logits, int* n_out);     // [T/4][vocab]
int asr_vocab_size(void);
const char* asr_error(void);

// 2コア並列用。run2(fn, a, b) は fn(a) と fn(b) を同時に実行し、両方終わってから戻ること。
// 設定しなければ順番に実行する（結果は同じ）
typedef void (*asr_job_fn)(void* arg);
typedef void (*asr_run2_fn)(asr_job_fn fn, void* a, void* b);
void asr_set_parallel(asr_run2_fn run2);

// 処理時間の計測用。now_us を渡すと、直前の認識の段階別時間[us]を asr_profile() で読める
enum { ASR_PROF_FEAT, ASR_PROF_CONV1, ASR_PROF_CONV2, ASR_PROF_GRU_IH, ASR_PROF_GRU_HH,
       ASR_PROF_FC, ASR_PROF_DECODE, ASR_PROF_N };
void asr_set_clock(uint32_t (*now_us)(void));
const uint32_t* asr_profile(void);  // [ASR_PROF_N]

#ifdef __cplusplus
}
#endif
