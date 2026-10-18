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

// 大きな作業領域だけを呼出側の4バイト境界arenaへ配置する任意の初期化方法。
// bytesは同じblob/max_samplesでasr_scratch_bytesが返す値以上。小さな内部RAM用領域はallocを使う。
// arenaは認識～asr_match完了までASRが専有する。他処理に貸す前にasr_release_scratchを呼ぶ。
size_t asr_scratch_bytes(const void* blob, size_t blob_bytes, int max_samples);
int asr_init_with_scratch(const void* blob, size_t blob_bytes, int max_samples, asr_alloc_fn alloc,
                          void* scratch, size_t scratch_bytes);
void asr_release_scratch(void);

// 0は従来の計算。独立した比較用スイッチであり、推論中に変更しないこと。
enum { ASR_OPT_SKIP_UNUSED_CONFIDENCE = 1u, ASR_OPT_RAW_MATCH = 2u };
void asr_set_options(unsigned options);
unsigned asr_options(void);

// 録音中のFFT/log-mel先行計算。同じPCMバッファへ末尾追加だけを行い、長さを増やしてpushする。
// 認識APIは完成フレームを再利用し、従来通り発話全体でCMVNを実行する（1回だけ再利用可）。
// 最後のtail切捨ては可。PCMの既存部分を変更した場合はresetしてやり直す。
// 全ASR APIは単一instance: pushと認識/照合を別タスクから並行実行してはいけない。
void asr_frontend_begin(const int16_t* pcm);
int asr_frontend_push(int n_samples);  // 戻り値は今回追加計算したフレーム数、負は失敗
void asr_frontend_reset(void);
int asr_frontend_reused_frames(void);  // 直前の認識で再利用したフレーム数

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
