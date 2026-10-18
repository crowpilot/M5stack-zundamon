// ずんだもんLLM（気持ちタグつき）で1回分の返事を作る。Core2 とPCで共通（C99）
//
//   zunda_brain_init(&llm);                      // llm_init 済みの Llm を渡す
//   char reply[256]; int feel;
//   zunda_brain_reply(&llm, "こんにちは。", reply, sizeof(reply), &feel, on_piece, ctx);
//   → reply = "こんにちは！ボクもだいすきなのだ！"、zunda_feel_name(feel) = "うれしい"
//
// 会話の形: <BOS>Ａ：{話しかけ}Ｂ：{返事}〈気持ち〉。1往復ずつ。入力cacheはLlm側で個別にON/OFF可能。
// 生成は m->temperature が0なら greedy、>0ならサンプリング。タグか止める条件（Ａ：/Ｂ：/BOS/EOS）が出た位置、または48トークン目で、
// その位置の logits をタグ8個だけに絞って最大のものを気持ちにする（資料の手順どおり。必ずタグ1つ）
#pragma once
#include "llm_core.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ZUNDA_N_FEEL 8
#define ZUNDA_MAX_REPLY 48
#ifndef ZUNDA_FAST_PREFILL
#define ZUNDA_FAST_PREFILL 1
#endif

// タグ・話者マーカーを語彙から引き、固定先頭のKVを準備する（FAST_PREFILL時）。
// 0=成功、負=見つからない（タグなしモデルなど）
int zunda_brain_init(Llm *m);

// 返事を1つ作る。on_piece は1トークンごとに、表示用の文字片と「次の位置の logits」で呼ばれる（NULL可）
// 戻り値: 生成したトークン数。feel には 0〜7（zunda_feel_name で名前）
typedef void (*zunda_piece_fn)(const char *piece, const float *next_logits, void *ctx);
// 次のforwardの前。音声準備用で、ここでは次のlogits/声色はまだ未確定。
// callback内で同じLlmを操作しないこと。文字片はon_pieceにも同じ順序で渡される。
typedef void (*zunda_prepare_fn)(const char *piece, void *ctx);
int zunda_brain_reply(Llm *m, const char *heard, char *out, int out_size, int *feel, zunda_piece_fn on_piece,
                      void *ctx);
int zunda_brain_reply_ex(Llm *m, const char *heard, char *out, int out_size, int *feel,
                        zunda_prepare_fn on_prepare, zunda_piece_fn on_piece, void *ctx);
// 返事を生成せずpromptだけ処理する（export/診断用）。n_tokensはNULL可。
float *zunda_brain_prefill_prompt(Llm *m, const char *heard, int *n_tokens);
// 同じ入力構築・切り詰め規則を使う。size問い合わせはKV/RNGを変えない。
size_t zunda_brain_prompt_snapshot_size(Llm *m, const char *heard, int skip_prefix);
int zunda_brain_export_prompt(Llm *m, const char *heard, int skip_prefix,
                              const unsigned char identity[32], llm_snapshot_write_fn writer, void *ctx);

// logits をタグ8個だけに絞って一番の気持ち（0〜7）。返事の途中で「今終わるならどの気持ちか」を見るのに使う
int zunda_feel_from_logits(const float *logits);

const char *zunda_feel_name(int feel);
int zunda_brain_last_prompt_tokens(void);  // BOS・Ａ：・Ｂ：込み
int zunda_brain_last_cached_prompt_tokens(void);  // 固定先頭のKVを再利用した数

#ifdef __cplusplus
}
#endif
