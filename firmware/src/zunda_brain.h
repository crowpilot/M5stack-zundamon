// ずんだもんLLM（気持ちタグつき）で1回分の返事を作る。Core2 とPCで共通（C99）
//
//   zunda_brain_init(&llm);                      // llm_init 済みの Llm を渡す
//   char reply[256]; int feel;
//   zunda_brain_reply(&llm, "こんにちは。", reply, sizeof(reply), &feel, on_piece, ctx);
//   → reply = "こんにちは！ボクもだいすきなのだ！"、zunda_feel_name(feel) = "うれしい"
//
// 会話の形: <BOS>Ａ：{話しかけ}Ｂ：{返事}〈気持ち〉。毎回 KV キャッシュを最初から使う（1往復ずつ。学習と同じ形）
// 生成は m->temperature が0なら greedy、>0ならサンプリング。タグか止める条件（Ａ：/Ｂ：/BOS/EOS）が出た位置、または48トークン目で、
// その位置の logits をタグ8個だけに絞って最大のものを気持ちにする（資料の手順どおり。必ずタグ1つ）
#pragma once
#include "llm_core.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ZUNDA_N_FEEL 8
#define ZUNDA_MAX_REPLY 48

// タグ・話者マーカーのトークン番号を語彙の文字列から引く。0=成功、負=見つからない（タグなしモデルなど）
int zunda_brain_init(Llm *m);

// 返事を1つ作る。on_piece は1トークンごとに、表示用の文字片と「次の位置の logits」で呼ばれる（NULL可）
// 戻り値: 生成したトークン数。feel には 0〜7（zunda_feel_name で名前）
typedef void (*zunda_piece_fn)(const char *piece, const float *next_logits, void *ctx);
int zunda_brain_reply(Llm *m, const char *heard, char *out, int out_size, int *feel, zunda_piece_fn on_piece,
                      void *ctx);

// logits をタグ8個だけに絞って一番の気持ち（0〜7）。返事の途中で「今終わるならどの気持ちか」を見るのに使う
int zunda_feel_from_logits(const float *logits);

const char *zunda_feel_name(int feel);  // "ふつう" など（範囲外は "?"）

#ifdef __cplusplus
}
#endif
