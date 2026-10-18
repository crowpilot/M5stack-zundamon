// LLM の返事（漢字かな交じり）→ 読み上げ記法（kana.h の記法）。C99、Core2 とPCで共通
//   "こんにちは！ボクもだいすきなのだ！" → "コンニチワ'、ボクモダイスキナ'ノダ"
// 読み: 辞書（yomi_dict.h、LLM の返事をPCで pyopenjtalk にかけて作った表記→読み）の最長一致。
//       残りのひらがなはカタカナにし、辞書にない漢字・英数字は読まない（飛ばす）。
// アクセント: 端末では本当のアクセントは分からないので簡易規則。句読点・！？で句を切り、
//       「…ノダ」で終わる句は ノ の直前に核（ズンダモンナ'ノダ と同じ形）、それ以外は平板（句末に '）。
//       24モーラを超える句は / で分ける。
#pragma once
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
// 戻り値: 書いたバイト数。skipped には読めずに飛ばした文字数（NULL可）
size_t yomi_to_kana(const char* text, char* out, size_t cap, int* skipped);
// Experimental streaming boundary, in source UTF-8 bytes. Returns zero if no
// prefix of at least minimum_moras has a stable reading. Keeps an unfinished
// dictionary match / UTF-8 character / possible small-kana combination pending.
// This guarantees reading only: ending a TTS chunk here changes final pitch and
// VCV closure. It is NOT permission to claim bit-identical early speech.
size_t yomi_stable_prefix(const char* text, size_t minimum_moras);
#ifdef __cplusplus
}
#endif
