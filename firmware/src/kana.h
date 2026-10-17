#pragma once
// AquesTalk風記法（VOICEVOX kana互換）のパーサ
//   例: <joy>ボクワ'/ズンダモンナ'ノダ
//   '  … 直前のモーラがアクセント核（平板は句末のモーラに付ける）
//   /  … アクセント句の区切り（休止なし）
//   、 … 休止   ？ … 疑問（句末を上げる）   _ … 無声化（デモでは無視）
//   ッ … 促音（短い無音）   ー … 長音（直前の母音を延ばす）
#include <stdint.h>
#include <stddef.h>

enum MoraKind : uint8_t { MK_NORMAL = 0, MK_PAUSE, MK_SOKUON, MK_LONG };
enum Pitch : uint8_t { P_H = 0, P_L = 1 };

struct MoraTok {
  int16_t idx;      // moras.h のインデックス（MK_NORMAL のとき）
  uint8_t kind;
  uint8_t pitch;
};

enum Emotion : uint8_t { EMO_NEUTRAL = 0, EMO_JOY, EMO_SAD, EMO_ANGRY, EMO_SURPRISE, EMO_SLEEPY, EMO_CONFUSED, EMO_COUNT };
extern const char* const EMO_NAME[EMO_COUNT];

// 先頭の <tag> を読み取り、記法本体の開始位置を返す（タグなしなら NEUTRAL）
const char* parse_emotion(const char* s, Emotion* emo);

// 記法 → モーラ列。戻り値はトークン数（失敗した文字はスキップしてシリアルに警告）
size_t parse_kana(const char* kana, MoraTok* out, size_t cap);

// 表示用：記号を除きカタカナ→ひらがな（UTF-8）
void kana_to_display(const char* kana, char* out, size_t cap);
