#include "yomi.h"

#include <string.h>

#include "yomi_dict.h"

#define MAX_PHRASE_MORA 24

typedef struct {
  char* out;
  size_t cap, len;
  char seg[256];  // 今の句（カタカナ）
  size_t seg_len;
  int seg_mora;
} Buf;

static void put(Buf* b, const char* s, size_t n) {
  if (b->len + n < b->cap) {
    memcpy(b->out + b->len, s, n);
    b->len += n;
    b->out[b->len] = 0;
  }
}

static int is_small(const char* s) {  // 前の字と1モーラになる小書き
  static const char* const k[] = {"ャ", "ュ", "ョ", "ァ", "ィ", "ゥ", "ェ", "ォ"};
  for (int i = 0; i < 8; i++)
    if (strncmp(s, k[i], 3) == 0) return 1;
  return 0;
}

// 句を書き出す。sep: ',' 句読点（後ろに「、」）/ '?' 疑問（「？、」）/ '/' 長すぎて切るとき
static void flush(Buf* b, int sep) {
  if (b->seg_len == 0) return;
  b->seg[b->seg_len] = 0;
  size_t acc_at = b->seg_len;  // ' を入れるバイト位置（既定は句末＝平板）
  if (b->seg_mora > 2 && b->seg_len >= 6 && strcmp(b->seg + b->seg_len - 6, "ノダ") == 0) acc_at = b->seg_len - 6;
  put(b, b->seg, acc_at);
  put(b, "'", 1);
  put(b, b->seg + acc_at, b->seg_len - acc_at);
  if (sep == '/') put(b, "/", 1);
  else if (sep == '?') put(b, "？、", 6);
  else put(b, "、", 3);
  b->seg_len = 0;
  b->seg_mora = 0;
}

// カタカナ1字（UTF-8 3バイト）を句に足す
static void add_kata(Buf* b, const char* k) {
  // 実機の単位表にない字を近い音に寄せる（ヲは VOICEVOX・実機とも読めないので オ。LLM_SPEC.md §2）
  static const char* const from[] = {"ヲ", "ヂ", "ヅ", "ヮ", "ヰ", "ヱ", "ヵ", "ヶ"};
  static const char* const to[] = {"オ", "ジ", "ズ", "ワ", "イ", "エ", "カ", "ケ"};
  for (int i = 0; i < 8; i++)
    if (strncmp(k, from[i], 3) == 0) { k = to[i]; break; }
  if (!is_small(k)) {
    if (b->seg_mora >= MAX_PHRASE_MORA) flush(b, '/');
    b->seg_mora++;
  }
  if (b->seg_len + 3 < sizeof(b->seg)) {
    memcpy(b->seg + b->seg_len, k, 3);
    b->seg_len += 3;
  }
}

static void add_kata_str(Buf* b, const char* s) {
  for (; s[0] && s[1] && s[2]; s += 3) add_kata(b, s);
}

static unsigned utf8_cp(const unsigned char* s, int* n) {
  if (s[0] < 0x80) { *n = 1; return s[0]; }
  if ((s[0] & 0xE0) == 0xC0 && (s[1] & 0xC0) == 0x80) { *n = 2; return ((s[0] & 0x1F) << 6) | (s[1] & 0x3F); }
  if ((s[0] & 0xF0) == 0xE0 && (s[1] & 0xC0) == 0x80 && (s[2] & 0xC0) == 0x80) {
    *n = 3;
    return ((s[0] & 0x0F) << 12) | ((s[1] & 0x3F) << 6) | (s[2] & 0x3F);
  }
  if ((s[0] & 0xF8) == 0xF0 && s[1] && s[2] && s[3]) { *n = 4; return 0x10000; }
  *n = 1;  // 壊れたバイト（LLM のバイト片）は1バイトずつ飛ばす
  return 0xFFFD;
}

size_t yomi_to_kana(const char* text, char* out, size_t cap, int* skipped) {
  Buf b;
  memset(&b, 0, sizeof(b));
  b.out = out;
  b.cap = cap;
  if (cap) out[0] = 0;
  int skip = 0;
  const char* s = text;
  while (*s) {
    int hit = -1;
    for (int i = 0; i < YOMI_N; i++) {  // 長い順に並んでいるので最初に当たったものが最長
      size_t n = strlen(YOMI_SURF[i]);
      if (strncmp(s, YOMI_SURF[i], n) == 0) { hit = i; break; }
    }
    if (hit >= 0) {
      add_kata_str(&b, YOMI_PRON[hit]);
      s += strlen(YOMI_SURF[hit]);
      continue;
    }
    int n;
    unsigned cp = utf8_cp((const unsigned char*)s, &n);
    char k[3];
    if (cp >= 0x3041 && cp <= 0x3096) cp += 0x60;  // ひらがな → カタカナ
    if ((cp >= 0x30A1 && cp <= 0x30F6) || cp == 0x30FC) {
      k[0] = (char)(0xE0 | (cp >> 12));
      k[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
      k[2] = (char)(0x80 | (cp & 0x3F));
      add_kata(&b, k);
    } else if (cp == 0xFF1F || cp == '?') {
      flush(&b, '?');
    } else if (cp == 0x3001 || cp == 0x3002 || cp == 0xFF01 || cp == 0xFF0C || cp == 0xFF0E || cp == '!' ||
               cp == ',' || cp == '.' || cp == '\n' || cp == 0x2026) {
      flush(&b, ',');
    } else if (cp != ' ' && cp != 0x3000) {
      skip++;
    }
    s += n;
  }
  flush(&b, ',');
  // 末尾の区切りを落とす
  while (b.len >= 3 && strcmp(b.out + b.len - 3, "、") == 0) b.out[b.len -= 3] = 0;
  if (skipped) *skipped = skip;
  return b.len;
}
