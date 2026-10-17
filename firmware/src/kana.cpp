#include "kana.h"
#include "moras.h"
#include <string.h>
#include <Arduino.h>

const char* const EMO_NAME[EMO_COUNT] = {"neutral", "joy", "sad", "angry", "surprise", "sleepy", "confused"};

const char* parse_emotion(const char* s, Emotion* emo) {
  *emo = EMO_NEUTRAL;
  while (*s == ' ') s++;
  if (*s != '<') return s;
  const char* e = strchr(s, '>');
  if (!e) return s;
  size_t n = e - s - 1;
  for (int i = 0; i < EMO_COUNT; i++) {
    if (strlen(EMO_NAME[i]) == n && strncmp(s + 1, EMO_NAME[i], n) == 0) { *emo = (Emotion)i; break; }
  }
  return e + 1;
}

static bool starts(const char* s, const char* t) { return strncmp(s, t, strlen(t)) == 0; }

// 句ごとにピッチを付与して out に吐き出す
static size_t flush_phrase(MoraTok* ph, size_t n, int acc, bool question, bool fall,
                           MoraTok* out, size_t pos, size_t cap) {
  if (n == 0) return pos;
  // アクセント位置はモーラ数で数える（ッ・ー・ン も1モーラ）
  if (acc <= 0) acc = n;  // 平板
  for (size_t i = 0; i < n; i++) {
    uint8_t p;
    if (acc == 1) p = (i == 0) ? P_H : P_L;
    else          p = (i == 0) ? P_L : ((int)i < acc ? P_H : P_L);
    ph[i].pitch = p;
  }
  if (question)  ph[n - 1].pitch = P_H;
  else if (fall) ph[n - 1].pitch = P_L;  // 文末・休止前の下降
  for (size_t i = 0; i < n && pos < cap; i++) out[pos++] = ph[i];
  return pos;
}

size_t parse_kana(const char* s, MoraTok* out, size_t cap) {
  MoraTok ph[64];
  size_t n = 0, pos = 0;
  int acc = 0;
  bool question = false;
  while (*s) {
    if (*s == '\'') { acc = n; s++; continue; }
    if (*s == '_' || *s == ' ' || *s == '\n' || *s == '\r') { s++; continue; }
    if (*s == '?' || starts(s, "？")) { question = true; s += (*s == '?') ? 1 : 3; continue; }
    if (*s == '/') {
      pos = flush_phrase(ph, n, acc, question, false, out, pos, cap);
      n = 0; acc = 0; question = false; s++; continue;
    }
    if (*s == ',' || starts(s, "、") || starts(s, "。")) {
      pos = flush_phrase(ph, n, acc, question, true, out, pos, cap);
      n = 0; acc = 0; question = false;
      if (pos < cap) out[pos++] = MoraTok{-1, MK_PAUSE, 0};
      s += (*s == ',') ? 1 : 3; continue;
    }
    if (n >= 64) { s++; continue; }
    if (starts(s, "ッ")) { ph[n++] = MoraTok{-1, MK_SOKUON, 0}; s += 3; continue; }
    if (starts(s, "ー")) { ph[n++] = MoraTok{-1, MK_LONG, 0};   s += 3; continue; }
    bool hit = false;
    for (int k = 0; k < N_MORA; k++) {
      int i = MORA_SEARCH[k];
      if (starts(s, MORA_TEXT[i])) { ph[n++] = MoraTok{(int16_t)i, MK_NORMAL, 0}; s += strlen(MORA_TEXT[i]); hit = true; break; }
    }
    if (!hit) {
      Serial.printf("[kana] unknown char at: %.6s\n", s);
      s++; while ((*s & 0xC0) == 0x80) s++;   // UTF-8 1文字スキップ
    }
  }
  pos = flush_phrase(ph, n, acc, question, true, out, pos, cap);
  // 末尾の余計な休止を除く
  while (pos > 0 && out[pos - 1].kind == MK_PAUSE) pos--;
  return pos;
}

void kana_to_display(const char* s, char* out, size_t cap) {
  size_t o = 0;
  while (*s && o + 4 < cap) {
    unsigned char c = *s;
    if (c == '\'' || c == '/' || c == '_') { s++; continue; }
    if (c == '?') { memcpy(out + o, "？", 3); o += 3; s++; continue; }
    if (c == ',') { memcpy(out + o, "、", 3); o += 3; s++; continue; }
    if (c == 0xE3 && (unsigned char)s[1] >= 0x82 && (unsigned char)s[1] <= 0x83) {
      // U+30A1..U+30F6（カタカナ）→ -0x60（ひらがな）
      uint32_t cp = ((c & 0x0F) << 12) | (((unsigned char)s[1] & 0x3F) << 6) | ((unsigned char)s[2] & 0x3F);
      if (cp >= 0x30A1 && cp <= 0x30F6) {
        cp -= 0x60;
        out[o++] = 0xE0 | (cp >> 12);
        out[o++] = 0x80 | ((cp >> 6) & 0x3F);
        out[o++] = 0x80 | (cp & 0x3F);
        s += 3; continue;
      }
    }
    // その他の文字はそのままコピー
    int len = (c < 0x80) ? 1 : (c < 0xE0) ? 2 : (c < 0xF0) ? 3 : 4;
    for (int i = 0; i < len && *s; i++) out[o++] = *s++;
  }
  out[o] = 0;
}
