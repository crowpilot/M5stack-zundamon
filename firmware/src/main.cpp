// M5Stack Core2: 話しかける → 聞き取る（kana_asr）→ ずんだもんLLMが返事 → 画面に表示 → 読み上げ
// すべて Core2 の上で動く（Wi-Fi もサーバーも使わない）。
// 読み上げ: 返事（漢字かな交じり）を yomi.c で記法に変え、VCV単位接続TTS（tts.cpp）で合成する。
//   音声バンクはフラッシュの zbank 領域（tools/flash_voice.sh で書く）。無ければ画面表示だけで動く。
//
// LLM に渡す話しかけ: 登録フレーズに当てはまれば その表示用テキスト（漢字かな交じり）、なければ聞こえたままのかな。
//
// ボタン: A=雑音レベル再測定 / B=直前の返事をもう一度読み上げ / C=直前の話しかけにもう一度返事（温度0.8で別の返事）
// シリアル(115200):
//   受信 "@こんにちは\n" … マイクを通さずにその文で返事させる（試験用）
//   受信 "!<joy>オハヨオナ'/ノダ\n" … 記法をそのまま読み上げる（TTS だけの確認）
//   送信 1往復ごとに1行のJSON
//   {"heard":"こんにちは","kana":"こんにちわ","phrase":"こんにちは","asr_ms":350,
//    "reply":"こんにちは！ボクもだいすきなのだ！","feel":"うれしい","llm_ms":2100,"tokens":11,
//    "say":"コンニチワ'、ボクモダイスキナ'ノダ","tts_ms":300,"audio_ms":2900,...}
#include <M5Unified.h>
#include <atomic>

#ifndef ZUNDA_VERSION
#define ZUNDA_VERSION "dev"
#endif

#include "asr_core.h"
#include "asr_model_data.h"
#include "llm_core.h"
#include "llm_model_data.h"  // model_bin[], tok_bin[], MODEL_NAME（gen_headers.py が models/ から作る）
#include "phrases.h"
#include "kana.h"
#include "tts.h"
#include "yomi.h"
#include "zunda_brain.h"
#include "trial_control.h"
#include "prompt_identity.h"
#if __has_include("prompt_catalog.h")
#include "prompt_catalog.h"
#else
static const LlmPromptSnapshot* const prompt_catalog_entries = nullptr;
static constexpr size_t prompt_catalog_count = 0;
#endif

// ---- 設定（聞き取りは kana_asr と同じ値） --------------------------------------
static constexpr uint32_t SR = 16000;
static constexpr int CHUNK = 320;                 // 20ms
static constexpr int MAX_SAMPLES = SR * 7;        // 1発話の上限 7秒（前後の余白を含む）
static constexpr int PREROLL_CHUNKS = 25;         // 発話開始前 500ms も含める
static constexpr int TAIL_CHUNKS = 25;            // 発話終了後も 500ms 残す
static constexpr int START_CHUNKS = 3;            // 60ms 続いたら発話開始
static constexpr int END_CHUNKS = 35;             // 700ms 静かなら発話終了
static constexpr int MIN_SPEECH_CHUNKS = 12;      // 240ms 未満は捨てる
static constexpr float START_RATIO = 3.0f;
static constexpr float END_RATIO = 1.8f;
static constexpr float MIN_LEVEL = 40.0f;
static constexpr int CALIB_CHUNKS = 50;           // 起動時1秒で雑音を測る
static constexpr float MATCH_GAP = 2.0f;           // kana_asr v2-small・29フレーズで決めた値
static constexpr float MATCH_MARGIN = 1.0f;
static constexpr int LLM_STEPS = 256;             // KVキャッシュ長

static constexpr uint32_t DEAF_MS = 500;          // 喋り終わってからこの間はマイクの音を無視（自分の声に反応しない）

static Llm llm;
static UnitTTS tts;
static bool tts_ok = false;
static uint32_t deaf_until = 0;
// Publish enabled flags only after their resources and cache setup succeed.
static uint32_t trial_flags = 0;
static const char* trial_name = "default";
static bool replay_active = false;
static bool serial_hold = false;  // diagnostic sessions keep ambient mic input out of measurements
static unsigned response_count = 0;
static void* shared_arena = nullptr;
static size_t shared_arena_bytes = 0;
static int16_t* fixture_pcm = nullptr;
static int fixture_samples = 0;
static uint32_t last_voice_ms = 0, endpoint_ms = 0;
static int match_ms = 0, asr_wait_ms = 0, frontend_us = 0;
static bool used_preview = false;
static bool frontend_active = false;
static int frontend_covered_samples = 0;
static unsigned preview_cancellations = 0;
static std::atomic<unsigned> dropped_chunks{0};
static std::atomic<uint32_t> mic_gap_max_us{0};
static uint32_t mic_callback_us = 0;
static std::atomic<unsigned> mic_callbacks{0};
static float mic_peak_level = 0;

// ---- メモリ確保 ---------------------------------------------------------------
static void* asr_alloc(size_t n, int fast) {
  void* p = nullptr;
  if (fast) {
    p = heap_caps_malloc(n, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (!p) p = heap_caps_malloc(n, MALLOC_CAP_SPIRAM);
  } else {
    p = heap_caps_malloc(n, MALLOC_CAP_SPIRAM);
    if (!p) p = heap_caps_malloc(n, MALLOC_CAP_8BIT);
  }
  return p;
}

static void* llm_alloc(size_t n) {
  void* p = heap_caps_malloc(n, MALLOC_CAP_SPIRAM);
  return p ? p : malloc(n);
}

// ---- マイク: 録音済みチャンクを受け取るキュー --------------------------------
static constexpr int NBUF = 8;
static int16_t* chunk_buf[NBUF];
static int rec_idx = 0;
static int16_t* volatile done_q[NBUF];
static std::atomic<uint32_t> q_head{0}, q_tail{0};

static void on_chunk(void*, void* data, size_t) {
  mic_callbacks.fetch_add(1, std::memory_order_relaxed);
  uint32_t now = micros();
  if (mic_callback_us) {
    uint32_t gap = now - mic_callback_us;
    if (gap > mic_gap_max_us.load(std::memory_order_relaxed)) mic_gap_max_us.store(gap, std::memory_order_relaxed);
  }
  mic_callback_us = now;
  uint32_t h = q_head.load(std::memory_order_relaxed);
  if (h - q_tail.load(std::memory_order_acquire) < NBUF) {
    done_q[h % NBUF] = (int16_t*)data;
    q_head.store(h + 1, std::memory_order_release);
  } else dropped_chunks.fetch_add(1, std::memory_order_relaxed);
}

static int16_t* pop_chunk() {
  uint32_t t = q_tail.load(std::memory_order_relaxed);
  if (t == q_head.load(std::memory_order_acquire)) return nullptr;
  int16_t* p = done_q[t % NBUF];
  q_tail.store(t + 1, std::memory_order_release);
  return p;
}

static void mic_start() {
  if (replay_active || serial_hold) return;
  M5.Speaker.end();  // Core2はマイクとスピーカーが同じI2Sを使う
  q_head = 0;
  q_tail = 0;
  mic_callback_us = 0;
  mic_gap_max_us.store(0);
  mic_callbacks.store(0);
  mic_peak_level = 0;
  M5.Mic.setBufferReleaseCallback(nullptr, on_chunk);
  M5.Mic.begin();
}

static void mic_stop() {
  if (replay_active) return;
  while (M5.Mic.isRecording()) M5.delay(1);
  M5.Mic.end();
  M5.Mic.setBufferReleaseCallback(nullptr, nullptr);
}

// ---- 発話区間検出の状態 -------------------------------------------------------
enum class St { Calib, Idle, Speech };
static St st = St::Calib;
static float noise = 0;
static int calib_n = 0, above = 0, below = 0, speech_chunks = 0;
static int16_t* utt;
static int utt_len = 0;
static int16_t* preroll;
static int pre_pos = 0, pre_fill = 0;
static float dc = 0;

static unsigned asr_flags() {
  return ((trial_flags & TR_ASR_NO_CONF) ? ASR_OPT_SKIP_UNUSED_CONFIDENCE : 0) |
         ((trial_flags & TR_ASR_RAW) ? ASR_OPT_RAW_MATCH : 0);
}

static TtsRenderOptions render_options() {
  TtsRenderOptions o;
  if (trial_flags & TR_TTS_PAD) o.leading_silence_ms = 20;
  o.compute_envelope = !(trial_flags & TR_TTS_NO_ENV);
  return o;
}

static bool direct_pcm() { return TRIAL_NO_INTERNAL_PCM || (trial_flags & TR_TTS_DIRECT); }

static float chunk_level(const int16_t* x) {
  double e = 0;
  for (int i = 0; i < CHUNK; i++) {
    dc += 0.001f * (x[i] - dc);
    float v = x[i] - dc;
    e += v * v;
  }
  return sqrtf(e / CHUNK);
}

// ---- 画面（320x240） -----------------------------------------------------------
//   0-30   状態        34-42 音量メーター
//   48-    あなた：聞き取った文（水色）／ 聞こえたかな（灰）
//   ずんだもん：返事（白、24px、生成しながら表示）
//   222-   気持ち・時間
static constexpr int REPLY_Y = 104;

// 返事を表示している途中（読み上げ開始時など）でも呼べるよう、文字の位置・フォント・色を元に戻す
static void draw_status(const char* s, uint16_t color) {
  auto& d = M5.Display;
  int32_t cx = d.getCursorX(), cy = d.getCursorY();
  const lgfx::IFont* font = d.getFont();
  auto style = d.getTextStyle();
  d.fillRect(0, 0, 320, 30, TFT_BLACK);
  d.setFont(&fonts::efontJA_16);
  d.setTextColor(color);
  d.setCursor(4, 6);
  d.print(s);
  d.setFont(font);
  d.setTextStyle(style);
  d.setCursor(cx, cy);
}

static void draw_meter(float lv) {
  int w = (int)(lv / 8.0f);
  if (w > 316) w = 316;
  int th = (int)(fmaxf(noise * START_RATIO, MIN_LEVEL) / 8.0f);
  M5.Display.fillRect(2, 34, 316, 8, 0x2104);
  M5.Display.fillRect(2, 34, w, 8, st == St::Speech ? TFT_GREEN : TFT_DARKGREY);
  if (th < 316) M5.Display.drawFastVLine(2 + th, 32, 12, TFT_YELLOW);
}

static void draw_heard(const char* heard, const char* kana) {
  M5.Display.fillRect(0, 46, 320, 194, TFT_BLACK);
  M5.Display.setTextWrap(true, false);
  M5.Display.setFont(&fonts::efontJA_16);
  M5.Display.setTextColor(TFT_CYAN);
  M5.Display.setCursor(4, 50);
  M5.Display.printf("あなた：%s", heard);
  if (kana && kana[0] && strcmp(kana, heard) != 0) {
    M5.Display.setFont(&fonts::efontJA_12);
    M5.Display.setTextColor(TFT_DARKGREY);
    M5.Display.setCursor(4, 84);
    M5.Display.printf("（きこえた音：%s）", kana);
  }
  M5.Display.setFont(&fonts::efontJA_24);
  M5.Display.setTextColor(TFT_GREENYELLOW);
  M5.Display.setCursor(4, REPLY_Y);
  M5.Display.setTextColor(TFT_WHITE);
}

static void talk_feed(const char* piece, const float* next_logits);

static void on_piece(const char* piece, const float* next_logits, void*) {
  if (M5.Display.getCursorY() < 200) M5.Display.print(piece);  // はみ出す分は表示しない（シリアルには全文）
  talk_feed(piece, next_logits);
}

static void draw_footer(const char* feel, int asr_ms, int llm_ms, int tokens) {
  M5.Display.fillRect(0, 216, 320, 24, TFT_BLACK);
  M5.Display.setFont(&fonts::efontJA_16);
  M5.Display.setTextColor(TFT_PINK);
  M5.Display.setCursor(4, 220);
  M5.Display.printf("〈%s〉", feel);
  M5.Display.setFont(&fonts::efontJA_12);
  M5.Display.setTextColor(TFT_DARKGREY);
  M5.Display.setCursor(120, 224);
  if (asr_ms >= 0)
    M5.Display.printf("認識%dms 返事%dms %dtok", asr_ms, llm_ms, tokens);
  else
    M5.Display.printf("返事%dms %dtok", llm_ms, tokens);
}

// JSON 用に " と \ をエスケープして出す
static void json_str(const char* key, const char* s) {
  Serial.printf("\"%s\":\"", key);
  for (; *s; s++) {
    if (*s == '"' || *s == '\\') Serial.write('\\');
    if (*s == '\n') {
      Serial.print("\\n");
      continue;
    }
    Serial.write(*s);
  }
  Serial.print("\",");
}

// ---- 読み上げ -----------------------------------------------------------------
// LLM の気持ち（ふつう うれしい ドヤ びっくり かなしい ぷんぷん こまった ねむい）→ TTS の声色
static const Emotion FEEL_EMO[ZUNDA_N_FEEL] = {EMO_NEUTRAL, EMO_JOY,   EMO_JOY,      EMO_SURPRISE,
                                               EMO_SAD,     EMO_ANGRY, EMO_CONFUSED, EMO_SLEEPY};
// 声色は再生レートで変える
static const float EMO_RATE[EMO_COUNT] = {1.00f, 1.06f, 0.94f, 0.98f, 1.08f, 0.90f, 1.00f};
static MoraTok seq[256];
static int last_tts_ms = 0, last_audio_ms = 0;
static char last_say[512];
static Emotion last_emo = EMO_NEUTRAL;
static size_t render_standalone(const MoraTok* tokens, size_t n);

// 記法を合成して再生し終わるまで待つ。マイクは止めておくこと（Core2 は I2S を共有）
static void speak_kana(const char* kana, Emotion emo) {
  last_tts_ms = last_audio_ms = 0;
  if (!tts_ok || !kana[0]) return;
  uint32_t t0 = millis();
  size_t n = parse_kana(kana, seq, 256);
  size_t len = render_standalone(seq, n);
  last_tts_ms = (int)(millis() - t0);
  if (!len) return;
  float rate = EMO_RATE[emo];
  last_audio_ms = (int)(len * 1000 / (tts.sampleRate() * rate));
  draw_status("ずんだもんがしゃべってる", TFT_GREENYELLOW);
  M5.Speaker.begin();
  M5.Speaker.setVolume(180);
  M5.Speaker.playRaw(tts.pcm(), len, (uint32_t)(tts.sampleRate() * rate), false, 1, 0, true);
  while (M5.Speaker.isPlaying()) M5.delay(10);
  M5.Speaker.end();
  deaf_until = millis() + DEAF_MS;
}

// ---- 返事を作りながら読み上げる -------------------------------------------------
// LLM の文字片を貯め、句読点（、。！？）が来たらそこまでを記法にして合成し、スピーカーの待ち行列に積む。
// 再生は M5Unified のスピーカータスクが裏で行うので、その間も LLM は次の句を作り続けられる。
// スピーカーの待ち行列は1チャンネル2枠なので、PCM は3枚の使い回し（再生中・待ち・合成中）。
static constexpr int TALK_BUFS = 3;
static constexpr size_t TALK_MAX = 16000 * 6;
static constexpr size_t TALK_FORCE_CUT = 90;  // 句読点が来なくてもこのバイト数（ひらがな約30字）で区切る
static int16_t* talk_buf[TALK_BUFS];
static int talk_next = 0;
static char talk_pending[400];
static size_t talk_len = 0;
static bool talk_on = false, talk_emo_fixed = false;
static Emotion talk_emo = EMO_NEUTRAL;
static uint32_t talk_t0 = 0;
static int talk_first_piece_ms = -1;
static int talk_first_ms = -1, talk_chunks = 0, talk_tts_ms = 0, talk_audio_ms = 0, talk_skip = 0, talk_missing = 0;
static int first_pcm_nonzero_ms = -1, voice_to_enqueue_ms = -1;
static uint32_t pcm_hash = 2166136261u;
static SemaphoreHandle_t tts_job_start = nullptr, tts_job_done = nullptr;
static bool tts_job_active = false;
struct PreparedAudio {
  char text[400], kana[512];
  int skipped, missing, ms;
  size_t len;
  int16_t* pcm;
  bool external;
};
static PreparedAudio prepared;

static size_t render_standalone(const MoraTok* tokens, size_t n) {
  return direct_pcm() ? tts.renderTo(tokens, n, talk_buf[0], TALK_MAX, render_options())
                      : tts.render(tokens, n, render_options());
}

static void prepare_audio(const char* text) {
  strlcpy(prepared.text, text, sizeof(prepared.text));
  prepared.skipped = 0;
  yomi_to_kana(text, prepared.kana, sizeof(prepared.kana), &prepared.skipped);
  prepared.len = 0; prepared.missing = 0; prepared.ms = 0;
  prepared.external = direct_pcm();
  if (!prepared.kana[0]) return;
  uint32_t start = millis();
  size_t n = parse_kana(prepared.kana, seq, 256);
  prepared.len = prepared.external
      ? tts.renderTo(seq, n, talk_buf[talk_next], TALK_MAX, render_options())
      : tts.render(seq, n, render_options());
  prepared.pcm = const_cast<int16_t*>(tts.pcm());
  prepared.missing = tts.missing();
  prepared.ms = (int)(millis() - start);
}

static void tts_prepare_task(void*) {
  for (;;) {
    xSemaphoreTake(tts_job_start, portMAX_DELAY);
    // text is immutable until the main task consumes the completion semaphore.
    char text[sizeof(prepared.text)];
    strlcpy(text, prepared.text, sizeof(text));
    prepare_audio(text);
    xSemaphoreGive(tts_job_done);
  }
}

static void wait_tts_prepare() {
  if (tts_job_active) {
    xSemaphoreTake(tts_job_done, portMAX_DELAY);
    tts_job_active = false;
  }
}

static void talk_begin(Emotion emo, bool emo_fixed) {
  wait_tts_prepare();
  talk_len = 0;
  talk_pending[0] = 0;
  last_say[0] = 0;
  talk_emo = emo;
  talk_emo_fixed = emo_fixed;
  talk_t0 = millis();
  talk_first_ms = -1;
  talk_first_piece_ms = -1;
  first_pcm_nonzero_ms = voice_to_enqueue_ms = -1;
  pcm_hash = 2166136261u;
  talk_chunks = talk_tts_ms = talk_audio_ms = talk_skip = talk_missing = 0;
  talk_on = tts_ok && talk_buf[TALK_BUFS - 1];
  if (!talk_on) return;
  M5.Speaker.begin();
  M5.Speaker.setVolume(180);
}

// 句1つ（漢字かな交じり）を合成して待ち行列に積む
static void talk_chunk(const char* text) {
  bool was_prepared = tts_job_active;
  wait_tts_prepare();
  if (!was_prepared || strcmp(text, prepared.text) != 0) prepare_audio(text);
  talk_skip += prepared.skipped;
  if (!prepared.kana[0]) return;
  if (last_say[0]) strlcat(last_say, "、", sizeof(last_say));
  strlcat(last_say, prepared.kana, sizeof(last_say));
  size_t len = prepared.len;
  talk_tts_ms += prepared.ms;
  talk_missing += prepared.missing;
  if (!len) return;
  if (len > TALK_MAX) len = TALK_MAX;
  float rate = EMO_RATE[talk_emo];
  // 待ち行列が2枠とも埋まっていたら、1枠空くまで待つ（3枚目のバッファがまだ再生に使われていないことの保証）
  while (M5.Speaker.isPlaying(0) >= 2) M5.delay(2);
  int16_t* buf = talk_buf[talk_next];
  talk_next = (talk_next + 1) % TALK_BUFS;
  if (!prepared.external) memcpy(buf, prepared.pcm, len * sizeof(int16_t));
  // Only scan the leading silence before enqueueing; hash after the timestamp.
  size_t first_nonzero = len;
  for (size_t i = 0; i < len; i++) {
    if (buf[i] != 0) { first_nonzero = i; break; }
  }
  M5.Speaker.playRaw(buf, len, (uint32_t)(tts.sampleRate() * rate), false, 1, 0, false);
  if (talk_first_ms < 0) {
    talk_first_ms = (int)(millis() - talk_t0);
    first_pcm_nonzero_ms = (int)(first_nonzero * 1000 / (tts.sampleRate() * rate));
    if (endpoint_ms) voice_to_enqueue_ms = (int)(millis() - last_voice_ms);
    draw_status("ずんだもんがしゃべってる", TFT_GREENYELLOW);
  }
  for (size_t i = 0; i < len; i++) {
    pcm_hash = (pcm_hash ^ (uint8_t)buf[i]) * 16777619u;
    pcm_hash = (pcm_hash ^ (uint8_t)((uint16_t)buf[i] >> 8)) * 16777619u;
  }
  talk_chunks++;
  talk_audio_ms += (int)(len * 1000 / (tts.sampleRate() * rate));
}

static bool is_punct_at(const char* s, size_t* n) {
  static const char* const P[] = {"、", "。", "！", "？", "…", "!", "?", "\n"};
  for (const char* p : P) {
    size_t l = strlen(p);
    if (strncmp(s, p, l) == 0) {
      *n = l;
      return true;
    }
  }
  return false;
}

static size_t first_cut(const char* text, size_t len) {
  for (size_t i = 0, n; i < len; i++) {
    if (is_punct_at(text + i, &n)) {
      size_t cut = i + n;
      while (cut < len && is_punct_at(text + cut, &n)) cut += n;
      return cut;
    }
  }
  if (len >= TALK_FORCE_CUT) {
    while (len > 0 && (text[len] & 0xC0) == 0x80) len--;
    return len;
  }
  if (trial_flags & TR_EARLY_READ) return yomi_stable_prefix(text, 6);
  return 0;
}

static void on_prepare(const char* piece, void*) {
  if (!(trial_flags & TR_TTS_PARALLEL) || !talk_on || talk_chunks ||
      tts_job_active || !tts_job_start) return;
  char pending[sizeof(talk_pending)];
  strlcpy(pending, talk_pending, sizeof(pending));
  if (strlen(pending) + strlen(piece) >= sizeof(pending)) return;
  strlcat(pending, piece, sizeof(pending));
  size_t cut = first_cut(pending, strlen(pending));
  if (!cut) return;
  pending[cut] = 0;
  strlcpy(prepared.text, pending, sizeof(prepared.text));
  tts_job_active = true;
  xSemaphoreGive(tts_job_start);
}

// 貯まった文字から、句読点で終わる句を順に読み上げる（final なら残りも全部）
static void talk_flush(bool final) {
  for (;;) {
    size_t cut = first_cut(talk_pending, talk_len);
    if (!cut && final) cut = talk_len;
    if (!cut) return;
    char chunk[sizeof(talk_pending)];
    memcpy(chunk, talk_pending, cut);
    chunk[cut] = 0;
    memmove(talk_pending, talk_pending + cut, talk_len - cut + 1);
    talk_len -= cut;
    talk_chunk(chunk);
  }
}

static void talk_feed(const char* piece, const float* next_logits) {
  if (talk_first_piece_ms < 0) talk_first_piece_ms = (int)(millis() - talk_t0);  // 話しかけの読み込み＋1トークン目
  if (!talk_on) return;
  size_t l = strlen(piece);
  if (talk_len + l < sizeof(talk_pending)) {
    memcpy(talk_pending + talk_len, piece, l + 1);
    talk_len += l;
  }
  // 声色は最初の句を喋る前に決める（その時点で返事が終わるとしたらどの気持ちか）。最後のタグとずれることがある
  if (!talk_emo_fixed && next_logits) {
    size_t dummy;
    bool has_punct = false;
    for (size_t i = 0; i < talk_len && !has_punct; i++) has_punct = is_punct_at(talk_pending + i, &dummy);
    if (has_punct || talk_len >= TALK_FORCE_CUT ||
        ((trial_flags & TR_EARLY_READ) && yomi_stable_prefix(talk_pending, 6))) {
      talk_emo = FEEL_EMO[zunda_feel_from_logits(next_logits)];
      talk_emo_fixed = true;
    }
  }
  talk_flush(false);
}

static void talk_end() {
  if (!talk_on) return;
  talk_flush(true);
  wait_tts_prepare();
  while (M5.Speaker.isPlaying()) M5.delay(5);
  M5.Speaker.end();
  talk_on = false;
  deaf_until = millis() + DEAF_MS;
}

// 文字列をまとめて読み上げる（ボタンB の読み直し用）
static void talk_text(const char* text, Emotion emo) {
  talk_begin(emo, true);
  talk_feed(text, nullptr);
  talk_end();
}

// ---- 返事を作る ---------------------------------------------------------------
static char last_heard[320];
static char last_reply[256];

static void respond(const char* heard, const char* kana, const char* phrase, int asr_ms) {
  strlcpy(last_heard, heard, sizeof(last_heard));
  draw_status("ずんだもんが考え中…", TFT_ORANGE);
  draw_heard(heard, kana);
  M5.Display.setTextColor(TFT_GREENYELLOW);
  M5.Display.print("ずんだもん：");
  M5.Display.setTextColor(TFT_WHITE);
  char* reply = last_reply;
  int feel = 0;
  uint32_t t0 = millis();
  talk_begin(EMO_NEUTRAL, false);
  int tokens = zunda_brain_reply_ex(&llm, heard, reply, sizeof(last_reply), &feel,
                                   on_prepare, on_piece, nullptr);
  int llm_ms = (int)(millis() - t0);
  draw_footer(zunda_feel_name(feel), asr_ms, llm_ms, tokens);
  talk_end();
  int talk_total_ms = (int)(millis() - t0);
  last_emo = talk_emo;

  Serial.print("{");
  json_str("trial", trial_name);
  json_str("heard", heard);
  if (kana) json_str("kana", kana);
  if (phrase) json_str("phrase", phrase);
  if (asr_ms >= 0) Serial.printf("\"asr_ms\":%d,", asr_ms);
  json_str("reply", reply);
  json_str("feel", zunda_feel_name(feel));
  json_str("say", last_say);
  json_str("voice", EMO_NAME[talk_emo]);
  Serial.printf("\"prompt_tokens\":%d,\"first_piece_ms\":%d,", zunda_brain_last_prompt_tokens(), talk_first_piece_ms);
  Serial.printf("\"cached_prompt_tokens\":%d,", zunda_brain_last_cached_prompt_tokens());
  Serial.printf("\"prefill_source\":%d,\"pcm_hash\":%u,\"pcm_leading_ms\":%d,"
                "\"voice_to_enqueue_ms\":%d,\"match_ms\":%d,\"asr_wait_ms\":%d,"
                "\"frontend_us\":%d,\"frontend_frames\":%d,\"preview_used\":%s,"
                "\"preview_cancellations\":%u,\"dropped_chunks\":%u,\"mic_gap_max_us\":%u,",
                llm_last_prefill_source(&llm), pcm_hash, first_pcm_nonzero_ms,
                voice_to_enqueue_ms, match_ms, asr_wait_ms, frontend_us,
                asr_frontend_reused_frames(), used_preview ? "true" : "false",
                preview_cancellations, dropped_chunks.load(), mic_gap_max_us.load());
  Serial.printf("\"first_sound_ms\":%d,\"chunks\":%d,\"tts_ms\":%d,\"audio_ms\":%d,\"done_ms\":%d,\"yomi_skip\":%d,"
                "\"tts_missing\":%d,",
                talk_first_ms, talk_chunks, talk_tts_ms, talk_audio_ms, talk_total_ms, talk_skip,
                tts_ok ? talk_missing : -1);
  Serial.printf("\"llm_ms\":%d,\"tokens\":%d,\"temp\":%.1f,\"free_internal\":%u,\"free_psram\":%u}\n", llm_ms,
                tokens, llm.temperature, (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
                (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
  response_count++;
}

// ---- 2コア並列（認識中だけ使う） ---------------------------------------------
static asr_job_fn job_fn;
static void* job_arg;
static SemaphoreHandle_t job_start, job_done;

static void worker_task(void*) {
  for (;;) {
    xSemaphoreTake(job_start, portMAX_DELAY);
    job_fn(job_arg);
    xSemaphoreGive(job_done);
  }
}

static void run2(asr_job_fn fn, void* a, void* b) {
  job_fn = fn;
  job_arg = a;
  xSemaphoreGive(job_start);
  fn(b);
  xSemaphoreTake(job_done, portMAX_DELAY);
}

static void start_worker() {
  job_start = xSemaphoreCreateBinary();
  job_done = xSemaphoreCreateBinary();
  if (job_start && job_done &&
      xTaskCreatePinnedToCore(worker_task, "asr1", 4096, nullptr, 2, nullptr, 0) == pdPASS)
    asr_set_parallel(run2);
}

// Speculative recognition starts only after the final 500 ms of input exist.
// Recording continues; speech resumption invalidates the result, never the PCM.
static SemaphoreHandle_t preview_start = nullptr;
static std::atomic<int> preview_state{0};  // 0=idle, 1=running, 2=complete
static bool preview_valid = false;
static int preview_n = 0, preview_result = -1, preview_ms = 0;
static char preview_kana[256];
static uint32_t preview_started_ms = 0;

static void preview_task(void*) {
  for (;;) {
    xSemaphoreTake(preview_start, portMAX_DELAY);
    float confidence;
    uint32_t start = millis();
    preview_result = asr_recognize(utt, preview_n, preview_kana, sizeof(preview_kana),
        (trial_flags & TR_ASR_NO_CONF) ? nullptr : &confidence);
    preview_ms = (int)(millis() - start);
    preview_state.store(2, std::memory_order_release);
  }
}

static void settle_preview() {
  while (preview_state.load(std::memory_order_acquire) == 1) M5.delay(1);
}

static void reset_preview() {
  settle_preview();
  preview_state.store(0, std::memory_order_release);
  preview_valid = false;
  if (TRIAL_SHARED_ARENA) asr_release_scratch();
  else asr_frontend_reset();
  frontend_active = false;
  frontend_covered_samples = 0;
}

static void start_trial_workers() {
  preview_start = xSemaphoreCreateBinary();
  if (!preview_start || xTaskCreatePinnedToCore(preview_task, "asr_preview", 8192, nullptr,
                                               1, nullptr, 1) != pdPASS) preview_start = nullptr;
  tts_job_start = xSemaphoreCreateBinary();
  tts_job_done = xSemaphoreCreateBinary();
  if (!tts_job_start || !tts_job_done ||
      xTaskCreatePinnedToCore(tts_prepare_task, "tts_prepare", 8192, nullptr,
                             1, nullptr, 0) != pdPASS) tts_job_start = nullptr;
}

// ---- 認識 → 返事 --------------------------------------------------------------
static void finish_utterance() {
  if (speech_chunks < MIN_SPEECH_CHUNKS) return;
  draw_status("ききとり中…", TFT_ORANGE);
  char kana[256];
  float conf = 0;
  uint32_t t0 = millis();
  settle_preview();
  used_preview = preview_state.load() == 2 && preview_valid && preview_n == utt_len;
  int n, asr_ms;
  if (used_preview) {
    strlcpy(kana, preview_kana, sizeof(kana));
    n = preview_result;
    asr_ms = preview_ms;
  } else {
    if (preview_state.load() != 0) asr_frontend_reset();
    n = asr_recognize(utt, utt_len, kana, sizeof(kana),
                     (trial_flags & TR_ASR_NO_CONF) ? nullptr : &conf);
    asr_ms = (int)(millis() - t0);
  }
  asr_wait_ms = (int)(millis() - t0);
  preview_state.store(0);
  preview_valid = false;
  if (n < 0) {
    if (TRIAL_SHARED_ARENA) asr_release_scratch();
    Serial.printf("{\"error\":\"%s\"}\n", asr_error());
    return;
  }
  if (n == 0) {  // 何も聞き取れなければ返事しない
    if (TRIAL_SHARED_ARENA) asr_release_scratch();
    Serial.printf("{\"kana\":\"\",\"asr_ms\":%d}\n", asr_ms);
    return;
  }
  static float gaps[kPhraseCount];
  uint32_t mt0 = millis();
  int best = asr_match(kPhraseKana, kPhraseCount, gaps);
  match_ms = (int)(millis() - mt0);
  float second = 1e9f;
  for (int i = 0; i < kPhraseCount; i++)
    if (i != best && gaps[i] < second) second = gaps[i];
  const char* phrase =
      (best >= 0 && gaps[best] <= MATCH_GAP && second - gaps[best] >= MATCH_MARGIN) ? kPhraseText[best] : nullptr;
  if (TRIAL_SHARED_ARENA) asr_release_scratch();
  respond(phrase ? phrase : kana, kana, phrase, asr_ms);
}

static void back_to_listening() {
  reset_preview();
  st = St::Idle;
  above = below = speech_chunks = 0;
  pre_fill = pre_pos = 0;
  draw_status(serial_hold ? "試験中（シリアル入力待ち）" : "きいてるよ（話しかけてね）",
              serial_hold ? TFT_ORANGE : TFT_GREEN);
  mic_start();
}

static void start_speech() {
  reset_preview();
  frontend_us = match_ms = asr_wait_ms = 0;
  used_preview = false;
  preview_cancellations = 0;
  dropped_chunks.store(0);
  mic_gap_max_us.store(0);
  endpoint_ms = 0;
  last_voice_ms = millis();
  st = St::Speech;
  utt_len = 0;
  for (int i = 0; i < pre_fill; i++) {
    int k = (pre_pos - pre_fill + i + PREROLL_CHUNKS) % PREROLL_CHUNKS;
    memcpy(utt + utt_len, preroll + k * CHUNK, sizeof(int16_t) * CHUNK);
    utt_len += CHUNK;
  }
  if (trial_flags & TR_FRONTEND) {
    asr_frontend_begin(utt);
    frontend_active = true;
    uint32_t start = micros();
    // Catch up the 500 ms preroll in small steps, without blocking a complete
    // two-buffer microphone queue (40 ms) with a single FFT burst.
    frontend_covered_samples = utt_len < CHUNK * 6 ? utt_len : CHUNK * 6;
    asr_frontend_push(frontend_covered_samples);
    frontend_us += (int)(micros() - start);
  }
  below = 0;
  draw_status("きいてる…", TFT_CYAN);
}

static void handle_chunk(const int16_t* x) {
  if ((int32_t)(millis() - deaf_until) < 0) return;  // 喋り終わった直後（残響・自分の声）は聞かない
  float lv = chunk_level(x);
  if (lv > mic_peak_level) mic_peak_level = lv;
  static uint32_t last_draw = 0;
  if (millis() - last_draw > 50) {
    draw_meter(lv);
    last_draw = millis();
  }

  if (st == St::Calib) {
    noise += lv;
    if (++calib_n >= CALIB_CHUNKS) {
      noise /= calib_n;
      st = St::Idle;
      draw_status("きいてるよ（話しかけてね）", TFT_GREEN);
    }
    return;
  }

  if (st == St::Idle) {
    memcpy(preroll + pre_pos * CHUNK, x, sizeof(int16_t) * CHUNK);
    pre_pos = (pre_pos + 1) % PREROLL_CHUNKS;
    if (pre_fill < PREROLL_CHUNKS) pre_fill++;
    if (lv > fmaxf(noise * START_RATIO, MIN_LEVEL)) {
      if (++above >= START_CHUNKS) {
        start_speech();
        speech_chunks = above;
      }
    } else {
      above = 0;
      noise += 0.02f * (lv - noise);
    }
    return;
  }

  // St::Speech
  if (utt_len + CHUNK <= MAX_SAMPLES) {
    memcpy(utt + utt_len, x, sizeof(int16_t) * CHUNK);
    utt_len += CHUNK;
  }
  if (lv < fmaxf(noise * END_RATIO, MIN_LEVEL * 0.6f)) {
    below++;
  } else {
    below = 0;
    speech_chunks++;
    last_voice_ms = millis();
    if (preview_valid) {
      preview_valid = false;
      preview_cancellations++;
    }
  }
  if (!preview_valid && preview_state.load(std::memory_order_acquire) == 2) {
    preview_state.store(0);
    asr_frontend_reset();
    frontend_active = false;
    // Cancelled previews consumed the precomputed frontend. Rebuild at final ASR
    // instead of an expensive catch-up burst during ongoing recording.
  }
  if (frontend_active && preview_state.load() == 0) {
    uint32_t start = micros();
    frontend_covered_samples += CHUNK * 3;
    if (frontend_covered_samples > utt_len) frontend_covered_samples = utt_len;
    asr_frontend_push(frontend_covered_samples);
    frontend_us += (int)(micros() - start);
  }
  if ((trial_flags & TR_EARLY_ASR) && preview_start && below == TAIL_CHUNKS &&
      speech_chunks >= MIN_SPEECH_CHUNKS && preview_state.load() == 0) {
    preview_n = utt_len;
    preview_valid = true;
    preview_started_ms = millis();
    preview_state.store(1, std::memory_order_release);
    xSemaphoreGive(preview_start);
  }
  if (below >= END_CHUNKS || utt_len + CHUNK > MAX_SAMPLES) {
    endpoint_ms = millis();
    mic_stop();  // 認識と返事の間は録音を止める
    utt_len -= (below > TAIL_CHUNKS ? below - TAIL_CHUNKS : 0) * CHUNK;
    finish_utterance();
    back_to_listening();
  }
}

#include "trial_io.h"

// シリアルから "@文\n" を受けたら、その文で返事する（マイクを通さない試験用）
static void poll_serial() {
  static char line[320];
  static int len = 0;
  while (Serial.available()) {
    int c = Serial.read();
    if (c == '\r') continue;
    if (c != '\n') {
      if (len < (int)sizeof(line) - 1) line[len++] = (char)c;
      continue;
    }
    line[len] = 0;
    len = 0;
    if (line[0] == '?') trial_command(line + 1);
    if (line[0] == '@' && line[1] && st != St::Speech) {
      mic_stop();
      reset_preview();
      endpoint_ms = last_voice_ms = 0;
      frontend_us = match_ms = asr_wait_ms = 0;
      used_preview = false; preview_cancellations = 0;
      respond(line + 1, nullptr, nullptr, -1);
      back_to_listening();
    }
    if (line[0] == '!' && line[1] && st != St::Speech) {
      Emotion emo;
      const char* kana = parse_emotion(line + 1, &emo);
      mic_stop();
      speak_kana(kana, emo);
      Serial.printf("{\"say\":\"%s\",\"tts_ms\":%d,\"audio_ms\":%d,\"tts_missing\":%d}\n", kana, last_tts_ms,
                    last_audio_ms, tts_ok ? tts.missing() : -1);
      back_to_listening();
    }
  }
}

void setup() {
  auto cfg = M5.config();
  M5.begin(cfg);
  Serial.begin(115200);
  M5.Display.setRotation(1);
  M5.Display.fillScreen(TFT_BLACK);
  draw_status("じゅんび中… v" ZUNDA_VERSION, TFT_WHITE);

  utt = (int16_t*)asr_alloc(sizeof(int16_t) * MAX_SAMPLES, 0);
  preroll = (int16_t*)asr_alloc(sizeof(int16_t) * CHUNK * PREROLL_CHUNKS, 1);
  for (int i = 0; i < NBUF; i++) chunk_buf[i] = (int16_t*)asr_alloc(sizeof(int16_t) * CHUNK, 1);
  bool input_ok = utt && preroll;
  for (int i = 0; i < NBUF; i++) input_ok = input_ok && chunk_buf[i];
  if (!input_ok) {
    draw_status("録音メモリ確保失敗", TFT_RED);
    Serial.println("{\"error\":\"microphone buffer allocation failed\"}");
    while (true) M5.delay(1000);
  }

  // 認識モデルは PSRAM へ写すと速い（kana_asr と同じ）。確保できなければフラッシュのまま
  const void* blob = asr_model_blob;
  if (void* p = heap_caps_malloc(sizeof(asr_model_blob), MALLOC_CAP_SPIRAM)) {
    memcpy(p, asr_model_blob, sizeof(asr_model_blob));
    blob = p;
  }
  int asr_rc;
  if (TRIAL_SHARED_ARENA) {
    shared_arena_bytes = asr_scratch_bytes(blob, sizeof(asr_model_blob), MAX_SAMPLES);
    if (shared_arena_bytes < UnitTTS::sharedWorkBytes()) shared_arena_bytes = UnitTTS::sharedWorkBytes();
    shared_arena = heap_caps_malloc(shared_arena_bytes, MALLOC_CAP_SPIRAM);
    asr_rc = asr_init_with_scratch(blob, sizeof(asr_model_blob), MAX_SAMPLES, asr_alloc,
                                  shared_arena, shared_arena_bytes);
  } else asr_rc = asr_init(blob, sizeof(asr_model_blob), MAX_SAMPLES, asr_alloc);
  if (asr_rc < 0) {
    draw_status("認識モデル読み込み失敗", TFT_RED);
    Serial.printf("{\"error\":\"%s\"}\n", asr_error());
    while (true) M5.delay(1000);
  }
  asr_set_clock([]() -> uint32_t { return micros(); });
  start_worker();

  const void* llm_blob = model_bin;
  if (TRIAL_LLM_PSRAM) {
    void* copy = heap_caps_malloc(model_bin_len, MALLOC_CAP_SPIRAM);
    if (!copy) {
      Serial.println("{\"error\":\"LLM PSRAM copy allocation failed\"}");
      while (true) M5.delay(1000);
    }
    memcpy(copy, model_bin, model_bin_len);
    llm_blob = copy;
  }
  int rc = llm_init(&llm, llm_blob, model_bin_len, tok_bin, tok_bin_len, LLM_STEPS, llm_alloc);
  if (rc == 0) rc = zunda_brain_init(&llm) ? -10 : 0;
  if (rc != 0) {
    draw_status("LLM 読み込み失敗", TFT_RED);
    Serial.printf("{\"error\":\"llm_init %d\"}\n", rc);
    while (true) M5.delay(1000);
  }
  llm.temperature = 0;  // greedy（テストベクタと同じ返事になる）

  const esp_partition_t* bank =
      esp_partition_find_first(ESP_PARTITION_TYPE_DATA, (esp_partition_subtype_t)0x40, "zbank");
  TtsInitOptions tts_init;
  tts_init.allocate_internal_pcm = !TRIAL_NO_INTERNAL_PCM;
  tts_init.shared_work = shared_arena;
  tts_init.shared_work_bytes = shared_arena_bytes;
  if (TRIAL_SHARED_ARENA) asr_release_scratch();
  tts_ok = bank && tts.begin(bank, tts_init);
  if (tts_ok) {
    for (int i = 0; i < TALK_BUFS; i++) {
      talk_buf[i] = (int16_t*)asr_alloc(TALK_MAX * sizeof(int16_t), 0);
      if (!talk_buf[i]) tts_ok = false;
    }
    if (!tts_ok) Serial.println("{\"error\":\"playback buffer allocation failed\"}");
  }
  if (!tts_ok) Serial.println("{\"warn\":\"音声バンクがありません（tools/flash_voice.sh）。画面表示だけで動きます\"}");
  start_trial_workers();
  if (!apply_trial_flags(TRIAL_DEFAULT_MASK)) Serial.println("{\"error\":\"default profile unavailable\"}");
  llm.rng = ((unsigned long long)esp_random() << 32) | esp_random();

  Serial.printf("{\"ready\":true,\"version\":\"" ZUNDA_VERSION "\",\"fast_prefill\":%s,\"tts\":%s,\"llm\":\"%s\",\"free_internal\":%u,\"free_psram\":%u}\n", ZUNDA_FAST_PREFILL ? "true" : "false", tts_ok ? "true" : "false", MODEL_NAME,
                (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
                (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
  trial_status();

  auto mc = M5.Mic.config();
  mc.sample_rate = SR;
  M5.Mic.config(mc);
  // 起動画面にクレジット（最初の返事で消える）
  M5.Display.setFont(&fonts::efontJA_16);
  M5.Display.setTextColor(TFT_LIGHTGREY);
  M5.Display.setCursor(4, 200);
  M5.Display.print("M5stack-zundamon v" ZUNDA_VERSION);
  M5.Display.setCursor(4, 220);
  M5.Display.print("VOICEVOX:ずんだもん");
  draw_status("まわりの音をはかってるよ…", TFT_YELLOW);
  mic_start();
}

void loop() {
  M5.update();
  if (M5.Mic.isEnabled()) {
    while (M5.Mic.isRecording() < 2) {
      if (!M5.Mic.record(chunk_buf[rec_idx], CHUNK, SR)) break;
      rec_idx = (rec_idx + 1) % NBUF;
    }
    while (int16_t* x = pop_chunk()) handle_chunk(x);
  }
  if (M5.BtnA.wasClicked()) {
    mic_stop();
    reset_preview();
    serial_hold = false;
    st = St::Calib;
    noise = 0;
    calib_n = 0;
    above = below = pre_fill = pre_pos = 0;
    draw_status("まわりの音をはかってるよ…", TFT_YELLOW);
    mic_start();
  }
  if (M5.BtnB.wasClicked() && last_reply[0] && st != St::Speech) {
    mic_stop();
    reset_preview();
    if (TRIAL_SHARED_ARENA) asr_release_scratch();
    talk_text(last_reply, last_emo);
    back_to_listening();
  }
  if (M5.BtnC.wasClicked() && last_heard[0] && st != St::Speech) {
    mic_stop();
    reset_preview();
    if (TRIAL_SHARED_ARENA) asr_release_scratch();
    endpoint_ms = last_voice_ms = 0;
    llm.temperature = 0.8f;  // 同じ話しかけに別の返事
    respond(last_heard, nullptr, nullptr, -1);
    llm.temperature = 0;
    back_to_listening();
  }
  poll_serial();
  M5.delay(1);
}
