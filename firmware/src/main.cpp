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
static constexpr float MATCH_GAP = 2.5f;
static constexpr float MATCH_MARGIN = 1.0f;
static constexpr int LLM_STEPS = 256;             // KVキャッシュ長

static constexpr uint32_t DEAF_MS = 500;          // 喋り終わってからこの間はマイクの音を無視（自分の声に反応しない）

static Llm llm;
static UnitTTS tts;
static bool tts_ok = false;
static uint32_t deaf_until = 0;

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
  uint32_t h = q_head.load(std::memory_order_relaxed);
  if (h - q_tail.load(std::memory_order_acquire) < NBUF) {
    done_q[h % NBUF] = (int16_t*)data;
    q_head.store(h + 1, std::memory_order_release);
  }
}

static int16_t* pop_chunk() {
  uint32_t t = q_tail.load(std::memory_order_relaxed);
  if (t == q_head.load(std::memory_order_acquire)) return nullptr;
  int16_t* p = done_q[t % NBUF];
  q_tail.store(t + 1, std::memory_order_release);
  return p;
}

static void mic_start() {
  M5.Speaker.end();  // Core2はマイクとスピーカーが同じI2Sを使う
  q_head = 0;
  q_tail = 0;
  M5.Mic.setBufferReleaseCallback(nullptr, on_chunk);
  M5.Mic.begin();
}

static void mic_stop() {
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

static void draw_status(const char* s, uint16_t color) {
  M5.Display.fillRect(0, 0, 320, 30, TFT_BLACK);
  M5.Display.setFont(&fonts::efontJA_16);
  M5.Display.setTextColor(color);
  M5.Display.setCursor(4, 6);
  M5.Display.print(s);
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

static void on_piece(const char* piece, void*) {
  if (M5.Display.getCursorY() < 200) M5.Display.print(piece);  // はみ出す分は表示しない（シリアルには全文）
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

// 記法を合成して再生し終わるまで待つ。マイクは止めておくこと（Core2 は I2S を共有）
static void speak_kana(const char* kana, Emotion emo) {
  last_tts_ms = last_audio_ms = 0;
  if (!tts_ok || !kana[0]) return;
  uint32_t t0 = millis();
  size_t n = parse_kana(kana, seq, 256);
  size_t len = tts.render(seq, n);
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

// ---- 返事を作る ---------------------------------------------------------------
static char last_heard[320];
static char last_say[512];
static Emotion last_emo = EMO_NEUTRAL;

static void respond(const char* heard, const char* kana, const char* phrase, int asr_ms) {
  strlcpy(last_heard, heard, sizeof(last_heard));
  draw_status("ずんだもんが考え中…", TFT_ORANGE);
  draw_heard(heard, kana);
  M5.Display.setTextColor(TFT_GREENYELLOW);
  M5.Display.print("ずんだもん：");
  M5.Display.setTextColor(TFT_WHITE);
  char reply[256];
  int feel = 0;
  uint32_t t0 = millis();
  int tokens = zunda_brain_reply(&llm, heard, reply, sizeof(reply), &feel, on_piece, nullptr);
  int llm_ms = (int)(millis() - t0);
  draw_footer(zunda_feel_name(feel), asr_ms, llm_ms, tokens);
  int skipped = 0;
  yomi_to_kana(reply, last_say, sizeof(last_say), &skipped);
  last_emo = FEEL_EMO[feel];
  speak_kana(last_say, last_emo);

  Serial.print("{");
  json_str("heard", heard);
  if (kana) json_str("kana", kana);
  if (phrase) json_str("phrase", phrase);
  if (asr_ms >= 0) Serial.printf("\"asr_ms\":%d,", asr_ms);
  json_str("reply", reply);
  json_str("feel", zunda_feel_name(feel));
  json_str("say", last_say);
  Serial.printf("\"tts_ms\":%d,\"audio_ms\":%d,\"yomi_skip\":%d,\"tts_missing\":%d,", last_tts_ms, last_audio_ms,
                skipped, tts_ok ? tts.missing() : -1);
  Serial.printf("\"llm_ms\":%d,\"tokens\":%d,\"temp\":%.1f,\"free_internal\":%u,\"free_psram\":%u}\n", llm_ms,
                tokens, llm.temperature, (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
                (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
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

// ---- 認識 → 返事 --------------------------------------------------------------
static void finish_utterance() {
  if (speech_chunks < MIN_SPEECH_CHUNKS) return;
  draw_status("ききとり中…", TFT_ORANGE);
  char kana[256];
  float conf = 0;
  uint32_t t0 = millis();
  int n = asr_recognize(utt, utt_len, kana, sizeof(kana), &conf);
  int asr_ms = (int)(millis() - t0);
  if (n < 0) {
    Serial.printf("{\"error\":\"%s\"}\n", asr_error());
    return;
  }
  if (n == 0) {  // 何も聞き取れなければ返事しない
    Serial.printf("{\"kana\":\"\",\"asr_ms\":%d}\n", asr_ms);
    return;
  }
  static float gaps[kPhraseCount];
  int best = asr_match(kPhraseKana, kPhraseCount, gaps);
  float second = 1e9f;
  for (int i = 0; i < kPhraseCount; i++)
    if (i != best && gaps[i] < second) second = gaps[i];
  const char* phrase =
      (best >= 0 && gaps[best] <= MATCH_GAP && second - gaps[best] >= MATCH_MARGIN) ? kPhraseText[best] : nullptr;
  respond(phrase ? phrase : kana, kana, phrase, asr_ms);
}

static void back_to_listening() {
  st = St::Idle;
  above = below = speech_chunks = 0;
  pre_fill = pre_pos = 0;
  draw_status("きいてるよ（話しかけてね）", TFT_GREEN);
  mic_start();
}

static void start_speech() {
  st = St::Speech;
  utt_len = 0;
  for (int i = 0; i < pre_fill; i++) {
    int k = (pre_pos - pre_fill + i + PREROLL_CHUNKS) % PREROLL_CHUNKS;
    memcpy(utt + utt_len, preroll + k * CHUNK, sizeof(int16_t) * CHUNK);
    utt_len += CHUNK;
  }
  below = 0;
  draw_status("きいてる…", TFT_CYAN);
}

static void handle_chunk(const int16_t* x) {
  if ((int32_t)(millis() - deaf_until) < 0) return;  // 喋り終わった直後（残響・自分の声）は聞かない
  float lv = chunk_level(x);
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
  }
  if (below >= END_CHUNKS || utt_len + CHUNK > MAX_SAMPLES) {
    mic_stop();  // 認識と返事の間は録音を止める
    utt_len -= (below > TAIL_CHUNKS ? below - TAIL_CHUNKS : 0) * CHUNK;
    finish_utterance();
    back_to_listening();
  }
}

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
    if (line[0] == '@' && line[1] && st != St::Speech) {
      mic_stop();
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

  // 認識モデルは PSRAM へ写すと速い（kana_asr と同じ）。確保できなければフラッシュのまま
  const void* blob = asr_model_blob;
  if (void* p = heap_caps_malloc(sizeof(asr_model_blob), MALLOC_CAP_SPIRAM)) {
    memcpy(p, asr_model_blob, sizeof(asr_model_blob));
    blob = p;
  }
  if (asr_init(blob, sizeof(asr_model_blob), MAX_SAMPLES, asr_alloc) < 0) {
    draw_status("認識モデル読み込み失敗", TFT_RED);
    Serial.printf("{\"error\":\"%s\"}\n", asr_error());
    while (true) M5.delay(1000);
  }
  start_worker();

  int rc = llm_init(&llm, model_bin, model_bin_len, tok_bin, tok_bin_len, LLM_STEPS, llm_alloc);
  if (rc == 0) rc = zunda_brain_init(&llm) ? -10 : 0;
  if (rc != 0) {
    draw_status("LLM 読み込み失敗", TFT_RED);
    Serial.printf("{\"error\":\"llm_init %d\"}\n", rc);
    while (true) M5.delay(1000);
  }
  llm.temperature = 0;  // greedy（テストベクタと同じ返事になる）

  const esp_partition_t* bank =
      esp_partition_find_first(ESP_PARTITION_TYPE_DATA, (esp_partition_subtype_t)0x40, "zbank");
  tts_ok = bank && tts.begin(bank);
  if (!tts_ok) Serial.println("{\"warn\":\"音声バンクがありません（tools/flash_voice.sh）。画面表示だけで動きます\"}");
  llm.rng = ((unsigned long long)esp_random() << 32) | esp_random();

  Serial.printf("{\"ready\":true,\"version\":\"" ZUNDA_VERSION "\",\"tts\":%s,\"llm\":\"%s\",\"free_internal\":%u,\"free_psram\":%u}\n", tts_ok ? "true" : "false", MODEL_NAME,
                (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
                (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));

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
    st = St::Calib;
    noise = 0;
    calib_n = 0;
    draw_status("まわりの音をはかってるよ…", TFT_YELLOW);
  }
  if (M5.BtnB.wasClicked() && last_say[0] && st != St::Speech) {
    mic_stop();
    speak_kana(last_say, last_emo);
    back_to_listening();
  }
  if (M5.BtnC.wasClicked() && last_heard[0] && st != St::Speech) {
    mic_stop();
    llm.temperature = 0.8f;  // 同じ話しかけに別の返事
    respond(last_heard, nullptr, nullptr, -1);
    llm.temperature = 0;
    back_to_listening();
  }
  poll_serial();
  M5.delay(1);
}
