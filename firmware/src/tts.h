#pragma once
// VCV単位接続TTS（VOICEVOX事前生成音声 bank.bin を SD から読む）
// フラッシュのデータ領域 zbank（8bit μ-law 版、版番号2）から読む口もある
#include <esp_partition.h>
#include <Arduino.h>
#include <FS.h>
#include "kana.h"

struct BankEntry { uint32_t off; uint16_t len, sa, ea, rsv; };

struct TtsInitOptions {
  // renderTo() users can omit the six-second (192,000-byte) internal PCM buffer.
  bool allocate_internal_pcm = true;
  // Optional float-aligned scratch owned by the caller. ASR may reuse this memory
  // between renders, but must never access it while TTS render is running.
  void* shared_work = nullptr;
  size_t shared_work_bytes = 0;
};

struct TtsRenderOptions {
  uint16_t leading_silence_ms = 100;
  uint16_t trailing_silence_ms = 100;
  bool compute_envelope = true;
};

class UnitTTS {
 public:
  static constexpr size_t sharedWorkBytes() {
    return MAX_OUT * sizeof(float) + MAX_UNIT * (sizeof(float) + sizeof(int16_t));
  }
  bool begin(fs::FS& fs, const char* path, const TtsInitOptions& options = TtsInitOptions());
  bool begin(const uint8_t* mem, size_t len, const TtsInitOptions& options = TtsInitOptions());
  bool begin(const esp_partition_t* part, const TtsInitOptions& options = TtsInitOptions());
  int missing() const { return missing_; }       // 直近 render で未収録だった単位数
  // モーラ列を PCM に描画。戻り値はサンプル数（buf は内部所有、次の呼び出しまで有効）
  size_t render(const MoraTok* seq, size_t n, const TtsRenderOptions& options = TtsRenderOptions());
  // dst remains caller-owned; do not reuse it while the speaker is playing it.
  // One UnitTTS instance must only be rendered by one task at a time.
  size_t renderTo(const MoraTok* seq, size_t n, int16_t* dst, size_t capacity,
                  const TtsRenderOptions& options = TtsRenderOptions());
  const int16_t* pcm() const { return rendered_pcm_; }
  uint16_t sampleRate() const { return sr_; }
  // 口パク用：20ms ごとの音量（0-255）
  const uint8_t* envelope() const { return env_; }
  size_t envelopeLen() const { return envLen_; }

 private:
  bool readUnit(const BankEntry& e, float* dst);
  bool readAt(size_t off, void* dst, size_t n);
  bool init(const TtsInitOptions& options);
  void ensure(size_t n);
  void crossfadeTail(size_t pos);
  void addAt(const float* x, size_t len, long start);

  fs::File f_;
  const uint8_t* mem_ = nullptr;
  size_t memLen_ = 0;
  const esp_partition_t* part_ = nullptr;
  int bytesPer_ = 2;           // 1 = μ-law（版番号2）
  int missing_ = 0;
  uint32_t dataOff_ = 0;
  uint16_t sr_ = 16000, M_ = 192, nMora_ = 0, nCtx_ = 7, nEnd_ = 12;
  BankEntry* idx_ = nullptr;
  BankEntry end_[12];
  float* out_ = nullptr;     // 作業バッファ（PSRAM）
  int16_t* out16_ = nullptr;
  const int16_t* rendered_pcm_ = nullptr;
  float* unit_ = nullptr;    // 単位1個分
  int16_t* raw_ = nullptr;
  float* fin_ = nullptr;     // フェードイン窓 2M
  size_t cap_ = 0, len_ = 0;
  uint8_t* env_ = nullptr;
  size_t envLen_ = 0;
  static const size_t MAX_UNIT = 16000;      // 1単位最大1秒
  static const size_t MAX_OUT = 16000 * 6;   // 1回の合成は最大6秒（句ごとに合成するので足りる。PSRAM 節約）
};
