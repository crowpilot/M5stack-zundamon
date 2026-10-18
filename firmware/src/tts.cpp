#include "tts.h"
#include "moras.h"
#include <esp_heap_caps.h>

// アルゴリズムは tools/concat.py と同一（ホストで聴いた音＝実機の音）
static void* pmalloc(size_t n) {
  void* p = heap_caps_malloc(n, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  return p ? p : malloc(n);
}

bool UnitTTS::readAt(size_t off, void* dst, size_t n) {
  if (part_) return esp_partition_read(part_, off, dst, n) == ESP_OK;
  if (mem_) {
    if (off + n > memLen_) return false;
    memcpy(dst, mem_ + off, n);
    return true;
  }
  f_.seek(off);
  return f_.read((uint8_t*)dst, n) == n;
}

bool UnitTTS::begin(fs::FS& fs, const char* path, const TtsInitOptions& options) {
  f_ = fs.open(path, "r");
  if (!f_) { Serial.printf("[tts] cannot open %s\n", path); return false; }
  return init(options);
}

bool UnitTTS::begin(const uint8_t* mem, size_t len, const TtsInitOptions& options) {
  mem_ = mem; memLen_ = len;
  return init(options);
}

bool UnitTTS::begin(const esp_partition_t* part, const TtsInitOptions& options) {
  part_ = part;
  return part_ && init(options);
}

static int16_t mulaw_dec(uint8_t u) {  // G.711 μ-law
  u = ~u;
  int x = (((u & 0x0F) << 3) + 0x84) << ((u >> 4) & 7);
  x -= 0x84;
  return (u & 0x80) ? -x : x;
}

bool UnitTTS::init(const TtsInitOptions& options) {
  if (options.shared_work &&
      (options.shared_work_bytes < sharedWorkBytes() ||
       (reinterpret_cast<uintptr_t>(options.shared_work) % alignof(float)) != 0)) return false;
  char magic[4]; uint16_t h[6];
  readAt(0, magic, 4); readAt(4, h, sizeof(h));
  if (memcmp(magic, "ZBNK", 4) != 0 || (h[0] != 1 && h[0] != 2)) { Serial.println("[tts] bad bank"); return false; }
  bytesPer_ = h[0] == 2 ? 1 : 2;
  sr_ = h[1]; M_ = h[2]; nMora_ = h[3]; nCtx_ = h[4]; nEnd_ = h[5];
  if (nMora_ != N_MORA || nEnd_ != 12) { Serial.println("[tts] moras.h と bank.bin が不一致"); return false; }
  size_t nIdx = (size_t)nCtx_ * nMora_ * 4;
  size_t hdr = 4 + sizeof(h);
  idx_ = (BankEntry*)pmalloc(nIdx * sizeof(BankEntry));
  readAt(hdr, idx_, nIdx * sizeof(BankEntry));
  readAt(hdr + nIdx * sizeof(BankEntry), end_, sizeof(end_));
  dataOff_ = hdr + (nIdx + 12) * sizeof(BankEntry);

  if (options.shared_work) {
    uint8_t* work = static_cast<uint8_t*>(options.shared_work);
    out_ = reinterpret_cast<float*>(work);
    unit_ = reinterpret_cast<float*>(work + MAX_OUT * sizeof(float));
    raw_ = reinterpret_cast<int16_t*>(work + (MAX_OUT + MAX_UNIT) * sizeof(float));
  } else {
    out_ = (float*)pmalloc(MAX_OUT * sizeof(float));
    unit_ = (float*)pmalloc(MAX_UNIT * sizeof(float));
    raw_ = (int16_t*)pmalloc(MAX_UNIT * sizeof(int16_t));
  }
  out16_ = options.allocate_internal_pcm ? (int16_t*)pmalloc(MAX_OUT * sizeof(int16_t)) : nullptr;
  env_ = (uint8_t*)pmalloc(MAX_OUT / 320 + 1);
  fin_ = (float*)pmalloc(2 * M_ * sizeof(float));
  if (!idx_ || !out_ || !unit_ || !raw_ || !env_ || !fin_ ||
      (options.allocate_internal_pcm && !out16_)) return false;
  // np.hanning(4M)[:2M] と同じ
  for (int i = 0; i < 2 * M_; i++) fin_[i] = 0.5f - 0.5f * cosf(2.f * PI * i / (4 * M_ - 1));
  cap_ = MAX_OUT;
  Serial.printf("[tts] bank ok (%s): sr=%u M=%u moras=%u\n", part_ ? (bytesPer_ == 1 ? "partition mu-law" : "partition") : mem_ ? "flash" : "SD", sr_, M_, nMora_);
  return true;
}

bool UnitTTS::readUnit(const BankEntry& e, float* dst) {
  if (e.len == 0) { missing_++; return false; }   // 未収録（ミニ版バンク）
  if (e.len > MAX_UNIT) return false;
  if (!readAt(dataOff_ + (size_t)e.off * bytesPer_, raw_, e.len * bytesPer_)) return false;
  if (bytesPer_ == 1) {
    const uint8_t* u = (const uint8_t*)raw_;
    for (int i = 0; i < e.len; i++) dst[i] = mulaw_dec(u[i]);
  } else {
    for (int i = 0; i < e.len; i++) dst[i] = raw_[i];
  }
  return true;
}

void UnitTTS::ensure(size_t n) {   // 長さ n まで 0 埋めで伸ばす
  if (n > cap_) n = cap_;
  if (n > len_) { memset(out_ + len_, 0, (n - len_) * sizeof(float)); len_ = n; }
}

// out を pos+M で切り、[pos-M, pos+M) をフェードアウト
void UnitTTS::crossfadeTail(size_t pos) {
  size_t t = pos + M_;
  if (t < len_) len_ = t; else ensure(t);
  for (int i = 0; i < 2 * M_; i++) {
    long j = (long)pos - M_ + i;
    if (j >= 0 && (size_t)j < len_) out_[j] *= fin_[2 * M_ - 1 - i];
  }
}

void UnitTTS::addAt(const float* x, size_t n, long start) {
  if (start < 0) { x -= start; n += start; start = 0; }
  ensure(start + n);
  for (size_t i = 0; i < n && start + i < cap_; i++) out_[start + i] += x[i];
}

size_t UnitTTS::render(const MoraTok* seq, size_t n, const TtsRenderOptions& options) {
  return renderTo(seq, n, out16_, MAX_OUT, options);
}

size_t UnitTTS::renderTo(const MoraTok* seq, size_t n, int16_t* dst, size_t capacity,
                         const TtsRenderOptions& options) {
  rendered_pcm_ = nullptr;
  envLen_ = 0;
  if (!idx_ || !dst || !capacity || (!seq && n)) return 0;
  cap_ = capacity < MAX_OUT ? capacity : MAX_OUT;
  missing_ = 0;
  len_ = 0; ensure((size_t)sr_ * options.leading_silence_ms / 1000);
  size_t pos = len_;
  int ctx = 0;          // 0=無音, 1..6 = 母音 a i u e o N
  int pp = P_L;

  auto close = [&]() {
    const BankEntry& e = end_[(ctx - 1) * 2 + pp];
    if (!readUnit(e, unit_)) return;
    for (int i = 0; i < 2 * M_; i++) unit_[i] *= fin_[i];
    crossfadeTail(pos);
    addAt(unit_, e.len, (long)pos - e.sa);
    pos = len_;
  };

  for (size_t k = 0; k < n; k++) {
    const MoraTok& t = seq[k];
    if (t.kind == MK_PAUSE || t.kind == MK_SOKUON) {
      if (ctx != 0) close();
      ensure(len_ + (t.kind == MK_PAUSE ? sr_ * 18 / 100 : sr_ * 6 / 100));
      pos = len_; ctx = 0; continue;
    }
    int m = t.idx;
    if (t.kind == MK_LONG) m = VOWEL_MORA[ctx ? ctx - 1 : 0];
    if (m < 0 || m >= nMora_) continue;
    int cp = t.pitch;
    int ppx = (ctx == 0) ? cp : pp;
    const BankEntry& e = idx_[(((size_t)ctx * nMora_ + m) * 2 + ppx) * 2 + cp];
    if (!readUnit(e, unit_)) continue;
    long start;
    if (ctx == 0) {
      start = pos;
    } else {
      for (int i = 0; i < 2 * M_; i++) unit_[i] *= fin_[i];
      crossfadeTail(pos);
      start = (long)pos - e.sa;
    }
    addAt(unit_, e.len, start);
    pos = start + e.ea;
    ctx = MORA_VOWEL[m] + 1;
    pp = cp;
  }
  if (ctx != 0) close();
  ensure(len_ + (size_t)sr_ * options.trailing_silence_ms / 1000);

  // int16 化と口パク用エンベロープ
  for (size_t i = 0; options.compute_envelope && i < len_; i += 320) {
    float s = 0; size_t e = min(len_, i + 320);
    for (size_t j = i; j < e; j++) s += out_[j] * out_[j];
    float rms = sqrtf(s / (e - i));
    env_[envLen_++] = (uint8_t)min(255.f, rms / 40.f);
  }
  for (size_t i = 0; i < len_; i++) {
    float v = out_[i];
    dst[i] = (int16_t)(v > 32767 ? 32767 : v < -32767 ? -32767 : v);
  }
  rendered_pcm_ = dst;
  return len_;
}
