#include "tts.h"
#include "yomi.h"
#include "yomi_dict.h"
#include <cassert>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

HostSerial Serial;
size_t host_allocated_bytes = 0;

static std::vector<uint8_t> read_bytes(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  assert(f.good());
  return std::vector<uint8_t>(std::istreambuf_iterator<char>(f), {});
}

static const char* const CASES[] = {
  "おはようなのだ！", "こんにちは！ボクもだいすきなのだ！", "テストで100点とったよ！",
  "今日はいい天気ですね？", "わかった、ありがとうなのだ。", "え？", "いいのだ！",
  "ずんだもんはきょうもげんきいっぱいでたくさんおはなししたいのだ。", ""
};

#ifndef TTS_REFERENCE
static size_t moras_for(const std::string& text, MoraTok* out, char* kana) {
  yomi_to_kana(text.c_str(), kana, 1024, nullptr);
  return parse_kana(kana, out, 512);
}

static void check_early_prefix() {
  size_t checked = 0;
  const char* examples[] = {"一", "一緒", "大", "大好きなのだ", "あ", "あのだ", "いい", "いい？",
                           "き", "きゃ", "きょうはとってもいいてんきなのだ", "ずんだもんはスーパーなのだ"};
  std::vector<std::string> corpus;
  for (const char* s : CASES) corpus.emplace_back(s);
  for (const char* s : examples) corpus.emplace_back(s);
  for (int i = 0; i < YOMI_N; i++) corpus.emplace_back(std::string("きょうは") + YOMI_SURF[i] + "なのだ");
  for (const auto& full : corpus) {
    MoraTok expected[512]; char kana[1024];
    size_t nf = moras_for(full, expected, kana);
    for (size_t cut = 0; cut <= full.size(); cut++) {
      std::string partial = full.substr(0, cut);
      size_t stable = yomi_stable_prefix(partial.c_str(), 4);
      if (!stable) continue;
      assert(stable < partial.size());
      assert((static_cast<unsigned char>(partial[stable]) & 0xC0) != 0x80);
      MoraTok got[512];
      size_t ng = moras_for(partial.substr(0, stable), got, kana);
      assert(ng >= 4 && ng <= nf);
      for (size_t i = 0; i < ng; i++) {
        assert(got[i].kind == expected[i].kind);
        assert(got[i].idx == expected[i].idx);
      }
      checked++;
    }
  }
  assert(checked > 10);
  // These are concrete reasons not to use a completed-chunk renderer as an
  // allegedly identical streaming renderer, even if pronunciation is stable.
  const char* pairs[][2] = {{"一", "一緒"}, {"き", "きゃ"}, {"あ", "あのだ"}, {"いい", "いい？"},
                           {"こんにちは", "こんにちはなのだ"}};
  for (const auto& pair : pairs) {
    MoraTok short_moras[512], full_moras[512]; char sk[1024], fk[1024];
    size_t ns = moras_for(pair[0], short_moras, sk), nf = moras_for(pair[1], full_moras, fk);
    bool differs = ns > nf;
    for (size_t i = 0; i < min(ns, nf); i++)
      if (short_moras[i].idx != full_moras[i].idx || short_moras[i].kind != full_moras[i].kind ||
          short_moras[i].pitch != full_moras[i].pitch) differs = true;
    assert(differs);
    std::cout << "early-cut counterexample: " << pair[0] << " -> " << pair[1]
              << " | " << sk << " -> " << fk << "\n";
  }
  std::cout << "reading-only stable-prefix checks=" << checked
            << "; completed-chunk early speech is NOT PCM-equivalent\n";
}
#endif

int main(int argc, char** argv) {
  assert(argc == 3);
  auto bank = read_bytes(argv[1]);
  UnitTTS base;
  size_t before = host_allocated_bytes;
  assert(base.begin(bank.data(), bank.size()));
  const size_t base_bytes = host_allocated_bytes - before;
#ifndef TTS_REFERENCE
  UnitTTS direct;
  TtsInitOptions init; init.allocate_internal_pcm = false;
  before = host_allocated_bytes;
  assert(direct.begin(bank.data(), bank.size(), init));
  const size_t direct_bytes = host_allocated_bytes - before;
  assert(base_bytes - direct_bytes == 192000);
  assert(direct.render(nullptr, 0) == 0);
  std::vector<int16_t> dst(96000);
  std::vector<uint32_t> arena(UnitTTS::sharedWorkBytes() / sizeof(uint32_t) + 2, 0x12345678);
  UnitTTS shared;
  init.shared_work = arena.data() + 1;
  init.shared_work_bytes = UnitTTS::sharedWorkBytes();
  before = host_allocated_bytes;
  assert(shared.begin(bank.data(), bank.size(), init));
  assert(direct_bytes - (host_allocated_bytes - before) == UnitTTS::sharedWorkBytes());
  UnitTTS invalid;
  init.shared_work_bytes--;
  assert(!invalid.begin(bank.data(), bank.size(), init));
  init.shared_work_bytes++;
  init.shared_work = reinterpret_cast<uint8_t*>(arena.data()) + 1;
  assert(!invalid.begin(bank.data(), bank.size(), init));
#endif
  for (size_t c = 0; c < sizeof(CASES) / sizeof(CASES[0]); c++) {
    char kana[1024]; int skipped = -1; MoraTok seq[512];
    yomi_to_kana(CASES[c], kana, sizeof(kana), &skipped);
    const size_t n = parse_kana(kana, seq, 512);
    const size_t len = base.render(seq, n);
    assert(len && base.missing() == 0);
    const std::string path = std::string(argv[2]) + "/" + std::to_string(c) + ".pcm";
#ifdef TTS_REFERENCE
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(base.pcm()), len * 2);
#else
    const auto ref_bytes = read_bytes(path);
    assert(ref_bytes.size() == len * 2);
    assert(!std::memcmp(ref_bytes.data(), base.pcm(), len * 2));
    const std::vector<int16_t> ref(base.pcm(), base.pcm() + len);
    TtsRenderOptions opt;
    assert(direct.renderTo(seq, n, dst.data(), dst.size(), opt) == len);
    assert(direct.pcm() == dst.data());
    assert(!std::memcmp(ref.data(), dst.data(), len * 2));
    assert(direct.envelopeLen() == base.envelopeLen());
    assert(!std::memcmp(base.envelope(), direct.envelope(), base.envelopeLen()));
    // Simulate ASR overwriting every byte of the shared scratch between turns.
    std::memset(arena.data() + 1, 0x55, UnitTTS::sharedWorkBytes());
    assert(shared.renderTo(seq, n, dst.data(), dst.size(), opt) == len);
    assert(!std::memcmp(ref.data(), dst.data(), len * 2));
    assert(arena.front() == 0x12345678 && arena.back() == 0x12345678);
    opt.compute_envelope = false;
    assert(direct.renderTo(seq, n, dst.data(), dst.size(), opt) == len);
    assert(!std::memcmp(ref.data(), dst.data(), len * 2));
    assert(direct.envelopeLen() == 0);
    opt.leading_silence_ms = 20;
    assert(direct.renderTo(seq, n, dst.data(), dst.size(), opt) == len - 1280);
    assert(!std::memcmp(ref.data() + 1280, dst.data(), (len - 1280) * 2));
    for (size_t i = 0; i < 1280; i++) assert(ref[i] == 0);
    opt.trailing_silence_ms = 0;
    assert(direct.renderTo(seq, n, dst.data(), dst.size(), opt) == len - 2880);
    assert(!std::memcmp(ref.data() + 1280, dst.data(), (len - 2880) * 2));
    for (size_t i = len - 1600; i < len; i++) assert(ref[i] == 0);
    // A small output buffer must stay bounded; clipping must retain the prefix.
    std::vector<int16_t> bounded(3002, 12345);
    opt = TtsRenderOptions();
    assert(direct.renderTo(seq, n, bounded.data() + 1, 3000, opt) == 3000);
    assert(bounded.front() == 12345 && bounded.back() == 12345);
    assert(!std::memcmp(ref.data(), bounded.data() + 1, 3000 * 2));
    assert(direct.renderTo(seq, n, nullptr, dst.size(), opt) == 0);
    assert(direct.pcm() == nullptr);
#endif
    std::cout << "case=" << c << " samples=" << len << " kana=" << kana << "\n";
  }
#ifndef TTS_REFERENCE
  check_early_prefix();
  std::cout << "PASS: baseline PCM, external PCM, envelope off, independent padding, bounded output; "
            << "internal PCM allocation saved=" << (base_bytes - direct_bytes)
            << " bytes; reusable scratch=" << UnitTTS::sharedWorkBytes() << " bytes\n";
#endif
}
