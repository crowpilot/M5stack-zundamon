#pragma once
#include <stdint.h>

// Serial-only experiment controls. Normal buttons keep their existing roles.
enum TrialFlag : uint32_t {
  TR_LAST_CACHE = 1u << 0,
  TR_PHRASE_CACHE = 1u << 1,
  TR_ASR_NO_CONF = 1u << 2,
  TR_ASR_RAW = 1u << 3,
  TR_FRONTEND = 1u << 4,
  TR_EARLY_ASR = 1u << 5,
  TR_TTS_PAD = 1u << 6,
  TR_TTS_NO_ENV = 1u << 7,
  TR_TTS_DIRECT = 1u << 8,
  TR_TTS_PARALLEL = 1u << 9,
  TR_EARLY_READ = 1u << 10,
};
struct TrialProfile { const char* name; uint32_t flags; };
static constexpr uint32_t TR_INTEGRATED = TR_LAST_CACHE | TR_PHRASE_CACHE |
    TR_ASR_NO_CONF | TR_ASR_RAW | TR_FRONTEND | TR_TTS_PAD | TR_TTS_NO_ENV | TR_TTS_DIRECT;
static const TrialProfile trial_profiles[] = {
  {"baseline", 0}, {"last-cache", TR_LAST_CACHE}, {"phrase-cache", TR_PHRASE_CACHE},
  {"asr-no-conf", TR_ASR_NO_CONF}, {"asr-raw", TR_ASR_RAW},
  {"frontend", TR_FRONTEND}, {"early-asr", TR_EARLY_ASR},
  {"tts-pad", TR_TTS_PAD}, {"tts-no-env", TR_TTS_NO_ENV},
  {"tts-direct", TR_TTS_DIRECT}, {"tts-parallel", TR_TTS_PARALLEL},
  {"early-read", TR_EARLY_READ},
  {"integrated", TR_INTEGRATED},
  {"integrated-parallel", TR_INTEGRATED | TR_TTS_PARALLEL},
  {"integrated-early", TR_INTEGRATED | TR_EARLY_ASR},
  {"integrated-all", TR_INTEGRATED | TR_EARLY_ASR | TR_TTS_PARALLEL},
};
#ifndef TRIAL_DEFAULT_MASK
#define TRIAL_DEFAULT_MASK 0
#endif
#ifndef TRIAL_SHARED_ARENA
#define TRIAL_SHARED_ARENA 0
#endif
#ifndef TRIAL_NO_INTERNAL_PCM
#define TRIAL_NO_INTERNAL_PCM 0
#endif
#ifndef TRIAL_LLM_PSRAM
#define TRIAL_LLM_PSRAM 0
#endif
