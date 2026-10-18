# TTS integration experiments

Run the real-bank host check from the repository root:

```sh
python3 tools/tts/check_tts.py /path/to/voice/bank_mu.bin
```

The script compiles the pre-integration implementation from commit
`0a10df94440fa997456620bded5d107eb3bad0d2`, then the current implementation, with
AddressSanitizer and UndefinedBehaviorSanitizer. It uses small host stubs only for
Arduino logging, the allocator, and unused SD/partition APIs. The production
`UnitTTS`, kana parser, yomi dictionary and converter run unchanged against the
actual bank loaded into memory. This does not measure ESP32 speed or speaker output.

## Independent options

Existing `begin()` and `render()` calls preserve their defaults: 100 ms of leading
and trailing silence, an internal PCM buffer, and envelope computation.

```cpp
TtsInitOptions init;
init.allocate_internal_pcm = false;
// Optional: a float-aligned caller-owned arena, at least 480,000 bytes.
init.shared_work = arena;
init.shared_work_bytes = arena_bytes;
tts.begin(bank_partition, init);

TtsRenderOptions render;
render.leading_silence_ms = 20;
render.trailing_silence_ms = 100;
render.compute_envelope = false;
size_t samples = tts.renderTo(moras, mora_count, free_pcm_buffer, pcm_capacity, render);
```

- `allocate_internal_pcm=false` omits exactly 192,000 bytes. In that mode the
  legacy `render()` safely returns zero; callers must use `renderTo()`.
- `UnitTTS::sharedWorkBytes()` returns 480,000: float output 384,000, float unit
  64,000, and raw unit 32,000 bytes. An ASR arena can own these bytes during ASR,
  then TTS can reuse them after ASR and phrase matching finish. The caller must
  exclude concurrent access. A subsequent ASR may overwrite the entire arena:
  TTS does not retain useful scratch contents between renders. Indices, fade
  windows, and the optional envelope stay in permanent independent storage.
- `renderTo()` leaves the destination owned by the caller. The caller must not
  reuse a queued/playing PCM buffer. One `UnitTTS` object is not reentrant.
- `pcm()` points at the latest successful output, including a caller buffer;
  an invalid render clears it. `envelopeLen()` is zero when envelope computation
  is disabled. Envelope bytes must only be read up to that length.
- The caller controls initial and subsequent phrase padding separately through
  render options. Reducing a first phrase from 100 to 20 ms removes 1,280 leading
  zero samples at 16 kHz. Removing the trailing 100 ms removes 1,600 trailing
  zero samples. Default playback rate converts samples directly to those times;
  non-default voice rates scale both durations.

The host check verifies baseline PCM equality, direct-buffer equality, envelope
on/off equality, each padding change separately, output capacity guards, arena
guards, complete scratch overwrites between renders, and invalid arena rejection.

## Early speech: reading is not completed-phrase audio

`yomi_stable_prefix(text, minimum_moras)` is an experimental **reading-only**
boundary. It returns a UTF-8 byte position after enough confirmed morae, retaining
an unfinished longest dictionary match and at least one following mora. It does
not split before small kana or a long-vowel mark. Complete punctuation stays with
the existing phrase path. No baseline call site uses this function automatically.

The checker confirms pronunciation-prefix stability on partial UTF-8 inputs and
known completions. It also demonstrates why handing an arbitrary early prefix to
the existing completed-phrase renderer cannot claim unchanged sound:

| Prefix → eventual text | Existing behavior that can change |
| --- | --- |
| `一` → `一緒` | Dictionary reading `ヒト` → `イッショ` |
| `き` → `きゃ` | `キ` becomes the single mora `キャ` |
| `あ` → `あのだ` | The later `ノダ` ending changes the first mora's pitch |
| `いい` → `いい？` | A later question mark raises the final mora |
| `こんにちは` → `こんにちはなのだ` | Closing the early phrase lowers its final mora, while the continuous phrase keeps it high |

Even a reading-stable cut adds a VCV closing unit, final pitch fall, new phrase
context, and any configured padding. Therefore early-cut experiments must remain
separately flagged and assessed as a prosody change, not a numerical optimization.
The ordinary phrase path should remain available as the comparison.

A future exact streaming renderer would preserve VCV context and withhold every
sample that the next unit can modify. In the checked bank, fade half-width `M` and
maximum `sa` are both 192 samples (12 ms at 16 kHz). That is an acoustic tail bound,
not proof that only 12 ms of *text* look-ahead is sufficient: dictionary extension,
small kana, question intonation and the `ノダ` accent rule still require unresolved
reading/prosody to be held. A byte-boundary helper alone cannot solve those issues.

No model retraining is performed by these tools. Training a model to emit explicit
pronunciation/accent/phrase units would be a separate project and evaluation.
