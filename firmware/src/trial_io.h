#pragma once
// Serial diagnostics live in the main translation unit to use the real pipeline.
// No test-only approximation of the recognizer or speech endpoint is used.
static bool apply_trial_flags(uint32_t flags) {
  reset_preview();
  wait_tts_prepare();
  if ((flags & TR_PHRASE_CACHE) && !prompt_catalog_count) return false;
  if ((flags & TR_EARLY_ASR) && !preview_start) return false;
  if ((flags & TR_TTS_PARALLEL) && !tts_job_start) return false;
  // Keep the fixed two-token prefix warm while clearing the last-input cache.
  llm_set_prompt_cache(&llm, 0);
  if (llm_set_prompt_cache(&llm, !!(flags & TR_LAST_CACHE)) < 0 ||
      llm_set_prompt_snapshots(&llm, prompt_catalog_entries, prompt_catalog_count,
                              prompt_identity, !!(flags & TR_PHRASE_CACHE)) < 0) {
    llm_set_prompt_cache(&llm, 0);
    llm_set_prompt_snapshots(&llm, nullptr, 0, prompt_identity, 0);
    trial_flags = 0; trial_name = "baseline"; asr_set_options(0);
    return false;
  }
  trial_flags = flags;
  asr_set_options(asr_flags());
  return true;
}

static void trial_status() {
  Serial.printf("{\"control\":\"status\",\"ready\":true,\"version\":\"" ZUNDA_VERSION "\",");
  json_str("trial", trial_name);
  Serial.printf("\"flags\":%u,\"catalog_count\":%u,\"shared_arena\":%u,"
                "\"no_internal_pcm\":%d,\"llm_psram\":%d,\"tts\":%s,"
                "\"free_internal\":%u,\"free_psram\":%u,\"mic_callbacks\":%u,"
                "\"mic_peak_level\":%.2f,\"noise\":%.2f,\"vad_state\":%d,\"mic_gap_max_us\":%u}\n",
                trial_flags, (unsigned)prompt_catalog_count, (unsigned)shared_arena_bytes,
                TRIAL_NO_INTERNAL_PCM, TRIAL_LLM_PSRAM, tts_ok ? "true" : "false",
                (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
                (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM), mic_callbacks.load(),
                mic_peak_level, noise, (int)st, mic_gap_max_us.load());
}

static int cache_write(const void* bytes, size_t n, void*) {
  return Serial.write((const uint8_t*)bytes, n) == n ? 0 : -1;
}

static void trial_export() {
  uint32_t previous = trial_flags;
  apply_trial_flags(0);
  int count = 0;
  for (int i = 0; i < kPhraseCount; ++i) {
    size_t n = zunda_brain_prompt_snapshot_size(&llm, kPhraseText[i], 2);
    Serial.printf("{\"cache_index\":%d,\"bytes\":%u}\n", i, (unsigned)n);
    int rc = zunda_brain_export_prompt(&llm, kPhraseText[i], 2, prompt_identity, cache_write, nullptr);
    Serial.println();
    if (rc < 0) { Serial.printf("{\"error\":\"cache export %d\"}\n", rc); break; }
    ++count;
  }
  apply_trial_flags(previous);
  Serial.printf("{\"control\":\"export_done\",\"count\":%d}\n", count);
}

static void trial_load(int n) {
  if (n <= 0 || n > MAX_SAMPLES) { Serial.println("{\"error\":\"invalid fixture length\"}"); return; }
  if (!fixture_pcm) fixture_pcm = (int16_t*)heap_caps_malloc(MAX_SAMPLES * sizeof(int16_t), MALLOC_CAP_SPIRAM);
  if (!fixture_pcm) { Serial.println("{\"error\":\"fixture allocation failed\"}"); return; }
  fixture_samples = 0;
  size_t bytes = n * sizeof(int16_t), received = 0;
  Serial.printf("{\"control\":\"load_ready\",\"bytes\":%u}\n", (unsigned)bytes);
  uint32_t start = millis();
  while (received < bytes && millis() - start < 90000) {
    int available = Serial.available();
    if (!available) { M5.delay(1); continue; }
    size_t take = (size_t)available;
    if (take > bytes - received) take = bytes - received;
    received += Serial.readBytes((uint8_t*)fixture_pcm + received, take);
  }
  if (received != bytes) { Serial.println("{\"error\":\"fixture transfer timeout\"}"); return; }
  fixture_samples = n;
  Serial.printf("{\"control\":\"loaded\",\"samples\":%d}\n", n);
}

static void trial_asr() {
  if (!fixture_samples) { Serial.println("{\"error\":\"no fixture\"}"); return; }
  reset_preview();
  uint32_t front = 0;
  if (trial_flags & TR_FRONTEND) {
    asr_frontend_begin(fixture_pcm);
    uint32_t start = micros();
    for (int n = CHUNK; n < fixture_samples; n += CHUNK) asr_frontend_push(n);
    asr_frontend_push(fixture_samples);
    front = micros() - start;
  }
  char kana[256] = {0}; float confidence = 0;
  uint32_t start = micros();
  int result = asr_recognize(fixture_pcm, fixture_samples, kana, sizeof(kana),
                           (trial_flags & TR_ASR_NO_CONF) ? nullptr : &confidence);
  uint32_t recognized = micros();
  float gaps[kPhraseCount];
  int best = result >= 0 ? asr_match(kPhraseKana, kPhraseCount, gaps) : -1;
  uint32_t end = micros();
  float second = 1e9f, gap = best >= 0 ? gaps[best] : 1e9f;
  for (int i = 0; i < kPhraseCount; ++i) if (best >= 0 && i != best && gaps[i] < second) second = gaps[i];
  bool accepted = result > 0 && best >= 0 && gap <= MATCH_GAP && second - gap >= MATCH_MARGIN;
  Serial.print("{\"control\":\"asr\",");
  json_str("trial", trial_name); json_str("kana", kana);
  if (accepted) json_str("phrase", kPhraseText[best]);
  Serial.printf("\"samples\":%d,\"result\":%d,\"asr_ms\":%.3f,\"match_ms\":%.3f,"
                "\"total_ms\":%.3f,\"frontend_us\":%u,\"frontend_frames\":%d,"
                "\"best\":%d,\"gap\":%.8f,\"second\":%.8f,\"accepted\":%s,\"profile_us\":[",
                fixture_samples, result, (recognized-start)/1000.0, (end-recognized)/1000.0,
                (end-start)/1000.0, front, asr_frontend_reused_frames(), best, gap, second,
                accepted ? "true" : "false");
  const uint32_t* profile = asr_profile();
  for (int i = 0; i < ASR_PROF_N; i++) Serial.printf("%s%u", i ? "," : "", profile[i]);
  Serial.println("]}");
  if (TRIAL_SHARED_ARENA) asr_release_scratch();
}

static void trial_replay() {
  if (!fixture_samples) { Serial.println("{\"error\":\"no fixture\"}"); return; }
  replay_active = true;
  reset_preview();
  st = St::Idle; above = below = speech_chunks = 0;
  pre_fill = pre_pos = 0; noise = 35; dc = 0; deaf_until = 0;
  unsigned before = response_count;
  int16_t input[CHUNK];
  uint32_t start = millis(), max_late = 0;
  int delivered = 0;
  // 500 ms quiet preroll, original recording, then 1 s quiet endpoint input.
  int total = PREROLL_CHUNKS * CHUNK + fixture_samples + SR;
  for (int p = 0; p < total; p += CHUNK) {
    uint32_t deadline = start + (uint32_t)(p * 1000LL / SR);
    while ((int32_t)(millis() - deadline) < 0) M5.delay(1);
    uint32_t late = millis() - deadline;
    if (late > max_late) max_late = late;
    for (int j = 0; j < CHUNK; j++) {
      int k = p + j - PREROLL_CHUNKS * CHUNK;
      input[j] = k >= 0 && k < fixture_samples ? fixture_pcm[k] : 0;
    }
    handle_chunk(input);
    delivered++;
    if (response_count != before) break;
  }
  settle_preview();
  replay_active = false;
  Serial.printf("{\"control\":\"replay_done\",\"responses\":%u,\"delivered_chunks\":%d,"
                "\"max_arrival_late_ms\":%u,\"preview_cancellations\":%u}\n",
                response_count-before, delivered, max_late, preview_cancellations);
}

static void trial_command(const char* command) {
  mic_stop(); reset_preview(); wait_tts_prepare();
  serial_hold = true;
  if (!strcmp(command, "status")) trial_status();
  else if (!strcmp(command, "listen")) {
    serial_hold = false;
    Serial.println("{\"control\":\"listen\"}");
  }
  else if (!strncmp(command, "mode ", 5)) {
    bool ok = false;
    for (const auto& profile : trial_profiles) if (!strcmp(command + 5, profile.name)) {
      ok = apply_trial_flags(profile.flags);
      if (ok) trial_name = profile.name;
      break;
    }
    if (ok) {
      Serial.print("{\"control\":\"mode\","); json_str("trial", trial_name);
      Serial.printf("\"flags\":%u}\n", trial_flags);
    } else Serial.println("{\"error\":\"unknown or unavailable profile\"}");
  } else if (!strncmp(command, "flags ", 6)) {
    if (apply_trial_flags(strtoul(command+6, nullptr, 0))) {
      trial_name = "combined";
      Serial.printf("{\"control\":\"flags\",\"flags\":%u}\n", trial_flags);
    } else Serial.println("{\"error\":\"unavailable flags\"}");
  } else if (!strncmp(command, "temp ", 5)) {
    llm.temperature = strtof(command+5, nullptr);
    Serial.printf("{\"control\":\"temp\",\"temp\":%.3f}\n", llm.temperature);
  } else if (!strncmp(command, "seed ", 5)) {
    llm.rng = strtoull(command+5, nullptr, 0);
    Serial.println("{\"control\":\"seed\"}");
  } else if (!strcmp(command, "exportcache")) trial_export();
  else if (!strncmp(command, "load ", 5)) trial_load(atoi(command+5));
  else if (!strcmp(command, "asr")) trial_asr();
  else if (!strcmp(command, "replay")) trial_replay();
  else Serial.println("{\"error\":\"unknown command\"}");
  back_to_listening();
  if (!serial_hold) {
    st = St::Calib; noise = 0; calib_n = 0;
    draw_status("まわりの音をはかってるよ…", TFT_YELLOW);
  }
}
