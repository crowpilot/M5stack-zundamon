// 読み辞書づくり用: LLM に話しかけ（自由生成）と返事（greedy と温度0.8）を大量に作らせ、返事を1行ずつ出す
//   cc -O2 -Ifirmware/src tools/llm/gen_corpus.c firmware/src/zunda_brain.c firmware/src/llm_core.c -lm -o gen && ./gen model.bin tok.bin 3000
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "zunda_brain.h"
static void *rd(const char *p, size_t *n) {
  FILE *f = fopen(p, "rb"); fseek(f, 0, SEEK_END); *n = ftell(f); rewind(f);
  void *b = malloc(*n); fread(b, 1, *n, f); fclose(f); return b;
}
int main(int argc, char **argv) {
  size_t mn, tn; void *mb = rd(argv[1], &mn), *tb = rd(argv[2], &tn);
  int N = atoi(argv[3]);
  Llm llm; if (llm_init(&llm, mb, mn, tb, tn, 256, malloc) || zunda_brain_init(&llm)) return 1;
  llm.rng = 12345;
  int id_a = -1, id_b = -1;
  for (int i = 0; i < llm.tok.vocab_size; i++) {
    if (!strcmp(llm.tok.vocab[i], "Ａ：")) id_a = i;
    if (!strcmp(llm.tok.vocab[i], "Ｂ：")) id_b = i;
  }
  for (int k = 0; k < N; k++) {
    // 話しかけを自由生成: <BOS>Ａ： のあと Ｂ： が出るまで
    char in[400] = {0}; int len = 0, pos = 0, prev = id_a;
    llm.temperature = 1.0f;
    llm_forward(&llm, 1, pos++);
    float *lg = llm_forward(&llm, id_a, pos++);
    for (int s = 0; s < 40; s++) {
      int nx = llm_sample(&llm, lg);
      if (nx == id_b || nx == id_a || nx < 3) break;
      const char *p = llm_decode(&llm, prev, nx);
      if (len + strlen(p) < sizeof(in) - 1) { strcpy(in + len, p); len += strlen(p); }
      lg = llm_forward(&llm, nx, pos++); prev = nx;
    }
    if (!len) continue;
    char out[256]; int feel;
    llm.temperature = 0; zunda_brain_reply(&llm, in, out, sizeof(out), &feel, NULL, NULL); printf("%s\n", out);
    llm.temperature = 0.8f; zunda_brain_reply(&llm, in, out, sizeof(out), &feel, NULL, NULL); printf("%s\n", out);
  }
}
