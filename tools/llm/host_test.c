// PCでの確認: テストベクタ（small-tagend）と一致するか
//   cc -O2 -Ifirmware/src tools/llm/host_test.c firmware/src/zunda_brain.c firmware/src/llm_core.c -lm -o host_test && ./host_test models/zunda-small-tagend-679_q8.bin models/tok_tagend.bin "こんにちは。"
#include <stdio.h>
#include <stdlib.h>
#include "zunda_brain.h"
static void *rd(const char *p, size_t *n) {
  FILE *f = fopen(p, "rb"); fseek(f, 0, SEEK_END); *n = ftell(f); rewind(f);
  void *b = malloc(*n); fread(b, 1, *n, f); fclose(f); return b;
}
int main(int argc, char **argv) {
  size_t mn, tn; void *mb = rd(argv[1], &mn), *tb = rd(argv[2], &tn);
  Llm llm; if (llm_init(&llm, mb, mn, tb, tn, 256, malloc)) return 1;
  llm.temperature = 0;  // greedy（テストベクタと同じ条件）
  if (zunda_brain_init(&llm)) { printf("init failed\n"); return 1; }
  for (int i = 3; i < argc; i++) {
    char out[256]; int feel;
    zunda_brain_reply(&llm, argv[i], out, sizeof(out), &feel, NULL, NULL);
    printf("%s\t%s〈%s〉\n", argv[i], out, zunda_feel_name(feel));
  }
}
