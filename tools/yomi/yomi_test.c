// cc -Ifirmware/src tools/yomi/yomi_test.c firmware/src/yomi.c -o yt && ./yt < 返事.txt
#include <stdio.h>
#include <string.h>
#include "yomi.h"
int main(void) {
  char line[512], out[1024]; int sk, tot = 0, bad = 0, n = 0;
  while (fgets(line, sizeof line, stdin)) {
    line[strcspn(line, "\n")] = 0;
    yomi_to_kana(line, out, sizeof out, &sk);
    printf("%s\t%s%s\n", line, out, sk ? "\t[skip]" : "");
    n++; tot += sk; bad += sk > 0;
  }
  fprintf(stderr, "%d lines, %d with skipped chars (%d chars)\n", n, bad, tot);
}
