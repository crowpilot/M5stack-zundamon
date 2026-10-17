# PlatformIO の pre スクリプト: ../models/*.bin から src/asr_model_data.h と src/llm_model_data.h を作る
# （モデルを差し替えたら .bin を置き換えてビルドし直すだけでよい。ヘッダは生成物なので git に入れない）
import os, struct
try:
    Import("env")  # noqa: F821  PlatformIO から呼ばれたとき
    ROOT = env.subst("$PROJECT_DIR")  # noqa: F821
except NameError:
    ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(ROOT, "..", "models")
SRC = os.path.join(ROOT, "src")
ASR = os.path.join(MODELS, "kana_asr.bin")
LLM = os.path.join(MODELS, "zunda-small-tagend-679_q8.bin")
TOK = os.path.join(MODELS, "tok_tagend.bin")


def stale(out, *ins):
    return not os.path.exists(out) or any(os.path.getmtime(i) > os.path.getmtime(out) for i in ins)


def bytes_array(f, name, data):
    f.write(f"static const uint8_t {name}[] __attribute__((aligned(4))) = {{\n")
    for i in range(0, len(data), 24):
        f.write(",".join(str(b) for b in data[i:i + 24]) + ",\n")
    f.write(f"}};\nstatic const size_t {name}_len = {len(data)};\n\n")


for p in (ASR, LLM, TOK):
    if not os.path.exists(p):
        raise SystemExit(f"モデルがありません: {os.path.normpath(p)}（README の「モデルと音声データ」を参照）")

out = os.path.join(SRC, "asr_model_data.h")
if stale(out, ASR):
    b = open(ASR, "rb").read()
    n = len(b) // 4
    w = struct.unpack(f"<{n}I", b[:n * 4])
    with open(out, "w") as f:
        f.write("// 自動生成: firmware/gen_headers.py（元は models/kana_asr.bin）\n#pragma once\n#include <stdint.h>\n\n")
        f.write(f"#define ASR_MODEL_WORDS {n}u\nstatic const uint32_t asr_model_blob[ASR_MODEL_WORDS] __attribute__((aligned(4))) = {{\n")
        for i in range(0, n, 8):
            f.write("  " + ",".join(f"0x{x:08x}" for x in w[i:i + 8]) + ",\n")
        f.write("};\n")
    print("generated", out)

out = os.path.join(SRC, "llm_model_data.h")
if stale(out, LLM, TOK):
    with open(out, "w") as f:
        f.write("// 自動生成: firmware/gen_headers.py（元は models/ の LLM とトークナイザ）\n#pragma once\n#include <stdint.h>\n#include <stddef.h>\n\n")
        f.write('#define MODEL_NAME "zunda-small-tagend-679"\n\n')
        bytes_array(f, "model_bin", open(LLM, "rb").read())
        bytes_array(f, "tok_bin", open(TOK, "rb").read())
    print("generated", out)
