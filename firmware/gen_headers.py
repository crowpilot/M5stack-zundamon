# PlatformIO の pre スクリプト: ../models/*.bin から src/asr_model_data.h と src/llm_model_data.h を作る
# （モデルを差し替えたら .bin を置き換えてビルドし直すだけでよい。ヘッダは生成物なので git に入れない）
import os, struct, hashlib, re
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

# Cache provenance: target arithmetic ABI changes require a new catalog export.
# UI/serial scheduling changes do not change the arithmetic ABI.
math_abi = re.search(r"^#define LLM_MATH_ABI (\d+)$", open(os.path.join(SRC, "llm_core.h")).read(), re.M).group(1)
identity = hashlib.sha256((f"LLM_MATH_ABI={math_abi};ESP32-Xtensa;float32-LE;xtensa-gcc8.4;"
                          "espressif32-6.9.0;arduino-2.0.17;O2;no-fast-math\0").encode() +
                          open(LLM, "rb").read() + open(TOK, "rb").read()).digest()
identity_path = os.path.join(SRC, "prompt_identity.h")
identity_text = "#pragma once\nstatic const uint8_t prompt_identity[32] = {" + ",".join(map(str, identity)) + "};\n"
if not os.path.exists(identity_path) or open(identity_path).read() != identity_text:
    open(identity_path, "w").write(identity_text)

catalog_path = os.path.join(MODELS, "prompt_cache_679.bin")
catalog_header = os.path.join(SRC, "prompt_catalog.h")
if os.path.exists(catalog_path):
    catalog = open(catalog_path, "rb").read()
    if len(catalog) < 44 or catalog[:8] != b"ZPCAT01\0" or catalog[12:44] != identity:
        raise SystemExit("prompt cache identity mismatch; re-export on the target after model/math changes")
    count = struct.unpack_from("<I", catalog, 8)[0]
    entries = []
    offset = 44
    for i in range(count):
        size = struct.unpack_from("<I", catalog, offset)[0]
        offset += 4
        if offset + size > len(catalog):
            raise SystemExit("truncated prompt catalog")
        entries.append((offset, size))
        offset += size
    if offset != len(catalog):
        raise SystemExit("trailing prompt catalog data")
    if stale(catalog_header, catalog_path, LLM, TOK):
        with open(catalog_header, "w") as f:
            f.write("#pragma once\n#include \"llm_core.h\"\n")
            bytes_array(f, "prompt_catalog_blob", catalog)
            f.write("static const LlmPromptSnapshot prompt_catalog_entries[] = {\n")
            for offset, size in entries:
                f.write(f"{{prompt_catalog_blob + {offset}, {size}}},\n")
            f.write(f"}};\nstatic constexpr size_t prompt_catalog_count = {count};\n")
elif os.path.exists(catalog_header):
    os.remove(catalog_header)

out = os.path.join(SRC, "llm_model_data.h")
if stale(out, LLM, TOK):
    with open(out, "w") as f:
        f.write("// 自動生成: firmware/gen_headers.py（元は models/ の LLM とトークナイザ）\n#pragma once\n#include <stdint.h>\n#include <stddef.h>\n\n")
        f.write('#define MODEL_NAME "zunda-small-tagend-679"\n\n')
        bytes_array(f, "model_bin", open(LLM, "rb").read())
        bytes_array(f, "tok_bin", open(TOK, "rb").read())
    print("generated", out)
