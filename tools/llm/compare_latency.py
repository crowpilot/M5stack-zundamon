#!/usr/bin/env python3
"""Compare two serial_latency.py runs and reject output/content changes."""
import argparse
import json
import statistics
from pathlib import Path


def read(path):
    return [obj for line in path.read_text().splitlines()
            if (obj := json.loads(line)).get("kind") == "response"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("baseline", type=Path)
    ap.add_argument("trial", type=Path)
    args = ap.parse_args()
    before, after = read(args.baseline), read(args.trial)
    if not before or len(before) != len(after):
        raise ValueError("response counts differ or are empty")
    fields = ("round", "heard", "reply", "feel", "say", "voice", "prompt_tokens",
              "chunks", "audio_ms", "tokens", "temp", "yomi_skip", "tts_missing")
    for a, b in zip(before, after):
        for field in fields:
            if a[field] != b[field]:
                raise ValueError(f"{a['heard']}: {field} differs: {a[field]!r} != {b[field]!r}")
        if b.get("cached_prompt_tokens") != 2:
            raise ValueError(f"prefix was not reused: {b['heard']}")
    result = []
    for prompt in dict.fromkeys(row["heard"] for row in before):
        a = [row for row in before if row["heard"] == prompt]
        b = [row for row in after if row["heard"] == prompt]
        row = {"heard": prompt, "samples_per_version": len(a), "metrics": {}}
        for metric in ("first_piece_ms", "first_sound_ms", "llm_ms", "done_ms"):
            av, bv = [v[metric] for v in a], [v[metric] for v in b]
            am, bm = statistics.median(av), statistics.median(bv)
            row["metrics"][metric] = {
                "before_median": am, "after_median": bm, "saved_ms": am - bm,
                "reduction_percent": round(100 * (am - bm) / am, 2),
                "before_range": [min(av), max(av)], "after_range": [min(bv), max(bv)],
            }
        result.append(row)
    print(json.dumps({"content_equal": True, "responses_per_version": len(before),
                      "prompts": result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
