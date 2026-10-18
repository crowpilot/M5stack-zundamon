#!/usr/bin/env python3
"""Export prompt snapshots computed by the connected Core2, without flashing it.

Requires pyserial (available in PlatformIO's Python). Opening a USB serial port
can reset some adapters even when DTR/RTS are false. The default waits briefly
for a ready message, then also works with an already-running device; --attach
skips that initial wait. No output file is written until every blob validates.
"""
import argparse
import hashlib
import json
import os
import re
import struct
import tempfile
import time
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CATALOG_MAGIC = b"ZPCAT01\0"
SNAPSHOT_MAGIC = 0x434D4C4C
MAX_SNAPSHOT = 1024 * 1024


def read_line(ser, deadline):
    result = bytearray()
    while time.monotonic() < deadline:
        value = ser.read(1)
        if not value:
            continue
        result.extend(value)
        if value == b"\n":
            return bytes(result).rstrip(b"\r\n")
        if len(result) > 8192:
            raise ValueError("Serial text line exceeds 8192 bytes; framing was lost")
    if result:
        raise TimeoutError("Timed out in a partial serial text line")
    return None


def read_exact(ser, count, deadline):
    result = bytearray()
    while len(result) < count and time.monotonic() < deadline:
        result.extend(ser.read(count - len(result)))
    if len(result) != count:
        raise TimeoutError(f"Truncated snapshot: received {len(result)} of {count} bytes")
    return bytes(result)


def json_line(line):
    try:
        obj = json.loads(line)
    except (ValueError, UnicodeDecodeError):
        return None
    return obj if isinstance(obj, dict) else None


def header_identity(path):
    if not path.exists():
        return None
    match = re.search(r"\bprompt_identity\s*\[\s*32\s*\]\s*=\s*\{([^}]+)\}", path.read_text())
    if not match:
        raise ValueError(f"Cannot parse prompt_identity[32] in {path}")
    values = [int(value.strip(), 0) for value in match.group(1).split(",") if value.strip()]
    if len(values) != 32 or any(value < 0 or value > 255 for value in values):
        raise ValueError(f"Invalid prompt identity in {path}")
    return bytes(values)


def validate_snapshot(blob, identity):
    if len(blob) < 96:
        raise ValueError("Snapshot is shorter than its 96-byte header")
    h = struct.unpack_from("<24I", blob)
    if h[0] != SNAPSHOT_MAGIC or h[1] != 1 or h[2] != len(blob) or h[23] != 0:
        raise ValueError("Invalid snapshot magic, format, length, or reserved field")
    n, skip = h[3:5]
    dim, hidden, layers, heads, kv_heads, vocab, seq_len, group_size, math_abi = h[5:14]
    if not (0 < n <= seq_len and skip in (0, 2) and skip <= n and
            dim and hidden and layers and heads and kv_heads and vocab and math_abi):
        raise ValueError("Invalid snapshot dimensions")
    if dim % heads or heads % kv_heads:
        raise ValueError("Invalid attention dimensions")
    if group_size and (dim % group_size or hidden % group_size):
        raise ValueError("Invalid quantization group size")
    kv_dim = dim * kv_heads // heads
    expected = 96 + n * 4 + 2 * layers * (n - skip) * kv_dim * 4 + vocab * 4
    if expected != len(blob):
        raise ValueError("Snapshot payload length does not match dimensions")
    if zlib.crc32(blob[96:]) != h[22]:
        raise ValueError("Snapshot payload CRC32 mismatch")
    actual_identity = blob[56:88]
    if identity is not None and actual_identity != identity:
        raise ValueError("Snapshot identity differs from the current model/build or another entry")
    if any(token >= vocab for token in struct.unpack_from(f"<{n}I", blob, 96)):
        raise ValueError("Snapshot has an out-of-range token")
    # The complete shape/ABI must agree across entries, independently of identity.
    return actual_identity, h[5:14]


def collect(ser, count, timeout, identity):
    ser.write(b"?exportcache\n")
    ser.flush()
    deadline = time.monotonic() + timeout
    snapshots = []
    shape = None
    while time.monotonic() < deadline:
        line = read_line(ser, deadline)
        if line is None:
            break
        if not line:
            continue
        obj = json_line(line)
        if obj is None:
            print(line.decode("utf-8", errors="replace"), flush=True)
            continue
        if "cache_index" in obj:
            index, size = obj.get("cache_index"), obj.get("bytes")
            if type(index) is not int or index != len(snapshots) or index >= count:
                raise ValueError(f"Unexpected or duplicate cache index: {index!r}")
            if type(size) is not int or not (96 <= size <= MAX_SNAPSHOT):
                raise ValueError(f"Invalid snapshot length: {size!r}")
            blob = read_exact(ser, size, deadline)
            # The binary has arbitrary newline bytes; only the exact byte count
            # determines where it ends. Require the separate framing newline.
            if read_line(ser, deadline) != b"":
                raise ValueError("Missing newline after snapshot binary")
            identity, entry_shape = validate_snapshot(blob, identity)
            if shape is not None and shape != entry_shape:
                raise ValueError("Snapshot model dimensions or arithmetic ABI changed during export")
            shape = entry_shape
            snapshots.append(blob)
            print(f"cache {index + 1}/{count}: {size} bytes, CRC32 verified", flush=True)
        elif obj.get("control") == "export_done":
            if type(obj.get("count")) is not int or obj["count"] != count or len(snapshots) != count:
                raise ValueError(f"Incomplete export: done={obj.get('count')}, received={len(snapshots)}, expected={count}")
            return CATALOG_MAGIC + struct.pack("<I", count) + identity + b"".join(
                struct.pack("<I", len(blob)) + blob for blob in snapshots
            )
        elif ("error" in obj or obj.get("control") in ("export_error", "export_failed") or
              (obj.get("ready") and obj.get("control") != "status")):
            raise RuntimeError(f"Device interrupted/rejected export: {obj}")
        else:
            print(json.dumps(obj, ensure_ascii=False), flush=True)
    raise TimeoutError(f"Export timed out after receiving {len(snapshots)}/{count} snapshots")


def save_catalog(path, data, overwrite):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Validate everything first, then publish the complete file atomically.
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as out:
            out.write(data)
            out.flush()
            os.fsync(out.fileno())
        if overwrite:
            os.replace(temp_name, path)
        else:
            os.link(temp_name, path)  # fails safely if an existing export appeared
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "models/prompt_cache_679.bin")
    parser.add_argument("--identity-header", type=Path, default=ROOT / "firmware/src/prompt_identity.h")
    parser.add_argument("--count", type=int, default=29)
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--ready-timeout", type=float, default=8)
    parser.add_argument("--attach", action="store_true", help="Skip waiting for a startup ready message")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing catalog after a complete validated export")
    args = parser.parse_args()
    if args.count < 1 or args.count > 1024 or args.timeout <= 0 or args.ready_timeout < 0:
        parser.error("count must be 1..1024, timeout positive, and ready-timeout nonnegative")
    if args.output.exists() and not args.overwrite:
        parser.error(f"Output already exists: {args.output}; use another path or --overwrite")
    identity = header_identity(args.identity_header)
    import serial  # --help and offline format helpers do not require pyserial.
    with serial.Serial(port=None, baudrate=args.baud, timeout=0.2, write_timeout=5, exclusive=True) as ser:
        ser.dtr = False
        ser.rts = False
        ser.port = args.port
        ser.open()
        if not args.attach:
            deadline = time.monotonic() + args.ready_timeout
            while time.monotonic() < deadline:
                line = read_line(ser, deadline)
                if line is None:
                    break
                if line:
                    print(line.decode("utf-8", errors="replace"), flush=True)
                obj = json_line(line)
                if obj and obj.get("ready"):
                    break
        catalog = collect(ser, args.count, args.timeout, identity)
    save_catalog(args.output, catalog, args.overwrite)
    print(f"Saved {args.output}: {len(catalog)} bytes; sha256={hashlib.sha256(catalog).hexdigest()}")


if __name__ == "__main__":
    main()
