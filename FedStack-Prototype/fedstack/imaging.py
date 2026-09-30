"""Binary-to-image transformation (Section 3.2), entropy, and a minimal PE-header reader.

The file is never executed. Its raw bytes are read as unsigned 8-bit values and laid
out row by row into a grayscale image whose width depends on the file size (Nataraj
et al. rule, as used to build the thesis dataset); the image is then resized to the
dataset resolution (225 x 225) for display and to the backbone resolution (224 x 224)
for inference.
"""
from __future__ import annotations

import hashlib
import io
import math
import struct
from typing import Any

import numpy as np
from PIL import Image

DATASET_RES = 225

# (max file size in bytes, image width)
_WIDTH_TABLE = [
    (10 * 1024, 32), (30 * 1024, 64), (60 * 1024, 128), (100 * 1024, 256),
    (200 * 1024, 384), (500 * 1024, 512), (1000 * 1024, 768),
]


def image_width_for(n_bytes: int) -> int:
    for limit, w in _WIDTH_TABLE:
        if n_bytes < limit:
            return w
    return 1024


def bytes_to_image(data: bytes, target: int | None = DATASET_RES) -> tuple[Image.Image, dict[str, Any]]:
    """Map raw bytes to a grayscale image. Returns (PIL image 'L', geometry info)."""
    n = len(data)
    w = image_width_for(n)
    h = max(1, math.ceil(n / w))
    buf = np.frombuffer(data, dtype=np.uint8)
    pad = w * h - n
    if pad:
        buf = np.concatenate([buf, np.zeros(pad, dtype=np.uint8)])
    arr = buf.reshape(h, w)
    img = Image.fromarray(arr, mode="L")
    info = {"raw_width": w, "raw_height": h, "padding_bytes": pad}
    if target:
        img = img.resize((target, target), Image.BILINEAR)
    return img, info


def is_image_file(data: bytes) -> bool:
    return data[:8] == b"\x89PNG\r\n\x1a\n" or data[:3] == b"\xff\xd8\xff" or data[:2] == b"BM"


def load_image_bytes(data: bytes, target: int | None = DATASET_RES) -> Image.Image:
    img = Image.open(io.BytesIO(data)).convert("L")
    if target:
        img = img.resize((target, target), Image.BILINEAR)
    return img


def to_thumbnail(img: Image.Image, size: int) -> np.ndarray:
    return np.asarray(img.resize((size, size), Image.BILINEAR), dtype=np.uint8)


def to_backbone_array(img: Image.Image, size: int = 224) -> np.ndarray:
    """float32 array in [0,1], shape (1, size, size) — same as torchvision ToTensor on 'L'."""
    a = np.asarray(img.resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.0
    return a[None, :, :]


def png_bytes(img: Image.Image) -> bytes:
    b = io.BytesIO()
    img.save(b, format="PNG")
    return b.getvalue()


def shannon_entropy(data: bytes) -> float:
    if not data:
        return 0.0
    counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256).astype(np.float64)
    p = counts[counts > 0] / len(data)
    return float(-(p * np.log2(p)).sum())


def hashes(data: bytes) -> dict[str, str]:
    return {"md5": hashlib.md5(data).hexdigest(), "sha256": hashlib.sha256(data).hexdigest()}


def hex_preview(data: bytes, n: int = 256) -> list[dict[str, str]]:
    rows = []
    for off in range(0, min(n, len(data)), 16):
        chunk = data[off:off + 16]
        rows.append({
            "offset": f"{off:08x}",
            "hex": " ".join(f"{b:02x}" for b in chunk),
            "ascii": "".join(chr(b) if 32 <= b < 127 else "." for b in chunk),
        })
    return rows


# ---------------------------------------------------------------------------
# Minimal, bounds-checked PE header reader (static metadata only; nothing runs)
# ---------------------------------------------------------------------------
_MACHINES = {0x14C: "x86 (I386)", 0x8664: "x64 (AMD64)", 0x1C0: "ARM", 0xAA64: "ARM64", 0x200: "IA-64"}
_SUBSYSTEMS = {1: "Native", 2: "Windows GUI", 3: "Windows console", 9: "Windows CE GUI", 10: "EFI application"}
_SEC_FLAGS = [(0x20, "CODE"), (0x40, "IDATA"), (0x80, "UDATA"), (0x20000000, "X"), (0x40000000, "R"), (0x80000000, "W")]


def parse_pe(data: bytes) -> dict[str, Any] | None:
    try:
        if len(data) < 0x40 or data[:2] != b"MZ":
            return None
        e_lfanew = struct.unpack_from("<I", data, 0x3C)[0]
        if e_lfanew + 24 > len(data) or data[e_lfanew:e_lfanew + 4] != b"PE\0\0":
            return {"format": "MZ (DOS stub only, no PE signature)"}
        fh = e_lfanew + 4
        machine, nsec, tstamp, _, _, opt_size, chars = struct.unpack_from("<HHIIIHH", data, fh)
        out: dict[str, Any] = {
            "format": "PE", "machine": _MACHINES.get(machine, f"0x{machine:04x}"),
            "sections": nsec, "timestamp": int(tstamp),
            "characteristics": {"dll": bool(chars & 0x2000), "executable": bool(chars & 0x0002), "32bit": bool(chars & 0x0100)},
        }
        opt = fh + 20
        if opt_size >= 2 and opt + 2 <= len(data):
            magic = struct.unpack_from("<H", data, opt)[0]
            out["pe_type"] = {0x10B: "PE32", 0x20B: "PE32+"}.get(magic, f"0x{magic:x}")
            if opt + 68 + 4 <= len(data):
                out["entry_point"] = f"0x{struct.unpack_from('<I', data, opt + 16)[0]:08x}"
                if magic == 0x10B and opt + 32 <= len(data):
                    out["image_base"] = f"0x{struct.unpack_from('<I', data, opt + 28)[0]:08x}"
                elif magic == 0x20B and opt + 32 <= len(data):
                    out["image_base"] = f"0x{struct.unpack_from('<Q', data, opt + 24)[0]:016x}"
                out["subsystem"] = _SUBSYSTEMS.get(struct.unpack_from("<H", data, opt + 68)[0], "other")
        sec_off = opt + opt_size
        secs = []
        for i in range(min(nsec, 32)):
            o = sec_off + 40 * i
            if o + 40 > len(data):
                break
            name = data[o:o + 8].rstrip(b"\0").decode("latin-1", "replace")
            vsize, vaddr, rsize, rptr = struct.unpack_from("<IIII", data, o + 8)
            flags = struct.unpack_from("<I", data, o + 36)[0]
            raw = data[rptr:rptr + rsize] if rptr < len(data) else b""
            secs.append({
                "name": name or "(unnamed)", "virtual_size": int(vsize), "raw_size": int(rsize),
                "flags": "".join(f for bit, f in _SEC_FLAGS if flags & bit and len(f) == 1) or "-",
                "kind": ",".join(f for bit, f in _SEC_FLAGS if flags & bit and len(f) > 1) or "-",
                "entropy": round(shannon_entropy(raw), 3) if raw else None,
            })
        out["section_table"] = secs
        return out
    except Exception as e:  # never let a malformed header break a scan
        return {"format": "PE (unparseable header)", "error": str(e)[:80]}


def describe_bytes(data: bytes) -> dict[str, Any]:
    return {
        "size_bytes": len(data), "entropy": round(shannon_entropy(data), 4),
        **hashes(data), "pe": parse_pe(data), "is_image": is_image_file(data),
    }
