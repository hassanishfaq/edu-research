"""Normaliser -- Section 4.1, Eqs. (4.5)-(4.10) and the serialisation of Section 4.2.

    AV(f)            nested VirusTotal behaviour report                     Eq. (4.5)
    F_flat(AV(f))    one-level dictionary  "category.action.field" -> values   Eq. (4.6)
    AV_filtered(f)   F_flat without the unnecessary keys (success flags, pids) Eq. (4.7)
    C                the key set of the schema registry (union over the corpus) Eq. (4.8)
    AV_final(f, c)   value of key c, or the padding value "0" when absent      Eq. (4.9)
    M                the row of the CSV matrix                                 Eq. (4.10)

The row is then serialised as the text a transformer reads:
    [CLS] HK: ... [SEP] PR: ... [SEP] NT: ... [SEP] ... [SEP]
with the categories in the fixed order of the schema registry, so that truncation at
n tokens always removes the same low-priority tail (Section 4.4.2, normaliser row).
"""
from __future__ import annotations

import re
from typing import Any

# The 39 behaviour keys of the thesis CSV (Data.csv; its other columns are av_name, detected_signature, hash) = key set C, schema v1.
SCHEMA_V1: list[str] = [
    "extra",
    "filesystem.copied.dst", "filesystem.copied.src", "filesystem.deleted.path", "filesystem.moved.dst", "filesystem.moved.src",
    "filesystem.opened.path", "filesystem.read.path", "filesystem.replaced.replaced", "filesystem.replaced.replacement", "filesystem.written.path",
    "hooking.method", "hooking.type", "hosts_file",
    "mutex.created.mutex", "mutex.opened.mutex",
    "network.dns.hostname", "network.dns.ip", "network.http.method", "network.http.url", "network.http.user-agent", "network.tcp", "network.udp",
    "process.created.proc", "process.injected.proc", "process.shellcmds.cmd", "process.terminated.proc", "process.tree.name", "process.tree.pid",
    "registry.deleted.key", "registry.set.key", "registry.set.type", "registry.set.val",
    "runtime-dlls.file",
    "service.opened-managers.database", "service.opened-managers.machine", "service.opened.name",
    "windows.searched.class", "windows.searched.name",
]
SCHEMAS = {"v1": SCHEMA_V1}

# Section 4.2 segment codes (the thesis writes HK / NT01 / PR01 / FS01 ...)
CATEGORY_CODES = {"hooking": "HK", "process": "PR", "network": "NT", "registry": "RG", "filesystem": "FS", "mutex": "MX",
                  "service": "SV", "runtime-dlls": "DL", "windows": "WN", "extra": "EX", "hosts_file": "HF"}
DEFAULT_ORDER = ["hooking", "process", "network", "registry", "filesystem", "mutex", "service", "runtime-dlls", "windows", "extra", "hosts_file"]
PAD = "0"
_HEX64 = re.compile(r"[0-9a-fA-F]{64}")
_HEX32 = re.compile(r"^[0-9a-fA-F]{32}$")
# Path canonicalisation (part of k_unnecessary: the drive/user/system prefixes carry no behaviour and cost tokens)
PATH_ALIASES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"(?i)c:\\windows\\system32"), "%SYS32%"),
    (re.compile(r"(?i)c:\\windows\\syswow64"), "%SYSWOW64%"),
    (re.compile(r"(?i)c:\\windows"), "%WINDIR%"),
    (re.compile(r"(?i)c:\\documents and settings\\all users"), "%ALLUSERS%"),
    (re.compile(r"(?i)c:\\documents and settings\\<user>"), "%USERPROFILE%"),
    (re.compile(r"(?i)c:\\documents and settings\\[^\\]+"), "%USERPROFILE%"),
    (re.compile(r"(?i)c:\\users\\[^\\]+"), "%USERPROFILE%"),
    (re.compile(r"(?i)c:\\program files( \(x86\))?"), "%PROGRAMFILES%"),
]


# ---------------------------------------------------------------- Eq. (4.6): flatten
def flatten(obj: Any, prefix: str = "", out: dict[str, list[str]] | None = None) -> dict[str, list[str]]:
    """Nested dict/list -> {"a.b.c": [values...]}.  Lists of dicts fan out into their fields."""
    if out is None:
        out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            flatten(v, f"{prefix}.{k}" if prefix else str(k), out)
    elif isinstance(obj, list):
        if not obj:
            out.setdefault(prefix, [])
        for v in obj:
            flatten(v, prefix, out)
    else:
        out.setdefault(prefix, []).append("" if obj is None else str(obj))
    return out


# ---------------------------------------------------------------- Eq. (4.7): filter
def filter_keys(flat: dict[str, list[str]], drop_keys: list[str]) -> tuple[dict[str, list[str]], list[str]]:
    """Remove k_unnecessary: keys whose last component is in drop_keys (e.g. success, pid)."""
    kept, dropped = {}, []
    for k, v in flat.items():
        last = k.rsplit(".", 1)[-1]
        if last in drop_keys:
            dropped.append(k)
        else:
            kept[k] = v
    return kept, dropped


def _clean_value(v: str, mask_hash: bool, aliases: bool = True) -> str:
    v = v.strip().replace("\n", " ").replace("\r", " ")
    if mask_hash:
        v = _HEX64.sub("<sample>", v)
    if aliases and "\\" in v:
        for pat, rep in PATH_ALIASES:
            v = pat.sub(rep, v)
    return v


# ---------------------------------------------------------------- Eq. (4.9)-(4.10): align + pad
def align(filtered: dict[str, list[str]], schema: list[str], max_items: int = 40, mask_hash: bool = True) -> tuple[dict[str, str], dict[str, Any]]:
    """AV_final(f, c) for every c in C: the comma-joined (de-duplicated) values, or "0" when the key is absent.
    Returns the row and statistics (present / padded / keys outside the schema)."""
    row: dict[str, str] = {}
    present = 0
    for c in schema:
        vals = filtered.get(c)
        if vals:
            seen: list[str] = []
            for v in vals:
                cv = _clean_value(v, mask_hash)
                if cv and cv not in seen:
                    seen.append(cv)
                if len(seen) >= max_items:
                    break
            row[c] = ", ".join(seen) if seen else PAD
            present += 1 if seen else 0
        else:
            row[c] = PAD
    sset = set(schema)
    outside = sorted(k for k, v in filtered.items() if k not in sset and any(str(x).strip() for x in v))
    stats = {"schema_keys": len(schema), "present": present, "padded": len(schema) - present, "outside_schema": outside,
             "n_values": int(sum(len(v) for v in filtered.values()))}
    return row, stats


def category_counts(row: dict[str, str]) -> dict[str, int]:
    """Number of distinct values per behaviour category (hooking, process, ...)."""
    out: dict[str, int] = {c: 0 for c in DEFAULT_ORDER}
    for k, v in row.items():
        cat = k.split(".", 1)[0]
        if v != PAD:
            out[cat] = out.get(cat, 0) + (v.count(", ") + 1)
    return out


# ---------------------------------------------------------------- Section 4.2: serialise
def serialize(row: dict[str, str], order: list[str] | None = None, sep: str = " </s> ", max_chars: int = 0) -> tuple[str, list[dict[str, Any]]]:
    """Row -> "<s> HK: ... </s> PR: ... </s> ..." in the fixed category order.
    Returns the text and the list of segments (code, category, chars, start offset) for the UI / truncation report."""
    order = order or DEFAULT_ORDER
    segments = []
    parts = []
    pos = 4  # after "<s> "
    for cat in order:
        items = []
        for k, v in row.items():
            if k.split(".", 1)[0] != cat or v == PAD:
                continue
            sub = k[len(cat) + 1:] if k.startswith(cat + ".") else k
            sub = sub.replace(".", " ")
            items.append(f"{sub} {v}" if sub else v)
        body = "; ".join(items) if items else "none"
        seg = f"{CATEGORY_CODES.get(cat, cat[:2].upper())}: {body}"
        segments.append({"code": CATEGORY_CODES.get(cat, cat[:2].upper()), "category": cat, "chars": len(seg), "start": pos, "empty": not items})
        parts.append(seg)
        pos += len(seg) + len(sep)
    text = "<s> " + sep.join(parts) + " </s>"
    if max_chars and len(text) > max_chars:
        text = text[:max_chars - 5].rstrip() + " </s>"
    return text, segments


def display_text(text: str) -> str:
    """The thesis writes [CLS] ... [SEP]; RoBERTa's tokenizer uses <s> ... </s>."""
    return text.replace("<s>", "[CLS]").replace("</s>", "[SEP]")


def normalise_report(report: Any, schema: list[str], drop_keys: list[str], order: list[str], max_items: int = 40,
                     mask_hash: bool = True, max_chars: int = 0) -> dict[str, Any]:
    """Full normaliser: Eq. (4.5) -> (4.10) -> serialised text.  Returns everything the UI shows."""
    flat = flatten(report)
    filtered, dropped = filter_keys(flat, drop_keys)
    row, stats = align(filtered, schema, max_items=max_items, mask_hash=mask_hash)
    text, segments = serialize(row, order, max_chars=max_chars)
    return {"flat_keys": len(flat), "dropped_keys": dropped, "row": row, "stats": stats, "category_counts": category_counts(row),
            "text": text, "segments": segments}


def hash_from_name(name: str) -> str | None:
    stem = name.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    stem = stem.rsplit(".", 1)[0]
    return stem.lower() if _HEX32.match(stem) else None
