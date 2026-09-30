"""Behaviour acquisition adapter (Section 4.4.2): VirusTotal v3 look-ups by hash, converted to the
schema-v1 nested layout that the normaliser expects.  Only used when acquisition.virustotal_api_key
is set; every fetched report is cached by hash and never re-fetched."""
from __future__ import annotations

import json
import urllib.request
from typing import Any

API = "https://www.virustotal.com/api/v3"


def _get(url: str, key: str) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"x-apikey": key, "accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def fetch_behaviour(h: str, key: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        summary = _get(f"{API}/files/{h}/behaviour_summary", key)
        rep = v3_to_v1(summary)
        try:
            meta = _get(f"{API}/files/{h}", key)
            res = (meta.get("data", {}).get("attributes", {}) or {}).get("last_analysis_results", {}) or {}
            rep["_engine_results"] = {k: v.get("result") for k, v in res.items() if v.get("result")}
        except Exception:
            pass
        return rep, None
    except Exception as e:
        return None, f"VirusTotal fetch failed: {e}"


def v3_to_v1(v3: dict[str, Any]) -> dict[str, Any]:
    """Map the v3 behaviour_summary fields onto the v1 nested categories (best effort)."""
    a = v3.get("data", {}) if "data" in v3 else v3
    if isinstance(a, dict) and "attributes" in a:
        a = a["attributes"]
    g = lambda k: a.get(k) or []
    return {
        "hooking": [], "extra": [],
        "network": {"dns": [{"hostname": d.get("hostname"), "ip": (d.get("resolved_ips") or [None])[0]} for d in g("dns_lookups")],
                    "http": [{"url": c.get("url"), "method": c.get("request_method"), "user-agent": (c.get("request_headers") or {}).get("User-Agent")} for c in g("http_conversations")],
                    "tcp": [f"{c.get('destination_ip')}:{c.get('destination_port')}" for c in g("ip_traffic") if c.get("transport_layer_protocol") == "TCP"],
                    "udp": [f"{c.get('destination_ip')}:{c.get('destination_port')}" for c in g("ip_traffic") if c.get("transport_layer_protocol") == "UDP"]},
        "service": {"opened": [{"name": s} for s in g("services_opened")], "created": [{"name": s} for s in g("services_created")], "started": [{"name": s} for s in g("services_started")],
                    "controlled": [], "deleted": [], "opened-managers": []},
        "process": {"created": [{"proc": p} for p in g("processes_created")], "terminated": [{"proc": p} for p in g("processes_terminated")],
                    "injected": [{"proc": p} for p in g("processes_injected")], "shellcmds": [{"cmd": c} for c in g("command_executions")],
                    "tree": [{"name": p.get("name"), "pid": p.get("process_id")} for p in g("processes_tree")]},
        "hosts_file": None, "windows": {"searched": [{"class": w, "name": None} for w in g("windows_searched")]},
        "runtime-dlls": [{"file": m} for m in g("modules_loaded")],
        "mutex": {"created": [{"mutex": m} for m in g("mutexes_created")], "opened": [{"mutex": m} for m in g("mutexes_opened")]},
        "registry": {"set": [{"key": r.get("key"), "type": None, "val": r.get("value")} for r in g("registry_keys_set")], "deleted": [{"key": r} for r in g("registry_keys_deleted")]},
        "filesystem": {"opened": [{"path": p} for p in g("files_opened")], "read": [], "written": [{"path": p} for p in g("files_written")], "deleted": [{"path": p} for p in g("files_deleted")],
                       "copied": [{"src": c.get("source"), "dst": c.get("destination")} for c in g("files_copied")], "moved": [], "replaced": [], "downloaded": [{"path": p} for p in g("files_dropped")]},
    }
