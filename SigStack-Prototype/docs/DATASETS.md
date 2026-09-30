# Datasets

The prototype works on **sandbox behaviour reports** together with the **antivirus verdicts** of several engines. No reports are included in this repository. The report corpus of the thesis was obtained from VirusTotal and is not redistributed here.

## 1. The thesis corpus

| Property | Value |
|---|---|
| Source | VirusTotal: file behaviour reports and per-engine detection names |
| Engines | Baidu, Kaspersky, Symantec |
| Reports | 130,113 (Baidu 48,101 · Kaspersky 36,476 · Symantec 45,536) |
| Unique files (MD5) | 76,225, of which 11,513 are flagged by all three engines |
| Vendor signatures | 82 (Baidu) · 75 (Kaspersky) · 73 (Symantec) |
| Standardised categories | 9 (thesis Table 4.3) |

A report of the same file appears once per engine that flagged it. The prototype de-duplicates by hash, so every file is normalised and counted once, and it collects the vendor string of each engine.

## 2. Expected layout

A zip archive (read in place, never extracted) or a folder with the same structure:

```
<root>/
├── Baidu/
│   ├── Win32.Adware.Generic.bb/
│   │   ├── 11885c462e58ec65ab34744a6313aaf0.json
│   │   └── ...
│   └── ...
├── Kaspersky/
│   └── <vendor family string>/<md5>.json
└── Symantec/
    └── <vendor family string>/<md5>.json
```

- The **engine** is the third-last path component and must match `data.engines[].folder` in `config.yaml`.
- The **folder name is the vendor string** that the category map receives.
- The **file name is the MD5** of the sample, which is how the three engines' verdicts are joined.

Set `data.archive` to the zip file (default `datasets/multi_av_behaviour_reports.zip`) or set `data.reports_dir` to the folder. `data.max_hashes` limits the corpus to a deterministic sample, which is useful on small machines.

## 3. Report format (schema v1)

Each JSON file is the VirusTotal *behaviour* object with the top-level categories used by the thesis:

```json
{
  "hooking":      [{"type": "WH_CBT", "method": "SetWindowsHook", "success": true}],
  "process":      {"created": [...], "injected": [...], "shellcmds": [...], "terminated": [...], "tree": [{"pid": 828, "name": "...", "children": []}]},
  "network":      {"dns": [{"hostname": "...", "ip": "..."}], "http": [{"url": "...", "method": "GET", "user-agent": "..."}], "tcp": ["ip:port"], "udp": ["ip:port"]},
  "registry":     {"set": [{"key": "...", "type": "...", "val": "..."}], "deleted": [...]},
  "filesystem":   {"opened": [{"path": "..."}], "read": [...], "written": [...], "deleted": [...], "copied": [{"src": "...", "dst": "..."}], "moved": [...], "replaced": [...], "downloaded": [...]},
  "mutex":        {"created": [{"mutex": "..."}], "opened": [...]},
  "service":      {"opened": [...], "created": [...], "started": [...], "controlled": [...], "deleted": [...], "opened-managers": [...]},
  "runtime-dlls": [{"file": "kernel32.dll", "success": true}],
  "windows":      {"searched": [{"class": "...", "name": "..."}]},
  "extra":        ["IsDebuggerPresent", "..."],
  "hosts_file":   null
}
```

The 39 leaf keys used by the normaliser are listed in `sigstack/flatten.py` (`SCHEMA_V1`). They are identical to the behaviour columns of the thesis CSV (`Data.csv`). Keys that are absent are padded with `"0"`. Keys outside the schema are reported per report.

A **VirusTotal API v3** `behaviour_summary` document is also accepted. It is converted to the v1 layout by `sigstack/vt.py`.

## 4. Building a compatible corpus

1. Choose the engines and collect, for each sample, the engine's detection name and the behaviour report. The VirusTotal API returns both. Respect its rate limits and terms.
2. Write each report to `<root>/<engine>/<detection name>/<md5>.json` for every engine that flags the sample.
3. Extend `category_map.yaml` with rules for any new engine or naming scheme, and bump its `version`.
4. Run `python scripts/prepare_data.py`.

For single look-ups, set `acquisition.virustotal_api_key` in a **local** copy of the configuration and drop a `<hash>.txt` file on the dashboard. The report is fetched once and cached in `data/cache/reports/`. Never commit API keys.

## 5. Preparation pipeline

`python scripts/prepare_data.py` performs the following steps:

1. Scans the archive and groups reports by hash (130,113 reports into 76,225 hashes in about 1 s).
2. Normalises each unique report: flatten, filter, align, pad and serialise (Eqs. 4.5–4.10). It keeps up to 3,000 characters of text, the event counts per category and the number of schema keys present.
3. Counts RoBERTa tokens for every serialised report (the `roberta-base` tokenizer is downloaded once, about 2 MB, and cached).
4. Writes `data/cache/reports.pkl` (≈ 140 MB) and copies five raw reports per consensus category to `data/samples/`, with an index of their engine verdicts.

## 6. Ethics and licensing

- Behaviour reports contain indicators of compromise: domains, IP addresses, file paths and mutex names. Treat them as potentially sensitive threat intelligence.
- VirusTotal data are subject to the [VirusTotal Terms of Service](https://docs.virustotal.com/docs/terms-of-service). Do not redistribute report corpora without permission.
- The prototype never executes samples. Collect behaviour only inside isolated sandboxes.
