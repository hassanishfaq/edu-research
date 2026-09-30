/* SigStack demo — dashboard logic (no build step, no framework) */
(() => {
  "use strict";
  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => Array.from(el.querySelectorAll(s));
  const REDUCED = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const C = { accent: "#ff7a59", teal: "#3fd0c9", gold: "#f5c451", violet: "#b9a6ff", green: "#5ed49a", red: "#ff6767", muted: "#9c92b5", line: "#33284a", text: "#f1ecfa" };
  const SLOT_COLORS = { category: C.gold };
  const ENGINE_COLORS = ["#3fd0c9", "#7dd3fc", "#b9a6ff", "#5ed49a"];
  const SCHEMA = ["extra", "filesystem.copied.dst", "filesystem.copied.src", "filesystem.deleted.path", "filesystem.moved.dst", "filesystem.moved.src", "filesystem.opened.path", "filesystem.read.path", "filesystem.replaced.replaced", "filesystem.replaced.replacement", "filesystem.written.path", "hooking.method", "hooking.type", "hosts_file", "mutex.created.mutex", "mutex.opened.mutex", "network.dns.hostname", "network.dns.ip", "network.http.method", "network.http.url", "network.http.user-agent", "network.tcp", "network.udp", "process.created.proc", "process.injected.proc", "process.shellcmds.cmd", "process.terminated.proc", "process.tree.name", "process.tree.pid", "registry.deleted.key", "registry.set.key", "registry.set.type", "registry.set.val", "runtime-dlls.file", "service.opened-managers.database", "service.opened-managers.machine", "service.opened.name", "windows.searched.class", "windows.searched.name"];

  async function api(path, opts = {}) {
    const r = await fetch(path, opts);
    if (!r.ok) throw new Error(`${path}: ${r.status} ${await r.text()}`);
    return r.json();
  }
  const postJSON = (path, body) => api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const fmtBytes = (b) => b == null ? "—" : b < 1e3 ? `${b} B` : b < 1e6 ? `${(b / 1e3).toFixed(1)} KB` : b < 1e9 ? `${(b / 1e6).toFixed(2)} MB` : `${(b / 1e9).toFixed(2)} GB`;
  const pct = (x, d = 1) => x == null ? "—" : `${(x * 100).toFixed(d)}%`;
  const f3 = (x) => x == null ? "—" : x.toFixed(3);
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const engColor = (i) => ENGINE_COLORS[i % ENGINE_COLORS.length];

  let STATUS = null, LABELS = null;
  let lastEventId = 0;
  let trPollTimer = null, hoPollTimer = null;
  let ENGINES = [];   // [{folder, display}]

  // =====================================================================
  // Canvas charts (tiny, dependency-free)
  // =====================================================================
  function setupCanvas(canvas) {
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth || canvas.parentElement.clientWidth || 400;
    const h = parseInt(canvas.getAttribute("height") || "220", 10);
    canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
    canvas.style.width = w + "px"; canvas.style.height = h + "px";
    const ctx = canvas.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    return { ctx, w, h };
  }
  function lineChart(canvas, { series, xMin, xMax, yMin = 0, yMax = 1, yFmt = (v) => v, xFmt = (v) => v, xTicks = 5, yTicks = 4, vlines = [] }) {
    const { ctx, w, h } = setupCanvas(canvas);
    const pad = { l: 54, r: 12, t: 12, b: 28 };
    const X = (x) => pad.l + (x - xMin) / (xMax - xMin || 1) * (w - pad.l - pad.r);
    const Y = (y) => h - pad.b - (y - yMin) / (yMax - yMin || 1) * (h - pad.t - pad.b);
    ctx.font = "10.5px IBM Plex Mono, monospace"; ctx.fillStyle = C.muted; ctx.strokeStyle = C.line; ctx.lineWidth = 1;
    for (let i = 0; i <= yTicks; i++) { const v = yMin + (yMax - yMin) * i / yTicks, y = Y(v); ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke(); ctx.textAlign = "right"; ctx.fillText(yFmt(v), pad.l - 6, y + 3); }
    for (let i = 0; i <= xTicks; i++) { const v = xMin + (xMax - xMin) * i / xTicks; ctx.textAlign = "center"; ctx.fillText(xFmt(v), X(v), h - pad.b + 16); }
    for (const vl of vlines) { ctx.save(); ctx.setLineDash([4, 4]); ctx.strokeStyle = vl.color || C.muted; ctx.beginPath(); ctx.moveTo(X(vl.x), pad.t); ctx.lineTo(X(vl.x), h - pad.b); ctx.stroke(); ctx.fillStyle = vl.color || C.muted; ctx.textAlign = "left"; ctx.fillText(vl.label || "", X(vl.x) + 4, pad.t + 10); ctx.restore(); }
    for (const s of series) {
      if (!s.points.length) continue;
      ctx.strokeStyle = s.color; ctx.lineWidth = s.width || 2; ctx.setLineDash(s.dash || []); ctx.beginPath();
      s.points.forEach(([x, y], i) => i ? ctx.lineTo(X(x), Y(y)) : ctx.moveTo(X(x), Y(y))); ctx.stroke(); ctx.setLineDash([]);
      if (s.dots !== false) { ctx.fillStyle = s.color; for (const [x, y] of s.points) { ctx.beginPath(); ctx.arc(X(x), Y(y), 2.6, 0, Math.PI * 2); ctx.fill(); } }
    }
  }
  function barChart(canvas, { categories, series, yMax, yFmt = (v) => v, xFmt = (v) => v, labelEvery = null, colorFn = null, rotate = false }) {
    const { ctx, w, h } = setupCanvas(canvas);
    const pad = { l: 46, r: 8, t: 10, b: rotate ? 62 : 26 };
    const n = categories.length, gw = (w - pad.l - pad.r) / n, bw = gw / (series.length + 0.6);
    const ymax = yMax || Math.max(1e-9, ...series.flatMap((s) => s.values));
    const Y = (v) => h - pad.b - v / ymax * (h - pad.t - pad.b);
    ctx.font = "10.5px IBM Plex Mono, monospace"; ctx.fillStyle = C.muted; ctx.strokeStyle = C.line;
    for (let i = 0; i <= 4; i++) { const v = ymax * i / 4, y = Y(v); ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke(); ctx.textAlign = "right"; ctx.fillText(yFmt(v), pad.l - 6, y + 3); }
    const every = labelEvery || Math.ceil(n / 8);
    categories.forEach((c, i) => {
      if (i % every !== 0) return;
      const x = pad.l + gw * (i + 0.5);
      if (rotate) { ctx.save(); ctx.translate(x, h - pad.b + 8); ctx.rotate(-Math.PI / 4); ctx.textAlign = "right"; ctx.fillStyle = C.muted; ctx.fillText(xFmt(c), 0, 0); ctx.restore(); }
      else { ctx.textAlign = "center"; ctx.fillText(xFmt(c), x, h - pad.b + 16); }
    });
    series.forEach((s, si) => { s.values.forEach((v, i) => { ctx.fillStyle = colorFn ? colorFn(i, si, s) : s.color; const x = pad.l + gw * i + bw * (si + 0.3); ctx.fillRect(x, Y(v), bw, h - pad.b - Y(v)); }); });
  }
  const legend = (el, items) => { el.innerHTML = items.map((i) => `<span><i style="background:${i.color}"></i>${esc(i.name)}</span>`).join(""); };

  // =====================================================================
  // Status / chips / system
  // =====================================================================
  async function refreshStatus() {
    try { STATUS = await api("/api/status"); } catch (e) { return; }
    const s = STATUS;
    ENGINES = s.config.engines;
    const modeChip = $("#chip-mode");
    modeChip.textContent = s.mode === "roberta" ? "encoder RoBERTa (fine-tuned folders)" : s.mode === "mixed" ? "encoder mixed RoBERTa / surrogate" : "encoder surrogate heads (stand-in)";
    modeChip.className = "chip " + (s.mode === "roberta" ? "on" : "warn");
    const ls = s.label_stats || {};
    $("#chip-corpus").textContent = ls.unique_hashes ? `corpus ${ls.unique_hashes.toLocaleString()} hashes · ${(ls.labelled_corpus || 0).toLocaleString()} labelled` : "corpus missing";
    $("#chip-map").textContent = `map ${s.map_version} · schema ${s.schema_version || "—"} · ${s.categories.length} categories`;
    const m = s.metrics_summary && s.metrics_summary.stacking;
    $("#chip-meta").textContent = m ? `stacking ${pct(m.accuracy)} acc · F1 ${pct(m.f1_macro)}` : "stacking not fitted";
    $("#chip-meta").className = "chip " + (m ? "on" : "warn");
    const r = s.resources || {};
    $("#chip-res").textContent = `cpu ${r.cpu_percent != null ? r.cpu_percent.toFixed(0) + "%" : "—"} · ram ${r.rss_mb != null ? r.rss_mb.toFixed(0) + " MB" : "—"}`;
    $("#not-ready").hidden = !!s.ready;
    $("#watch-toggle").checked = !!s.watch_enabled;
    $("#watch-dir").textContent = s.config.watch.dir;
    $("#vt-note").textContent = s.config.acquisition.virustotal_configured ? "VirusTotal key configured: unknown hashes are fetched once and cached by hash." : "No VirusTotal key (acquisition.virustotal_api_key): hash look-ups answer from the corpus and the report cache only.";
    renderSlotCards(s);
    renderSystem(s);
    renderRecent(s.recent || []);
    if (!MAP.built) buildMap(s);
    if (s.training && !trPollTimer) startTrainPolling();
    if (s.holdout_running && !hoPollTimer) startHoldoutPolling();
    $("#tr-params").textContent = `${s.config.train.folds}-fold stratified · up to ${s.config.train.max_samples || "all"} reports · meta-learner ${s.config.train.meta_learner} · β = ${s.config.open_set.known_reject_budget}`;
  }

  function slotColor(name) { const i = ENGINES.findIndex((e) => e.folder === name); return name === "category" ? C.gold : engColor(i < 0 ? 0 : i); }

  function renderSlotCards(s) {
    const el = $("#slot-cards");
    const busy = s.progress && s.progress.slot;
    el.innerHTML = s.slots.map((n) => `
      <div class="slot">
        <div class="dot ${busy === n.slot ? "busy" : ""}" style="background:${slotColor(n.slot)};box-shadow:0 0 10px ${slotColor(n.slot)}"></div>
        <div><div class="name">${esc(n.display)}</div>
          <div class="kind">${n.kind === "roberta" ? "RoBERTa (fine-tuned offline on GPU)" : "surrogate head on the shared TF-IDF/SVD space"} · ${n.n_classes || "—"} classes${n.info.n_train ? " · " + n.info.n_train.toLocaleString() + " train rows" : ""}</div></div>
        <div class="acc"><b>${n.trained ? "trained" : "—"}</b><div class="muted small">${n.info.train_ms ? (n.info.train_ms / 1000).toFixed(1) + " s" : n.kind}</div></div>
      </div>`).join("");
  }

  function renderSystem(s) {
    const cfg = s.config;
    $("#sys-models").innerHTML = `<div class="mf">
      ${cfg.model_slots.map((n) => `<div class="row"><span class="${n.exists ? "ok" : "miss"}">${n.exists ? "●" : "○"}</span><div><b>${esc(n.display)}</b> ${n.exists ? "RoBERTa folder loaded" : "folder missing → surrogate head"}<div class="path">${esc(n.dir)}</div></div></div>`).join("")}
      <div class="row"><span class="${cfg.archive_exists ? "ok" : "miss"}">${cfg.archive_exists ? "●" : "○"}</span><div><b>behaviour reports</b> ${s.cache ? s.cache.hashes.toLocaleString() + " hashes cached" : "cache missing"}<div class="path">${esc(cfg.archive || cfg.reports_dir)}</div></div></div>
      <div class="row"><span class="ok">●</span><div><b>category map</b> ${esc(s.map_version)}<div class="path">${esc(cfg.labels.category_map)}</div></div></div>
      <div class="row"><span class="ok">●</span><div><b>config</b><div class="path">${esc(cfg.config_file)}</div></div></div></div>`;
    const r = s.resources || {};
    $("#sys-uptime").textContent = r.uptime_s != null ? `up ${Math.floor(r.uptime_s / 60)} min` : "";
    const g = (name, val, max, fmt) => `<div class="gauge">${name}<b>${fmt}</b><div class="g"><i style="width:${Math.min(100, val / max * 100)}%"></i></div></div>`;
    $("#gauges").innerHTML = g("process CPU", r.cpu_percent || 0, 100 * (r.cpu_count || 1), `${(r.cpu_percent || 0).toFixed(0)}% of ${r.cpu_count || "?"} cores`)
      + g("process RSS", r.rss_mb || 0, 4096, `${(r.rss_mb || 0).toFixed(0)} MB`)
      + g("system CPU", r.system_cpu_percent || 0, 100, `${(r.system_cpu_percent || 0).toFixed(0)}%`)
      + g("system RAM", r.system_ram_percent || 0, 100, `${(r.system_ram_percent || 0).toFixed(0)}%`)
      + `<div class="gauge">threads<b>${r.threads ?? "—"}</b></div>`;
    const loaded = s.slots.filter((x) => x.kind === "roberta");
    const rows = s.cost_table.map((c) => `<tr class="${c.used ? "hl" : ""}"><td>${esc(c.model)}</td><td>${c.layers}</td><td>${c.hidden}</td><td>${c.n}</td><td>${(c.params / 1e6).toFixed(1)} M</td><td>${c.weights_fp32_mb} MB</td><td>${c.weights_fp16_mb} MB</td><td>${c.weights_int8_mb} MB</td><td>${c.gmacs}</td><td>${c.train_gmacs}</td><td>${c.attention_mb} MB</td><td>—</td></tr>`).join("")
      + loaded.map((x) => `<tr class="ref"><td>loaded: ${esc(x.display)}</td><td>${x.info.layers}</td><td>${x.info.hidden}</td><td>${s.config.roberta.max_length}</td><td>${(x.info.params / 1e6).toFixed(1)} M</td><td>${fmtBytes(x.info.bytes_fp32)}</td><td>${fmtBytes(x.info.bytes_fp16)}</td><td>${fmtBytes(x.info.bytes_int8)}</td><td>${x.info.gmacs_forward}</td><td>—</td><td>${fmtBytes(x.info.attention_map_bytes)}</td><td>${x.info.load_ms} ms load</td></tr>`).join("")
      + s.slots.filter((x) => x.kind === "surrogate" && x.info.params).map((x) => `<tr class="ref"><td>stand-in: ${esc(x.display)}</td><td>—</td><td>${s.surrogate_features ? s.surrogate_features.latent.replace("truncated SVD ", "") : "—"}</td><td>—</td><td>${(x.info.params / 1e3).toFixed(1)} k</td><td>${fmtBytes(x.info.bytes_fp32)}</td><td>${fmtBytes(x.info.bytes_fp16)}</td><td>${fmtBytes(x.info.bytes_int8)}</td><td>≈0</td><td>—</td><td>—</td><td>${x.info.train_ms} ms fit</td></tr>`).join("");
    $("#cost-table").innerHTML = `<tr><th>model</th><th>L</th><th>d</th><th>n</th><th>params</th><th>FP32</th><th>FP16</th><th>INT8</th><th>fwd GMAC</th><th>train GMAC</th><th>attn maps</th><th>measured</th></tr>${rows}`;
    $("#cost-note").textContent = "analytical rows: Table 4.14 closed forms · loaded / stand-in rows: measured from the state dict";
    const v = s.versions || {}, t = s.tokenizer || {};
    $("#sys-env").innerHTML = Object.entries({ python: v.python, numpy: v.numpy, "scikit-learn": v.sklearn, transformers: v.transformers || "not installed", torch: v.torch || "not installed", cuda: v.cuda ? "yes" : "no (CPU)",
      tokenizer: t.kind === "roberta-bpe" ? `${t.name} (vocab ${t.vocab})` : "approximate" + (t.error ? " · " + t.error : ""), "model version": s.model_version, "category map": s.map_version, "schema": s.schema_version || "—",
      "τ₄ threshold": s.threshold != null ? s.threshold.toFixed(3) + (typeof s.config.open_set.threshold === "string" ? " (auto)" : "") : "—",
      "surrogate space": s.surrogate_features ? `${s.surrogate_features.latent} · ${pct(s.surrogate_features.explained_variance, 0)} variance · built in ${(s.surrogate_features.build_ms / 1000).toFixed(0)} s` : (s.features_ready ? "—" : "building…") })
      .map(([k, val]) => `<span>${esc(k)}</span><span>${esc(val)}</span>`).join("");
  }

  function renderRecent(list) {
    $("#recent-table").innerHTML = `<tr><th>time</th><th>report</th><th>source</th><th>tokens</th><th>engines</th><th>consensus</th><th>verdict</th><th>conf</th><th>versions</th></tr>` +
      (list.length ? list.map((r) => `<tr><td>${new Date(r.t * 1000).toLocaleTimeString()}</td><td>${esc(r.name)}</td><td>${esc(r.source)}</td><td>${r.n_tokens}${r.truncated ? " ✂" : ""}</td><td>${r.score ?? "—"}</td><td>${esc(r.consensus || "—")}</td><td class="${r.unknown ? "unk" : "ok"}">${r.unknown ? "UNKNOWN · possible zero-day" : esc(r.category)}</td><td>${pct(r.confidence)}</td><td>${esc(r.versions.model)} · ${esc(r.versions.category_map)} · ${esc(r.versions.schema)}</td></tr>`).join("")
        : `<tr><td colspan="9" class="muted">No verdicts yet.</td></tr>`);
  }

  // =====================================================================
  // Event log
  // =====================================================================
  async function pollEvents() {
    try {
      const { events } = await api(`/api/events?since=${lastEventId}`);
      if (events.length) {
        const log = $("#log");
        for (const e of events) {
          lastEventId = e.id;
          const d = document.createElement("div");
          d.className = `e ${e.level}`;
          d.innerHTML = `<span>${new Date(e.t * 1000).toLocaleTimeString()}</span><span class="lv">${esc(e.level)}</span><span>${esc(e.msg)}</span>`;
          log.appendChild(d);
        }
        while (log.children.length > 400) log.removeChild(log.firstChild);
        log.scrollTop = log.scrollHeight;
        $("#log-count").textContent = `${lastEventId} events`;
      }
    } catch (e) { /* server restarting */ }
  }

  // =====================================================================
  // Samples + dropzone
  // =====================================================================
  let SAMPLES = null;
  async function loadSamples() {
    SAMPLES = (await api("/api/samples")).samples;
    const sel = $("#sample-cat");
    const cats = [...new Set(SAMPLES.map((s) => s.category))];
    sel.innerHTML = cats.map((c) => `<option value="${esc(c)}">${esc(c)}</option>`).join("");
    sel.onchange = renderSampleList; renderSampleList();
  }
  function renderSampleList() {
    const cat = $("#sample-cat").value;
    const items = SAMPLES.filter((s) => s.category === cat).slice(0, 6);
    $("#sample-list").innerHTML = items.map((s) => `<button type="button" title="${esc(Object.entries(s.engines_display).map(([e, v]) => e + ": " + v).join("\n"))}" data-path="${esc(s.path)}"><span class="h">${esc(s.hash)}</span><span class="e">${ENGINES.map((e) => `<i class="${s.engines[e.folder] ? "on" : ""}" title="${esc(e.display)}"></i>`).join("")}</span></button>`).join("");
    $$("#sample-list button").forEach((b) => b.onclick = () => analyzeSample(b.dataset.path));
  }
  async function analyzeSample(path) {
    setScanning(true);
    try { renderAnalysis(await postJSON("/api/analyze_sample", { path })); } catch (e) { alert(e.message); } finally { setScanning(false); }
  }
  async function analyzeFile(file) {
    setScanning(true);
    const fd = new FormData(); fd.append("file", file);
    try { renderAnalysis(await api("/api/analyze", { method: "POST", body: fd })); } catch (e) { alert(e.message); } finally { setScanning(false); }
  }
  function setScanning(on) { $("#dropzone").classList.toggle("over", on); $("#choose-btn").disabled = on; }
  function setupDropzone() {
    const dz = $("#dropzone"), inp = $("#file-input");
    $("#choose-btn").onclick = (e) => { e.stopPropagation(); inp.click(); };
    dz.onclick = () => inp.click();
    dz.onkeydown = (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); inp.click(); } };
    inp.onchange = () => { if (inp.files[0]) analyzeFile(inp.files[0]); inp.value = ""; };
    ["dragenter", "dragover"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("over"); }));
    ["dragleave", "drop"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("over"); }));
    dz.addEventListener("drop", (e) => { const f = e.dataTransfer.files[0]; if (f) analyzeFile(f); });
    document.addEventListener("dragover", (e) => e.preventDefault()); document.addEventListener("drop", (e) => e.preventDefault());
  }

  // =====================================================================
  // Analysis rendering: schema grid, token meter, serialised text, map, verdict
  // =====================================================================
  function buildSchemaGrid() {
    $("#schema-grid").innerHTML = SCHEMA.map((k) => `<i data-key="${esc(k)}" data-cat="${esc(k.split(".")[0])}" title="${esc(k)}"></i>`).join("");
  }
  async function renderAnalysis(res) {
    const v = res.verdict;
    if (v.status === "error") { $("#verdict-empty").hidden = false; $("#verdict-empty").textContent = v.message; $("#verdict").hidden = true; return; }
    const n = res.normalizer, st = n.stats;
    $("#norm-geom").textContent = `${res.file.size_bytes.toLocaleString()} B JSON → ${st.present}/${st.schema_keys} keys filled → ${n.text.length.toLocaleString()} chars`;
    // schema grid animation
    const cells = $$("#schema-grid i"); cells.forEach((c) => c.classList.remove("on"));
    const row = res.row_preview || {};
    const cc = n.category_counts || {};
    cells.forEach((c, i) => { const k = c.dataset.key; const on = row[k] != null ? row[k] !== "0" : (cc[c.dataset.cat] || 0) > 0; if (on) setTimeout(() => c.classList.add("on"), REDUCED ? 0 : 25 * i); c.title = `${k}: ${row[k] != null ? row[k] : (on ? "present" : "padded")}`; });
    $("#schema-stats").textContent = `${st.present} present · ${st.padded} padded · ${n.dropped_keys.length} keys dropped (k_unnecessary)${st.outside_schema.length ? " · " + st.outside_schema.length + " outside schema" : ""}`;
    // token meter
    const t = res.tokens, maxShow = Math.max(t.max_length * 1.6, t.n_tokens * 1.05);
    const fill = $("#tm-fill"); fill.classList.toggle("over", t.truncated); fill.style.width = "0";
    $("#tm-cut").style.left = `${t.max_length / maxShow * 100}%`;
    requestAnimationFrame(() => { fill.style.width = `${Math.min(100, t.n_tokens / maxShow * 100)}%`; });
    $("#tm-text").textContent = `${t.n_tokens.toLocaleString()} tokens (${t.kind}) · n = ${t.max_length}${t.truncated ? " → truncated, tail dropped" : " → fits"} · ${t.ms} ms`;
    // serialised text with [CLS]/[SEP] highlights and the cut marker
    const sv = $("#serial"); sv.innerHTML = "";
    const txt = res.text_display;
    const parts = txt.split(/(\[SEP\]|\[CLS\])/);
    let pos = 0, i = 0;
    for (const p of parts) {
      const span = document.createElement("span");
      if (p === "[SEP]" || p === "[CLS]") { span.className = "sp seg"; span.textContent = " " + p + " "; }
      else { span.className = "seg"; span.innerHTML = esc(p).replace(/^(\s*)([A-Z]{2}):/, "$1<span class=\"code\">$2:</span>"); }
      span.style.animationDelay = `${i++ * 60}ms`;
      if (t.truncated && t.cut_chars != null && pos <= t.cut_chars && pos + p.length > t.cut_chars) { span.insertAdjacentHTML("beforeend", `<span class="cut"> ◀ n = ${t.max_length} tokens ▶ </span>`); }
      pos += p.length; sv.appendChild(span);
    }
    if (txt.length < n.text.length) sv.insertAdjacentHTML("beforeend", `<span class="muted"> … (${(n.text.length - txt.length).toLocaleString()} more chars)</span>`);
    $("#segs").innerHTML = n.segments.map((s) => `<span class="${s.kept || ""} ${s.empty ? "empty" : ""}" title="${esc(s.category)}: ${s.chars} chars">${esc(s.code)} ${s.empty ? "∅" : s.chars}</span>`).join("");
    if (v.status !== "ok") { $("#verdict-empty").hidden = false; $("#verdict-empty").textContent = v.message; $("#verdict").hidden = true; renderLabelBoxOnly(res); return; }
    try { await animatePipeline(res); } catch (e) { console.warn("map animation", e); }
    renderVerdict(res);
    refreshStatus();
  }
  function renderLabelBoxOnly(res) { /* nothing: verdict hidden; labels shown after training */ }

  function labelBoxHtml(res) {
    const L = res.labels;
    if (!L) return `<div class="muted small">No engine verdicts for this report (unlabelled sample): the label service has nothing to map; the verdict below is the detector’s alone.</div>`;
    const rows = Object.entries(L.raw).map(([e, s]) => `<div class="r"><span class="e">${esc(STATUS ? (STATUS.config.engines.find((x) => x.folder === e) || {}).display || e : e)}</span><span class="s" title="${esc(s)}">${esc(s)}</span><span class="c ${L.mapped[e] === "Unmapped" ? "unm" : ""}">→ ${esc(L.mapped[e])}</span></div>`).join("");
    return rows + `<div class="tot">Eq. (4.4) score <b>${L.score}</b> of ${ENGINES.length} engines · consensus <b>${esc(L.consensus)}</b>${L.disputed ? " · <span class=\"unk\">disputed</span>" : (L.agreement === true ? " · engines agree" : "")} · map ${esc(L.map_version)}</div>`;
  }

  // ---------------------------------------------------------------- pipeline map
  const MAP = { built: false, boxes: [] };
  const svgEl = (tag, attrs = {}, text) => { const e = document.createElementNS("http://www.w3.org/2000/svg", tag); for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v); if (text != null) e.textContent = text; return e; };
  const STAGES = [["Intake", ["hash, size", "de-duplicate"]], ["Acquisition", ["corpus / cache", "sandbox · VirusTotal"]], ["Label service", ["Eq. (4.4) score", "category map"]], ["Normaliser", ["flatten · filter", "schema · serialise"]], ["Encoder", ["tokenise n = 512", "model slots"]], ["Stacking", ["meta-learner", "τ₄ unknown rule"]], ["Verdict store", ["category · conf.", "3 versions"]]];
  function buildMap(s) {
    const svg = $("#pipemap"); svg.innerHTML = ""; MAP.boxes = [];
    const w = 150, gap = 20, x0 = 12, y = 60, h = 96;
    STAGES.forEach(([title, sub], i) => {
      const x = x0 + i * (w + gap);
      const g = svgEl("g", { id: `st-${i}` });
      g.appendChild(svgEl("rect", { x, y, width: w, height: h, class: `box st${i}`, rx: 12 }));
      g.appendChild(svgEl("text", { x: x + 12, y: y + 24, class: "t" }, title));
      sub.forEach((line, j) => g.appendChild(svgEl("text", { x: x + 12, y: y + 44 + j * 14, class: "s" }, line)));
      g.appendChild(svgEl("text", { x: x + 12, y: y + 84, class: "ms", id: `ms-${i}` }, ""));
      svg.appendChild(g);
      MAP.boxes.push({ x, y, w, h, cx: x + w / 2, cy: y + h / 2 });
      if (i) svg.insertBefore(svgEl("path", { d: `M${x - gap},${y + h / 2} L${x},${y + h / 2}`, class: "link" }), svg.firstChild);
    });
    const b = MAP.boxes[4], b5 = MAP.boxes[6];
    svg.appendChild(svgEl("text", { x: MAP.boxes[2].x, y: 30, class: "s", id: "map-slots" }, "slots · " + s.slots.map((x) => `${x.slot === "category" ? "direct" : x.engine}: ${x.kind}`).join(" · ")));
    svg.appendChild(svgEl("text", { x: b5.x, y: y + h + 34, class: "s", id: "map-versions" }, `model ${s.model_version} · map ${s.map_version} · schema ${s.schema_version || "—"}`));
    MAP.built = true;
  }
  function movePacket(from, to, { cls = "", label = "", dur = 450 } = {}) {
    return new Promise((resolve) => {
      const svg = $("#pipemap");
      if (REDUCED) return resolve();
      const c = svgEl("circle", { r: 6, class: `pk ${cls}` }), t = svgEl("text", { class: "pk-l" }, label);
      svg.appendChild(c); svg.appendChild(t);
      const x0 = from.x + from.w, y0 = from.cy, x1 = to.x, y1 = to.cy, t0 = performance.now();
      let done = false;
      const finish = () => { if (done) return; done = true; c.remove(); t.remove(); resolve(); };
      const guard = setTimeout(finish, dur + 300);
      const step = (now) => {
        if (done) return;
        const p = Math.min(1, (now - t0) / dur), e = p < .5 ? 2 * p * p : -1 + (4 - 2 * p) * p;
        const cx = x0 + (x1 - x0) * e, cy = y0 + (y1 - y0) * e + Math.sin(p * Math.PI) * -16;
        c.setAttribute("cx", cx); c.setAttribute("cy", cy); t.setAttribute("x", cx - 10); t.setAttribute("y", cy - 10);
        if (p < 1) requestAnimationFrame(step); else { clearTimeout(guard); finish(); }
      };
      requestAnimationFrame(step);
    });
  }
  const glow = (i, on) => { const r = $(`#st-${i} rect`); if (r) r.classList.toggle("active", on); };
  async function animatePipeline(res) {
    const T = res.timings_ms, B = MAP.boxes;
    const msOf = [null, T.acquire, T.label_service, T.normalise + T.tokenise, T.encoder, T.stacking, null];
    B.forEach((_, i) => { const e = $(`#ms-${i}`); if (e) e.textContent = msOf[i] != null ? `${msOf[i].toFixed(1)} ms` : ""; });
    $("#map-caption").textContent = "report → hash → acquisition → labels → row → tokens → slots → meta-learner → verdict";
    const labels = [fmtBytes(res.payload.report_bytes), res.acquisition.source.split(" ")[0], res.labels ? `score ${res.labels.score}` : "no labels", `${res.payload.tokens_sent} tok`, `${res.verdict.slots.length}×${STATUS ? STATUS.categories.length : 9} p`, res.verdict.open_set.unknown ? "unknown" : res.verdict.category.slice(0, 12)];
    for (let i = 0; i < B.length - 1; i++) {
      glow(i, true);
      await movePacket(B[i], B[i + 1], { cls: i < 2 ? "rep" : i < 4 ? "txt" : "", label: labels[i], dur: 420 });
      glow(i, false);
    }
    glow(6, true); await sleep(200); glow(6, false);
    $("#map-caption").textContent = `verdict in ${T.total.toFixed(1)} ms · ${res.tokens.n_tokens} tokens${res.tokens.truncated ? " (cut at " + res.tokens.max_length + ")" : ""} · ${res.verdict.slots.length} slot replies (${fmtBytes(res.payload.slot_reply_bytes)})`;
    $("#map-versions").textContent = `model ${res.verdict.versions.model} · map ${res.verdict.versions.category_map} · schema ${res.verdict.versions.schema}`;
  }

  function renderVerdict(res) {
    const v = res.verdict, unk = v.open_set.unknown;
    $("#verdict-empty").hidden = true; $("#verdict").hidden = false;
    $("#verdict-time").textContent = new Date().toLocaleTimeString();
    const ring = $("#ring-fg"); ring.style.stroke = unk ? C.red : C.accent;
    requestAnimationFrame(() => { ring.style.strokeDashoffset = 326.7 * (1 - v.confidence); });
    $("#ring-txt").textContent = pct(v.confidence, 0);
    const badge = $("#verdict-badge"); badge.textContent = unk ? "unknown category · possible zero-day" : "known category"; badge.className = "badge " + (unk ? "unknown" : "");
    if (v.short_report) badge.insertAdjacentHTML("afterend", `<span class="badge short" id="badge-short">short report · possible evasion</span>`); const old = $("#badge-short"); if (old && !v.short_report) old.remove();
    $("#verdict-cat").textContent = unk ? `nearest: ${v.category}` : v.category;
    $("#verdict-sub").textContent = `confidence ${v.confidence.toFixed(3)} ${unk ? "<" : "≥"} τ₄ ${v.open_set.threshold.toFixed(3)} (${v.open_set.method}) · ${res.tokens.n_tokens} tokens${v.agrees_with_labels === true ? " · agrees with the engine consensus" : v.agrees_with_labels === false ? " · differs from the engine consensus" : ""}`;
    $("#top5").innerHTML = v.top5.map((t) => `<div class="row"><div><div class="lbl">${esc(t.category)}</div><div class="bar"><i data-w="${t.prob * 100}"></i></div></div><div class="pct">${pct(t.prob)}</div></div>`).join("");
    $("#votes").innerHTML = v.slots.map((s) => `<div class="vote ${s.slot === "category" ? "direct" : ""} ${s.top_category === v.category ? "agree" : ""}"><div class="n">${esc(s.slot === "category" ? "direct" : s.engine)}</div><div class="f">${esc(s.top_label)}${s.slot !== "category" ? "<br>→ " + esc(s.top_category) : ""}</div><div class="w">p=${s.top_prob.toFixed(2)} · ${s.kind === "roberta" ? "RoBERTa" : "stand-in"} · ${s.ms.toFixed(1)} ms</div><div class="p" data-w="${s.category_prob * 100}" style="background:${slotColor(s.slot)}"></div></div>`).join("");
    requestAnimationFrame(() => { $$("#top5 .bar i, .vote .p").forEach((b) => b.style.width = b.dataset.w + "%"); });
    $("#labelbox").innerHTML = labelBoxHtml(res);
    $("#meta-line").textContent = `ŷ = meta([ ${v.slots.map((s) => (s.slot === "category" ? "p_direct" : "p_" + s.engine)).join(" ; ")} ]) → ${v.meta.input_dim}-dim stacked input → ${v.top5[0].category} · ${v.meta.kind} · ${v.meta.ms} ms`;
    const t = res.timings_ms;
    $("#timings").innerHTML = `<span>acquire <b>${t.acquire} ms</b></span><span>normalise <b>${t.normalise} ms</b></span><span>tokenise <b>${t.tokenise} ms</b></span><span>labels <b>${t.label_service} ms</b></span><span>encoder <b>${t.encoder} ms</b></span><span>stacking <b>${t.stacking} ms</b></span><span>total <b>${t.total} ms</b></span>`;
  }

  // =====================================================================
  // Label service panel
  // =====================================================================
  async function loadLabels() {
    try { LABELS = await api("/api/labels"); } catch (e) { return; }
    const st = LABELS.stats; if (!st || !st.unique_hashes) return;
    const engines = STATUS ? STATUS.config.engines : [];
    $("#lab-hashes").textContent = `${st.unique_hashes.toLocaleString()} unique hashes · ${st.labelled_corpus.toLocaleString()} in the labelled corpus (min engines ${st.min_engines})`;
    const sd = st.score_distribution, tot = Object.values(sd).reduce((a, b) => a + b, 0);
    $("#scorebars").innerHTML = [1, 2, 3].map((k) => `<div class="r"><span>Score(f) = ${k} engine${k > 1 ? "s" : ""}</span><div class="bar"><i style="width:${(sd[k] || 0) / tot * 100}%"></i></div><span class="v">${(sd[k] || 0).toLocaleString()} · ${pct((sd[k] || 0) / tot, 0)}</span></div>`).join("");
    $("#engine-table").innerHTML = `<tr><th>engine</th><th>reports</th><th>signatures</th><th>unmapped</th><th>top families</th></tr>` + engines.map((e) => { const p = st.per_engine[e.folder]; return `<tr><td>${esc(p.display)}</td><td>${p.reports.toLocaleString()}</td><td>${p.families}</td><td>${p.unmapped.toLocaleString()}</td><td class="muted" style="white-space:normal;text-align:left">${p.top_families.slice(0, 5).map(([f, n]) => `${esc(f)} (${n})`).join(", ")}</td></tr>`; }).join("");
    $("#lab-agree").textContent = st.agreement_rate != null ? `agreement after mapping ${pct(st.agreement_rate)} of ${st.multi_engine_hashes.toLocaleString()} multi-engine hashes · ${st.disputed.toLocaleString()} disputed` : "";
    const cats = LABELS.categories, cd = st.consensus_distribution;
    const perEng = engines.map((e, i) => ({ color: engColor(i), values: cats.map((c) => st.per_engine[e.folder].categories[c] || 0), name: st.per_engine[e.folder].display }));
    barChart($("#chart-cats"), { categories: cats, series: [{ color: C.accent, values: cats.map((c) => cd[c] || 0) }, ...perEng], labelEvery: 1, rotate: true, xFmt: (c) => c.replace(" Trojans", "").replace("Fake Security Tools", "Fake sec."), yFmt: (v) => v >= 1000 ? (v / 1000).toFixed(0) + "k" : v.toFixed(0) });
    legend($("#legend-cats"), [{ name: "consensus (labelled corpus)", color: C.accent }, ...perEng.map((p) => ({ name: `${p.name} labels`, color: p.color }))]);
    const th = st.token_stats;
    $("#lab-tok").textContent = `median ${th.median} · p90 ${th.p90} · ${pct(th.share_over_512, 1)} of reports exceed n = 512`;
    barChart($("#chart-tokens"), { categories: th.histogram.bins, series: [{ color: C.teal, values: th.histogram.counts }], labelEvery: 2, xFmt: (b) => b >= 1e5 ? "" : b, yFmt: (v) => v >= 1000 ? (v / 1000).toFixed(0) + "k" : v.toFixed(0), colorFn: (i) => th.histogram.bins[i] >= 512 ? C.red : C.teal });
    legend($("#legend-tokens"), [{ name: "reports fitting n = 512", color: C.teal }, { name: "reports truncated (tail categories dropped)", color: C.red }]);
    $("#map-version").textContent = `category_map.yaml · ${LABELS.map_version} · ${LABELS.rules.length} rules · ${cats.length} categories`;
    $("#rules-table").innerHTML = `<tr><th>engine</th><th>rule (regex, first match wins)</th><th>category</th></tr>` + LABELS.rules.map((r) => `<tr><td>${esc(r.engine)}</td><td style="text-align:left;white-space:normal;word-break:break-all">${esc(r.match)}</td><td>${esc(r.category)}</td></tr>`).join("");
    $("#gaps-table").innerHTML = `<tr><th>engine</th><th>vendor string</th><th>reports</th></tr>` + (st.map_gaps.length ? st.map_gaps.map((g) => `<tr><td>${esc(g.engine)}</td><td style="text-align:left">${esc(g.raw)}</td><td>${g.n}</td></tr>`).join("") : `<tr><td colspan="3" class="muted">every vendor string is mapped</td></tr>`);
    const sel = $("#map-engine"); sel.innerHTML = engines.map((e) => `<option value="${esc(e.folder)}">${esc(e.display)}</option>`).join("");
    $("#map-try").onclick = async () => { const r = await postJSON("/api/labels/map", { engine: sel.value, raw: $("#map-raw").value }); $("#map-result").textContent = `→ ${r.category}${r.rule ? "  (rule " + r.rule + ")" : "  (no rule matched)"}`; };
    $("#map-raw").onkeydown = (e) => { if (e.key === "Enter") $("#map-try").click(); };
  }

  // =====================================================================
  // Training
  // =====================================================================
  async function runTraining() {
    const r = await postJSON("/api/train/run", {});
    if (r.started) { $("#tr-run").disabled = true; $("#tr-progress").hidden = false; startTrainPolling(); }
  }
  function startTrainPolling() { if (trPollTimer) return; trPollTimer = setInterval(pollTraining, 800); pollTraining(); }
  async function pollTraining() {
    let st; try { st = await api("/api/train/state"); } catch (e) { return; }
    renderTraining(st);
    if (!st.training) { clearInterval(trPollTimer); trPollTimer = null; $("#tr-run").disabled = false; $("#tr-progress").hidden = true; $("#tr-status").textContent = st.metrics && st.metrics.n ? `done · ${st.metrics.folds} folds · ${(st.metrics.ms_total / 1000).toFixed(0)} s · ${st.model_version}` : "idle"; refreshStatus(); }
    else { const p = st.progress || {}; $("#tr-status").textContent = "running"; const frac = p.fold ? ((p.fold - 1) + 0.5) / (p.folds || 1) : (p.phase && p.phase.startsWith("stacking") ? .92 : p.phase && p.phase.startsWith("final") ? .97 : .02); $("#tr-bar").style.setProperty("--p", `${frac * 100}%`); $("#tr-progress-txt").textContent = p.fold ? `fold ${p.fold}/${p.folds} · ${p.slot} · ${p.phase}` : (p.phase || "starting"); if (STATUS) renderSlotCards({ ...STATUS, progress: p }); }
  }
  function slotName(name) { if (name === "category") return "direct"; const e = ENGINES.find((x) => x.folder === name); return e ? e.display : name; }
  function renderTraining(st) {
    const H = st.history || [];
    if (H.length) {
      const slots = Object.keys(H[0].slots);
      const T = Math.max(H.length, 2);
      lineChart($("#chart-folds"), { xMin: 1, xMax: T, yMin: 0, yMax: 1, xFmt: (v) => Math.round(v), yFmt: (v) => pct(v, 0), xTicks: T - 1,
        series: slots.map((s) => ({ color: slotColor(s), width: s === "category" ? 2.5 : 1.6, points: H.filter((h) => h.slots[s] && h.slots[s].accuracy != null).map((h) => [h.fold, h.slots[s].accuracy]) })) });
      legend($("#legend-folds"), slots.map((s) => ({ name: s === "category" ? "direct category model (9 categories)" : `${slotName(s)} family model (own families)`, color: slotColor(s) })));
    }
    const m = st.metrics; if (!m || !m.stacking) return;
    const cats = m.categories;
    const row = (name, s, cls = "", note = "") => `<tr class="${cls}"><td>${esc(name)}</td><td>${pct(s.precision_macro)}</td><td>${pct(s.recall_macro)}</td><td>${pct(s.f1_macro)}</td><td>${pct(s.accuracy)}</td><td class="muted">${esc(note)}</td></tr>`;
    $("#strategy-table").innerHTML = `<tr><th>labelling strategy</th><th>precision</th><th>recall</th><th>macro-F1</th><th>accuracy</th><th></th></tr>` +
      Object.entries(m.engines).map(([e, v]) => row(`Single-AV (${v.display}) model`, v.vs_consensus.summary, "", `${v.kind} · ${v.n_families} families`)).join("") +
      row("Direct category model", m.direct.summary, "", m.direct.kind) + row("Voting (soft, all slots)", m.voting.summary, "", "mean of probabilities") +
      row("Multi-AV consensus · stacking (proposed)", m.stacking.summary, "hl", `${m.meta_learner} on ${m.stack_dim} inputs · ${m.n.toLocaleString()} out-of-fold rows`);
    const engs = Object.keys(m.engines);
    const hdr = `<tr><th>category</th>${engs.map((e) => `<th>${esc(m.engines[e].display)} P</th><th>${esc(m.engines[e].display)} F1</th>`).join("")}<th class="gold">stacking P (supp.-wtd)</th><th class="gold">stacking F1 (supp.-wtd)</th><th>meta-learner P</th><th>meta-learner R</th><th>meta-learner F1</th><th>support</th></tr>`;
    const best = (vals) => Math.max(...vals.filter((x) => x != null));
    const body = cats.map((c) => { const sw = m.stacking_support_weighted[c] || {}, mc = m.stacking.per_class[c]; const ps = engs.map((e) => (m.engines[e].own_labels.per_class[c] || {}).precision); const bp = best([...ps, mc.precision]);
      return `<tr><td>${esc(c)}</td>${engs.map((e) => { const pc = m.engines[e].own_labels.per_class[c]; return pc && pc.support ? `<td class="${pc.precision === bp ? "best" : ""}">${f3(pc.precision)}</td><td>${f3(pc.f1)}</td>` : `<td class="muted">n/a</td><td class="muted">n/a</td>`; }).join("")}<td class="gold">${f3(sw.precision)}</td><td class="gold">${f3(sw.f1)}</td><td class="${mc.precision === bp ? "best" : ""}">${f3(mc.precision)}</td><td>${f3(mc.recall)}</td><td>${f3(mc.f1)}</td><td>${mc.support}</td></tr>`; }).join("");
    let grp = "";
    if (m.groups) {
      grp = `<tr class="ref"><td colspan="${3 + engs.length * 2 + 4}">six-family view of Table 4.10 (categories grouped; meta-learner predictions re-grouped)</td></tr>` + m.groups.names.map((g) => { const s = m.groups.stacking[g]; return `<tr class="ref"><td>${esc(g)}</td>${engs.map((e) => { const pc = (m.groups.engines[e] || {})[g]; return pc && pc.support ? `<td>${f3(pc.precision)}</td><td>${f3(pc.f1)}</td>` : `<td>n/a</td><td>n/a</td>`; }).join("")}<td></td><td></td><td>${f3(s.precision)}</td><td>${f3(s.recall)}</td><td>${f3(s.f1)}</td><td>${s.support}</td></tr>`; }).join("");
    }
    $("#percat-table").innerHTML = hdr + body + grp;
    $("#percat-note").textContent = `engine columns: each engine’s family model on its own labelled rows, folded to categories · stacking (supp.-wtd) = Table 4.10 formula · meta-learner = Table 4.11 · n/a = the engine never labels that category`;
    // family tabs
    const tabs = $("#famtabs"); tabs.innerHTML = engs.map((e, i) => `<button type="button" data-e="${esc(e)}" class="${i === 0 ? "on" : ""}">${esc(m.engines[e].display)} · ${m.engines[e].n_families} families · acc ${pct(m.engines[e].family_accuracy, 1)}</button>`).join("");
    const showFam = (e) => { $$("#famtabs button").forEach((b) => b.classList.toggle("on", b.dataset.e === e)); const fm = m.engines[e].family_metrics; const rows = Object.entries(fm).slice(0, 40);
      $("#family-table").innerHTML = `<tr><th>malware name (${esc(m.engines[e].display)})</th><th>category</th><th>precision</th><th>recall</th><th>F1</th><th>support</th></tr>` + rows.map(([f, x]) => `<tr><td style="text-align:left">${esc(f)}</td><td>${esc(mapFamily(e, f))}</td><td>${f3(x.precision)}</td><td>${f3(x.recall)}</td><td>${f3(x.f1)}</td><td>${x.support}</td></tr>`).join("") + (Object.keys(fm).length > 40 ? `<tr><td colspan="6" class="muted">… ${Object.keys(fm).length - 40} more families</td></tr>` : ""); };
    $$("#famtabs button").forEach((b) => b.onclick = () => showFam(b.dataset.e)); if (engs.length) showFam(engs[0]);
    // confidence histogram
    const bins = Array.from({ length: 20 }, (_, i) => (i / 20).toFixed(2));
    barChart($("#chart-conf"), { categories: bins, series: [{ color: C.gold, values: m.confidence_hist }], labelEvery: 4, colorFn: (i) => (i / 20) < m.threshold ? C.red : C.gold, yFmt: (v) => v >= 1000 ? (v / 1000).toFixed(1) + "k" : v.toFixed(0) });
    legend($("#legend-conf"), [{ name: "out-of-fold max-probability of known reports", color: C.gold }, { name: `below τ₄ = ${m.threshold.toFixed(3)} → would be flagged unknown (β budget)`, color: C.red }]);
    $("#conf-note").textContent = `τ₄ = ${m.threshold.toFixed(3)} · Eq. (4.11), β = ${STATUS ? STATUS.config.open_set.known_reject_budget : "—"}`;
  }
  const FAMCACHE = {};
  function mapFamily(e, f) { return (STATUS && STATUS.label_stats && FAMCACHE[e + "|" + f]) || ""; }

  // =====================================================================
  // Hold-out
  // =====================================================================
  let HO = null, HO_SEL = null;
  async function runHoldout() {
    const v = $("#ho-cat").value;
    const r = await postJSON("/api/holdout/run", { categories: v === "*" ? null : [v] });
    if (r.started) { $("#ho-run").disabled = true; $("#ho-summary").textContent = "Re-training without the held-out category…"; startHoldoutPolling(); }
  }
  function startHoldoutPolling() { if (hoPollTimer) return; hoPollTimer = setInterval(pollHoldout, 1000); pollHoldout(); }
  async function pollHoldout() {
    let st; try { st = await api("/api/holdout/state"); } catch (e) { return; }
    if (st.result) renderHoldout(st.result);
    if (!st.running) { clearInterval(hoPollTimer); hoPollTimer = null; $("#ho-run").disabled = false; }
    else { const p = st.progress || {}; $("#ho-summary").textContent = `${p.category} (${p.i}/${p.n}) · ${p.phase} …`; }
  }
  function renderHoldout(z) {
    HO = z;
    const R = z.results.filter((r) => !r.skipped);
    if (!R.length) { $("#ho-summary").textContent = "No result."; return; }
    const mean = (k) => R.reduce((a, r) => a + r[k], 0) / R.length;
    $("#ho-summary").innerHTML = `${z.status === "ok" ? "" : "running · "}${R.length} categor${R.length > 1 ? "ies" : "y"} held out · ${z.rows ? z.rows.toLocaleString() + " reports · " : ""}mean flagged unknown <b>${pct(mean("flagged_unknown"))}</b> · mean known false-unknown <b>${pct(mean("known_false_unknown"))}</b> · mean known macro-F1 <b>${pct(mean("known_macro_f1"))}</b> · ${z.encoder || ""}${z.ms ? " · " + (z.ms / 1000).toFixed(0) + " s" : ""}`;
    $("#ho-table").innerHTML = `<tr><th>held-out category</th><th>unseen</th><th>τ₄</th><th>flagged unknown</th><th>known false-unknown</th><th>nearest known</th><th>known macro-F1</th><th>mean conf. unseen / known</th></tr>` +
      R.map((r) => `<tr class="click ${HO_SEL === r.category ? "sel" : ""}" data-c="${esc(r.category)}"><td>${esc(r.category)}</td><td>${r.n_unseen}</td><td>${r.threshold.toFixed(3)}</td><td class="${r.flagged_unknown >= .5 ? "ok" : "unk"}">${pct(r.flagged_unknown)}</td><td>${pct(r.known_false_unknown)}</td><td>${r.nearest.map((n) => `${esc(n.category)} (${pct(n.share, 0)})`).join(", ")}</td><td>${pct(r.known_macro_f1)}</td><td>${r.mean_conf_unseen.toFixed(3)} / ${r.mean_conf_known.toFixed(3)}</td></tr>`).join("");
    $$("#ho-table tr.click").forEach((tr) => tr.onclick = () => { HO_SEL = tr.dataset.c; renderHoldout(HO); });
    const r = R.find((x) => x.category === HO_SEL) || R[R.length - 1]; HO_SEL = r.category;
    if (!r.hist_known || !r.sweep) return;
    const bins = z.bins || Array.from({ length: 21 }, (_, i) => i / 20);
    const cats = bins.slice(0, -1).map((x) => x.toFixed(2));
    const norm = (a) => { const s = a.reduce((x, y) => x + y, 0) || 1; return a.map((v) => v / s); };
    barChart($("#chart-ho-hist"), { categories: cats, series: [{ color: C.green, values: norm(r.hist_known) }, { color: C.red, values: norm(r.hist_unseen) }], labelEvery: 4, yFmt: (v) => pct(v, 0) });
    legend($("#legend-ho-hist"), [{ name: "known categories (validation, out-of-fold)", color: C.green }, { name: `held out: ${r.category}`, color: C.red }, { name: `x = confidence · τ₄ ${r.threshold.toFixed(2)}`, color: C.muted }]);
    $("#ho-hist-note").textContent = `${r.category} held out · ${r.n_unseen} unseen vs ${r.n_known_val} known`;
    lineChart($("#chart-ho-sweep"), { xMin: 0, xMax: 1, yMin: 0, yMax: 1, xFmt: (v) => v.toFixed(1), yFmt: (v) => pct(v, 0), xTicks: 5,
      series: [{ color: C.accent, width: 2.5, points: r.sweep.map((s) => [s.threshold, s.balanced]) }, { color: C.red, dash: [3, 3], dots: false, points: r.sweep.map((s) => [s.threshold, s.unseen_flagged]) }, { color: C.green, dash: [3, 3], dots: false, points: r.sweep.map((s) => [s.threshold, s.known_accepted]) }],
      vlines: [{ x: r.threshold, color: C.teal, label: "τ₄" }] });
    legend($("#legend-ho-sweep"), [{ name: "balanced accuracy", color: C.accent }, { name: "unseen flagged", color: C.red }, { name: "known accepted", color: C.green }, { name: "τ₄ from the β budget", color: C.teal }]);
  }

  // =====================================================================
  // boot
  // =====================================================================
  async function boot() {
    buildSchemaGrid();
    setupDropzone();
    $("#tr-run").onclick = runTraining;
    $("#ho-run").onclick = runHoldout;
    $("#watch-toggle").onchange = async (e) => { await postJSON("/api/watch", { enabled: e.target.checked }); refreshStatus(); };
    await refreshStatus();
    if (STATUS) { $("#ho-cat").innerHTML = `<option value="*">every category in turn</option>` + STATUS.categories.map((c) => `<option value="${esc(c)}">${esc(c)}</option>`).join(""); }
    loadSamples().then(() => { if (new URLSearchParams(location.search).has("autoscan")) { const b = $("#sample-list button"); if (b) b.click(); } }).catch(() => {});
    loadLabels();
    api("/api/train/state").then(renderTraining).catch(() => {});
    api("/api/holdout/state").then((z) => { if (z.result) renderHoldout(z.result); }).catch(() => {});
    pollEvents(); setInterval(pollEvents, 900); setInterval(refreshStatus, 2500);
    window.addEventListener("resize", () => { api("/api/train/state").then(renderTraining).catch(() => {}); if (HO) renderHoldout(HO); loadLabels(); });
  }
  document.addEventListener("DOMContentLoaded", boot);
})();
