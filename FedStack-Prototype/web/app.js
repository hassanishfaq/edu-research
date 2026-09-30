/* FedStack demo — dashboard logic (no build step, no framework) */
(() => {
  "use strict";
  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => Array.from(el.querySelectorAll(s));
  const REDUCED = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const C = { amber: "#f3b23f", cyan: "#52cbe3", violet: "#a594ff", green: "#5ed49a", red: "#ff6767", muted: "#8d9ab8", line: "#27324b", text: "#e9eef9" };
  const NODE_COLORS = ["#52cbe3", "#7dd3fc", "#a594ff", "#5ed49a"];

  async function api(path, opts = {}) {
    const r = await fetch(path, opts);
    if (!r.ok) throw new Error(`${path}: ${r.status} ${await r.text()}`);
    return r.json();
  }
  const postJSON = (path, body) => api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const fmtBytes = (b) => b == null ? "—" : b < 1e3 ? `${b} B` : b < 1e6 ? `${(b / 1e3).toFixed(1)} KB` : b < 1e9 ? `${(b / 1e6).toFixed(2)} MB` : `${(b / 1e9).toFixed(2)} GB`;
  const pct = (x, d = 1) => x == null ? "—" : `${(x * 100).toFixed(d)}%`;
  const pretty = (s) => (s || "").replace(/_/g, " ");
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  let STATUS = null;
  let lastEventId = 0;
  let fedPollTimer = null;

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
  function lineChart(canvas, { series, xMin, xMax, yMin = 0, yMax = 1, yLog = false, yFmt = (v) => v, xFmt = (v) => v, xTicks = 5, yTicks = 4, vlines = [] }) {
    const { ctx, w, h } = setupCanvas(canvas);
    const pad = { l: 54, r: 12, t: 10, b: 28 };
    const X = (x) => pad.l + (x - xMin) / (xMax - xMin || 1) * (w - pad.l - pad.r);
    const ty = (y) => yLog ? Math.log10(Math.max(y, 1)) : y;
    const Y = (y) => h - pad.b - (ty(y) - ty(yMin)) / (ty(yMax) - ty(yMin) || 1) * (h - pad.t - pad.b);
    ctx.font = "10.5px IBM Plex Mono, monospace"; ctx.fillStyle = C.muted; ctx.strokeStyle = C.line; ctx.lineWidth = 1;
    const yVals = yLog ? Array.from({ length: Math.ceil(ty(yMax)) - Math.floor(ty(yMin)) + 1 }, (_, i) => Math.pow(10, Math.floor(ty(yMin)) + i)).filter((v) => v >= yMin && v <= yMax)
                       : Array.from({ length: yTicks + 1 }, (_, i) => yMin + (yMax - yMin) * i / yTicks);
    for (const v of yVals) {
      const y = Y(v); ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
      ctx.textAlign = "right"; ctx.fillText(yFmt(v), pad.l - 6, y + 3);
    }
    for (let i = 0; i <= xTicks; i++) {
      const v = xMin + (xMax - xMin) * i / xTicks; ctx.textAlign = "center"; ctx.fillText(xFmt(v), X(v), h - pad.b + 16);
    }
    for (const vl of vlines) {
      ctx.save(); ctx.setLineDash([4, 4]); ctx.strokeStyle = vl.color || C.muted; ctx.beginPath(); ctx.moveTo(X(vl.x), pad.t); ctx.lineTo(X(vl.x), h - pad.b); ctx.stroke();
      ctx.fillStyle = vl.color || C.muted; ctx.textAlign = "left"; ctx.fillText(vl.label || "", X(vl.x) + 4, pad.t + 10); ctx.restore();
    }
    for (const s of series) {
      if (!s.points.length) continue;
      ctx.strokeStyle = s.color; ctx.lineWidth = s.width || 2; ctx.setLineDash(s.dash || []); ctx.beginPath();
      s.points.forEach(([x, y], i) => i ? ctx.lineTo(X(x), Y(y)) : ctx.moveTo(X(x), Y(y))); ctx.stroke(); ctx.setLineDash([]);
      if (s.dots !== false) { ctx.fillStyle = s.color; for (const [x, y] of s.points) { ctx.beginPath(); ctx.arc(X(x), Y(y), 2.6, 0, Math.PI * 2); ctx.fill(); } }
    }
  }
  function barChart(canvas, { categories, series, yMax, yFmt = (v) => v, xFmt = (v) => v }) {
    const { ctx, w, h } = setupCanvas(canvas);
    const pad = { l: 44, r: 8, t: 10, b: 26 };
    const n = categories.length, gw = (w - pad.l - pad.r) / n, bw = gw / (series.length + 0.6);
    const ymax = yMax || Math.max(1, ...series.flatMap((s) => s.values));
    const Y = (v) => h - pad.b - v / ymax * (h - pad.t - pad.b);
    ctx.font = "10.5px IBM Plex Mono, monospace"; ctx.fillStyle = C.muted; ctx.strokeStyle = C.line;
    for (let i = 0; i <= 4; i++) { const v = ymax * i / 4, y = Y(v); ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke(); ctx.textAlign = "right"; ctx.fillText(yFmt(v), pad.l - 6, y + 3); }
    categories.forEach((c, i) => { if (i % Math.ceil(n / 8) === 0) { ctx.textAlign = "center"; ctx.fillText(xFmt(c), pad.l + gw * (i + 0.5), h - pad.b + 16); } });
    series.forEach((s, si) => { ctx.fillStyle = s.color; s.values.forEach((v, i) => { const x = pad.l + gw * i + bw * (si + 0.3); ctx.fillRect(x, Y(v), bw, h - pad.b - Y(v)); }); });
  }
  const legend = (el, items) => { el.innerHTML = items.map((i) => `<span><i style="background:${i.color}"></i>${esc(i.name)}</span>`).join(""); };

  // =====================================================================
  // Status / chips / system
  // =====================================================================
  async function refreshStatus() {
    try { STATUS = await api("/api/status"); } catch (e) { return; }
    const s = STATUS;
    const modeChip = $("#chip-mode");
    modeChip.textContent = s.mode === "resnet" ? `mode ResNet-50 checkpoints` : `mode surrogate nodes (stand-in)`;
    modeChip.className = "chip " + (s.mode === "resnet" ? "on" : "warn");
    $("#chip-nodes").textContent = `nodes ${s.nodes.length} · ${s.nodes.reduce((a, n) => a + n.n_families, 0) ? s.nodes.map((n) => n.n_families).join("+") : "—"} families`;
    $("#chip-families").textContent = `families ${s.n_families} known · ${s.zero_day_families.length} unseen`;
    const m = s.meta || {};
    $("#chip-meta").textContent = m.accuracy != null ? `meta-learner ${s.stacker} · ${pct(m.accuracy)} test acc` : `meta-learner not fitted`;
    $("#chip-meta").className = "chip " + (m.accuracy != null ? "on" : "warn");
    const r = s.resources || {};
    $("#chip-res").textContent = `cpu ${r.cpu_percent != null ? r.cpu_percent.toFixed(0) + "%" : "—"} · ram ${r.rss_mb != null ? r.rss_mb.toFixed(0) + " MB" : "—"}`;
    $("#not-ready").hidden = !!s.ready;
    $("#watch-toggle").checked = !!s.watch_enabled;
    $("#watch-dir").textContent = s.config.watch.dir;
    renderNodeCards(s);
    renderSystem(s);
    renderRecent(s.recent_scans || []);
    if (!MAP.built) buildMap(s);
    if (s.federation_running && !fedPollTimer) startFedPolling();
  }

  function renderNodeCards(s) {
    const el = $("#node-cards");
    const busy = s.federation_progress && s.federation_progress.node;
    el.innerHTML = s.nodes.map((n, i) => `
      <div class="node">
        <div class="dot ${busy === n.name ? "busy" : ""}" style="background:${NODE_COLORS[i]};box-shadow:0 0 10px ${NODE_COLORS[i]}"></div>
        <div><div class="name">${esc(n.name)} <span class="muted small">· ${esc(n.label)}</span></div>
          <div class="fams">${n.families.map(pretty).join(" · ")}</div>
          <div class="wbar" title="accuracy-aware weight"><i style="width:${(n.weight || 0) * 100 * 2}%"></i></div></div>
        <div class="acc"><b>${n.val_accuracy ? pct(n.val_accuracy, 0) : "—"}</b><div class="muted small">w=${n.weight != null ? n.weight.toFixed(3) : "—"}</div></div>
      </div>`).join("");
  }

  function renderSystem(s) {
    const cfg = s.config;
    $("#sys-models").innerHTML = `<div class="mf">
      ${cfg.nodes.map((n) => `<div class="row"><span class="${n.checkpoint_exists ? "ok" : "miss"}">${n.checkpoint_exists ? "●" : "○"}</span><div><b>${esc(n.name)}</b> ${n.checkpoint_exists ? "checkpoint loaded" : "checkpoint missing → surrogate node"}<div class="path">${esc(n.checkpoint)}</div></div></div>`).join("")}
      <div class="row"><span class="${cfg.stacker.exists ? "ok" : "miss"}">${cfg.stacker.exists ? "●" : "○"}</span><div><b>meta-learner</b> ${cfg.stacker.exists ? "stacker.joblib loaded" : `${esc(cfg.stacker.kind)} fitted by the server on the validation split`}<div class="path">${esc(cfg.stacker.path)}</div></div></div>
      <div class="row"><span class="${cfg.training_images_dir_exists ? "ok" : "miss"}">${cfg.training_images_dir_exists ? "●" : "○"}</span><div><b>training images</b><div class="path">${esc(cfg.training_images_dir)}</div></div></div>
      <div class="row"><span class="${cfg.zero_day_dir_exists ? "ok" : "miss"}">${cfg.zero_day_dir_exists ? "●" : "○"}</span><div><b>unseen families</b><div class="path">${esc(cfg.zero_day_dir)}</div></div></div>
      <div class="row"><span class="ok">●</span><div><b>config</b><div class="path">${esc(cfg.config_file)}</div></div></div></div>`;
    const r = s.resources || {};
    $("#sys-uptime").textContent = r.uptime_s != null ? `up ${Math.floor(r.uptime_s / 60)} min` : "";
    const g = (name, val, max, fmt) => `<div class="gauge">${name}<b>${fmt}</b><div class="g"><i style="width:${Math.min(100, val / max * 100)}%"></i></div></div>`;
    $("#gauges").innerHTML = g("process CPU", r.cpu_percent || 0, 100 * (r.cpu_count || 1), `${(r.cpu_percent || 0).toFixed(0)}% of ${r.cpu_count || "?"} cores`)
      + g("process RSS", r.rss_mb || 0, 2048, `${(r.rss_mb || 0).toFixed(0)} MB`)
      + g("system CPU", r.system_cpu_percent || 0, 100, `${(r.system_cpu_percent || 0).toFixed(0)}%`)
      + g("system RAM", r.system_ram_percent || 0, 100, `${(r.system_ram_percent || 0).toFixed(0)}%`)
      + `<div class="gauge">threads<b>${r.threads ?? "—"}</b></div>`;
    $("#node-table").innerHTML = `<tr><th>node</th><th>kind</th><th>families</th><th>params</th><th>FP32</th><th>INT8</th><th>local acc</th><th>detail</th></tr>` +
      s.nodes.map((n) => `<tr><td>${esc(n.name)}</td><td>${esc(n.kind)}</td><td>${n.n_families}</td><td>${(n.info.params || 0).toLocaleString()}</td><td>${fmtBytes(n.info.bytes_fp32)}</td><td>${fmtBytes(n.info.bytes_int8)}</td><td>${n.val_accuracy ? pct(n.val_accuracy) : "—"}</td><td>${esc(n.info.backbone || n.info.feature_extractor || "")} ${n.info.classifier ? "· " + esc(n.info.classifier) : ""}</td></tr>`).join("");
    const v = s.versions || {};
    $("#sys-env").innerHTML = Object.entries({ python: v.python, numpy: v.numpy, "scikit-learn": v.sklearn, torch: v.torch || "not installed", torchvision: v.torchvision || "—", cuda: v.cuda ? "yes" : "no (CPU)", "image cache": s.cache ? `${s.cache.known} known + ${s.cache.unseen} unseen @ ${s.cache.px}px` : "missing", "open-set threshold": s.threshold != null ? s.threshold.toFixed(3) + (typeof s.config.open_set.threshold === "string" ? " (auto)" : "") : "—", "stacker": s.stacker })
      .map(([k, val]) => `<span>${esc(k)}</span><span>${esc(val)}</span>`).join("");
  }

  function renderRecent(list) {
    $("#recent-table").innerHTML = `<tr><th>time</th><th>file</th><th>source</th><th>size</th><th>verdict</th><th>conf</th></tr>` +
      (list.length ? list.map((r) => `<tr><td>${new Date(r.t * 1000).toLocaleTimeString()}</td><td>${esc(r.name)}</td><td>${esc(r.source)}</td><td>${fmtBytes(r.size)}</td><td class="${r.unknown ? "unk" : "ok"}">${r.unknown ? "UNKNOWN · possible zero-day" : pretty(r.family)}</td><td>${pct(r.confidence)}</td></tr>`).join("")
        : `<tr><td colspan="6" class="muted">No scans yet.</td></tr>`);
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
    SAMPLES = await api("/api/samples");
    const sel = $("#sample-family");
    const fams = [...new Set(SAMPLES.known.map((s) => s.family))];
    const zd = [...new Set(SAMPLES.zero_day.map((s) => s.family))];
    sel.innerHTML = `<optgroup label="Known families (28)">${fams.map((f) => `<option value="known:${esc(f)}">${esc(pretty(f))}</option>`).join("")}</optgroup>` +
      `<optgroup label="Unseen families (zero-day set)">${zd.map((f) => `<option value="zero_day:${esc(f)}">${esc(pretty(f))} ✦</option>`).join("")}</optgroup>`;
    sel.onchange = renderStrip; renderStrip();
  }
  function renderStrip() {
    const [grp, fam] = $("#sample-family").value.split(":");
    const items = (SAMPLES[grp] || []).filter((s) => s.family === fam).slice(0, 6);
    $("#sample-strip").innerHTML = items.map((s) => `<button type="button" title="${esc(s.file)}" data-path="${esc(s.path)}"><img src="/api/samples/file/${encodeURIComponent(s.path).replace(/%2F/g, "/")}" alt="${esc(pretty(s.family))} sample" loading="lazy"></button>`).join("");
    $$("#sample-strip button").forEach((b) => b.onclick = () => scanSample(b.dataset.path));
  }
  async function scanSample(path) {
    setScanning(true);
    try { renderScan(await postJSON("/api/scan_sample", { path })); } catch (e) { alert(e.message); } finally { setScanning(false); }
  }
  async function scanFile(file) {
    setScanning(true);
    const fd = new FormData(); fd.append("file", file);
    try { renderScan(await api("/api/scan", { method: "POST", body: fd })); } catch (e) { alert(e.message); } finally { setScanning(false); }
  }
  function setScanning(on) { $("#dropzone").classList.toggle("over", on); $("#choose-btn").disabled = on; }
  function setupDropzone() {
    const dz = $("#dropzone"), inp = $("#file-input");
    $("#choose-btn").onclick = (e) => { e.stopPropagation(); inp.click(); };
    dz.onclick = () => inp.click();
    dz.onkeydown = (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); inp.click(); } };
    inp.onchange = () => { if (inp.files[0]) scanFile(inp.files[0]); inp.value = ""; };
    ["dragenter", "dragover"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("over"); }));
    ["dragleave", "drop"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("over"); }));
    dz.addEventListener("drop", (e) => { const f = e.dataTransfer.files[0]; if (f) scanFile(f); });
    document.addEventListener("dragover", (e) => e.preventDefault()); document.addEventListener("drop", (e) => e.preventDefault());
  }

  // =====================================================================
  // Scan rendering: byte loom, hex, meta, map packets, verdict
  // =====================================================================
  async function renderScan(res) {
    const f = res.file, g = res.geometry;
    $("#loom-geom").textContent = f.is_image ? `image ${g.raw_width}×${g.raw_height} → 225×225` : `${f.size_bytes.toLocaleString()} B → ${g.raw_width}×${g.raw_height} px → 225×225`;
    // hex ticker
    const hv = $("#hex-view"); hv.innerHTML = "";
    res.hex.forEach((r, i) => { const d = document.createElement("div"); d.className = "row"; d.style.animationDelay = `${i * 45}ms`; d.innerHTML = `<span class="off">${r.offset}</span>  ${esc(r.hex.padEnd(47))}  <span class="asc">${esc(r.ascii)}</span>`; hv.appendChild(d); });
    // file meta
    const pe = f.pe;
    $("#file-meta").innerHTML = `
      <div class="full"><span class="k">file</span> <span class="v">${esc(f.name)}</span></div>
      <div><span class="k">size</span> <span class="v">${fmtBytes(f.size_bytes)}</span></div>
      <div><span class="k">input</span> <span class="v">${esc(g.input)}</span></div>
      <div class="full"><span class="k">sha256</span> <span class="v" title="${esc(f.sha256)}">${esc(f.sha256)}</span></div>
      <div class="full"><span class="k">entropy</span> <span class="v">${f.entropy.toFixed(3)} / 8 bits ${f.entropy > 7.2 ? "· packed / encrypted?" : f.entropy > 6.5 ? "· compressed regions" : ""}</span><div class="entropy"><i id="entropy-bar"></i></div></div>
      ${pe ? `<div class="full"><span class="k">PE header</span> <span class="v">${esc(pe.format)} ${esc(pe.pe_type || "")} · ${esc(pe.machine || "")} · ${pe.sections ?? "?"} sections · ${esc(pe.subsystem || "")} ${pe.entry_point ? "· EP " + pe.entry_point : ""}${pe.timestamp ? " · " + new Date(pe.timestamp * 1000).toISOString().slice(0, 10) : ""}</span>
        ${pe.section_table && pe.section_table.length ? `<table class="pe-table"><tr><th>section</th><th>raw</th><th>virt</th><th>flags</th><th>H</th></tr>${pe.section_table.slice(0, 8).map((s) => `<tr><td>${esc(s.name)}</td><td>${fmtBytes(s.raw_size)}</td><td>${fmtBytes(s.virtual_size)}</td><td>${esc(s.flags)} ${esc(s.kind)}</td><td>${s.entropy ?? "—"}</td></tr>`).join("")}</table>` : ""}</div>` : ""}`;
    requestAnimationFrame(() => { $("#entropy-bar").style.width = `${f.entropy / 8 * 100}%`; });
    // loom animation
    await weave(res.image_png_b64);
    // packets on the map
    const v = res.verdict;
    if (v.status !== "ok") { $("#verdict-empty").hidden = false; $("#verdict-empty").textContent = v.message; $("#verdict").hidden = true; return; }
    try { await animateScanPackets(res); } catch (e) { console.warn("map animation", e); }
    renderVerdict(res);
    refreshStatus();
  }

  function weave(b64) {
    return new Promise((resolve) => {
      const cv = $("#loom-canvas"), ctx = cv.getContext("2d");
      const img = new Image();
      img.onload = () => {
        ctx.fillStyle = "#05080f"; ctx.fillRect(0, 0, 225, 225);
        if (REDUCED) { ctx.drawImage(img, 0, 0, 225, 225); return resolve(); }
        const rows = 225, dur = 1100, t0 = performance.now();
        let done = false;
        const finish = () => { if (done) return; done = true; ctx.drawImage(img, 0, 0, 225, 225); resolve(); };
        const guard = setTimeout(finish, dur + 400);
        const step = (t) => {
          if (done) return;
          const p = Math.min(1, (t - t0) / dur), y = Math.floor(p * rows);
          ctx.drawImage(img, 0, 0, 225, y || 1, 0, 0, 225, y || 1);
          ctx.fillStyle = "rgba(82,203,227,.9)"; if (p < 1) ctx.fillRect(0, y, 225, 2);
          if (p < 1) requestAnimationFrame(step); else { clearTimeout(guard); finish(); }
        };
        requestAnimationFrame(step);
      };
      img.src = "data:image/png;base64," + b64;
    });
  }

  // ---------------------------------------------------------------- map
  const MAP = { built: false, pos: {}, nodes: [] };
  const svgEl = (tag, attrs = {}, text) => { const e = document.createElementNS("http://www.w3.org/2000/svg", tag); for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v); if (text != null) e.textContent = text; return e; };
  function buildMap(s) {
    const svg = $("#fedmap"); svg.innerHTML = "";
    const box = (x, y, w, h, cls, title, sub, id) => {
      const g = svgEl("g", { id });
      g.appendChild(svgEl("rect", { x, y, width: w, height: h, class: `box ${cls}`, rx: 12 }));
      g.appendChild(svgEl("text", { x: x + 14, y: y + 24, class: "t" }, title));
      (Array.isArray(sub) ? sub : [sub]).forEach((line, i) => g.appendChild(svgEl("text", { x: x + 14, y: y + 44 + i * 14, class: "s" }, line)));
      svg.appendChild(g); return { cx: x + w / 2, cy: y + h / 2, x, y, w, h };
    };
    MAP.pos.endpoint = box(30, 100, 190, 100, "endpoint", "Endpoint agent", ["file → bytes → image", "no raw file leaves here"], "m-endpoint");
    MAP.pos.server = box(430, 90, 240, 120, "server", "Aggregation server", [`${s.stacker === "rf" ? "random-forest" : "logistic"} meta-learner`, "accuracy-aware weights", "open-set threshold"], "m-server");
    MAP.nodes = s.nodes.map((n, i) => { const p = box(830, 12 + i * 70, 250, 58, "node", `${n.name} · ${n.kind === "resnet" ? "ResNet-50" : "surrogate"}`, n.label, `m-${n.name}`); return { ...p, name: n.name }; });
    const link = (a, b) => svg.insertBefore(svgEl("path", { d: `M${a.x + a.w},${a.cy} C${a.x + a.w + 60},${a.cy} ${b.x - 60},${b.cy} ${b.x},${b.cy}`, class: "link" }), svg.firstChild);
    link(MAP.pos.endpoint, MAP.pos.server); MAP.nodes.forEach((n) => link(MAP.pos.server, n));
    MAP.built = true;
  }
  function movePacket(from, to, { cls = "", label = "", dur = 650 } = {}) {
    return new Promise((resolve) => {
      const svg = $("#fedmap");
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
        const cx = x0 + (x1 - x0) * e, cy = y0 + (y1 - y0) * e + Math.sin(p * Math.PI) * -18;
        c.setAttribute("cx", cx); c.setAttribute("cy", cy); t.setAttribute("x", cx + 9); t.setAttribute("y", cy - 8);
        if (p < 1) requestAnimationFrame(step); else { clearTimeout(guard); finish(); }
      };
      requestAnimationFrame(step);
    });
  }
  const glow = (id, on) => { const r = $(`#${id} rect`); if (r) r.classList.toggle("active", on); };
  async function animateScanPackets(res) {
    const { endpoint, server } = MAP.pos;
    $("#map-caption").textContent = "scan: image → server → nodes → votes → verdict";
    glow("m-endpoint", true);
    await movePacket(endpoint, server, { cls: "img", label: fmtBytes(res.payload.image_bytes_sent) });
    glow("m-endpoint", false); glow("m-server", true);
    await Promise.all(MAP.nodes.map((n, i) => sleep(i * 90).then(() => movePacket(server, n, { label: "image", dur: 520 }))));
    MAP.nodes.forEach((n) => glow(`m-${n.name}`, true));
    await sleep(150);
    await Promise.all(MAP.nodes.map((n, i) => { const v = res.verdict.node_votes[i]; return sleep(i * 70).then(() => movePacket({ x: n.x - n.w, w: n.w, cx: n.cx, cy: n.cy }, { x: server.x + server.w, cx: server.cx, cy: server.cy }, { label: `p_${i + 1} · ${v.reply_bytes} B`, dur: 520 })); }));
    MAP.nodes.forEach((n) => glow(`m-${n.name}`, false));
    await sleep(120); glow("m-server", false);
    $("#map-caption").textContent = `verdict in ${res.timings_ms.total.toFixed(0)} ms · ${res.verdict.node_votes.length} node replies (${fmtBytes(res.payload.node_reply_bytes)})`;
  }

  function renderVerdict(res) {
    const v = res.verdict, unk = v.open_set.unknown;
    $("#verdict-empty").hidden = true; $("#verdict").hidden = false;
    $("#verdict-time").textContent = new Date().toLocaleTimeString();
    const ring = $("#ring-fg"); ring.style.stroke = unk ? C.red : C.amber;
    requestAnimationFrame(() => { ring.style.strokeDashoffset = 326.7 * (1 - v.confidence); });
    $("#ring-txt").textContent = pct(v.confidence, 0);
    const badge = $("#verdict-badge"); badge.textContent = unk ? "unknown family · possible zero-day" : "known malware family"; badge.className = "badge " + (unk ? "unknown" : "");
    $("#verdict-family").textContent = unk ? `nearest: ${v.family_pretty}` : v.family_pretty;
    $("#verdict-owner").textContent = `confidence ${v.confidence.toFixed(3)} ${unk ? "<" : "≥"} threshold ${v.open_set.threshold.toFixed(3)} (${v.open_set.method}) · entropy ${v.normalised_entropy} · owner ${v.owner_node || "—"}${v.owner_confidence != null ? " @ " + pct(v.owner_confidence, 0) : ""}`;
    $("#top5").innerHTML = v.top5.map((t) => `<div class="row"><div><div class="lbl">${esc(t.pretty)}</div><div class="bar"><i data-w="${t.prob * 100}"></i></div></div><div class="pct">${pct(t.prob)}</div></div>`).join("");
    $("#votes").innerHTML = v.node_votes.map((n, i) => `<div class="vote ${n.node === v.owner_node ? "owner" : ""}"><div class="n">${esc(n.node)}</div><div class="f">${esc(pretty(n.top_family))}</div><div class="w">p=${n.top_prob.toFixed(2)} · w=${n.weight.toFixed(2)} · ${n.ms.toFixed(1)} ms</div><div class="p" data-w="${n.top_prob * 100}" style="background:${NODE_COLORS[i]}"></div></div>`).join("");
    requestAnimationFrame(() => { $$("#top5 .bar i, .vote .p").forEach((b) => b.style.width = b.dataset.w + "%"); });
    const w = v.meta.weights.map((x) => x.toFixed(2));
    $("#meta-line").textContent = `ŷ = meta([ ${w.map((x, i) => `${x}·p${i + 1}`).join(" ; ")} ]) → ${v.meta.input_dim}-dim stacked input → ${v.top5[0].pretty} · ${v.meta.ms} ms`;
    const t = res.timings_ms;
    $("#timings").innerHTML = `<span>convert <b>${t.convert} ms</b></span><span>nodes <b>${t.nodes} ms</b></span><span>meta <b>${t.meta} ms</b></span><span>total <b>${t.total} ms</b></span><span>uplink <b>${fmtBytes(res.payload.image_bytes_sent)}</b></span>`;
  }

  // =====================================================================
  // Federation
  // =====================================================================
  async function runFederation() {
    const rounds = parseInt($("#fed-rounds").value, 10), epochs = parseInt($("#fed-epochs").value, 10);
    const r = await postJSON("/api/federation/run", { rounds, epochs });
    if (r.started) { $("#fed-run").disabled = true; $("#fed-progress").hidden = false; startFedPolling(); }
  }
  function startFedPolling() { if (fedPollTimer) return; fedPollTimer = setInterval(pollFederation, 700); pollFederation(); }
  async function pollFederation() {
    let st; try { st = await api("/api/federation/state"); } catch (e) { return; }
    renderFederation(st);
    if (!st.running) { clearInterval(fedPollTimer); fedPollTimer = null; $("#fed-run").disabled = false; $("#fed-progress").hidden = true; $("#fed-status").textContent = st.history.length ? `done · ${st.history.length} round${st.history.length > 1 ? "s" : ""}` : "idle"; refreshStatus(); }
    else { const p = st.progress || {}; $("#fed-status").textContent = "running"; $("#fed-bar").style.setProperty("--p", `${(((p.round || 1) - 1) + (p.node === "server" ? .9 : .4)) / (p.rounds || 1) * 100}%`); $("#fed-progress-txt").textContent = `round ${p.round}/${p.rounds} · ${p.node} · ${p.phase}`; renderNodeCards({ nodes: STATUS ? STATUS.nodes : [], federation_progress: p }); }
  }
  function renderFederation(st) {
    const H = st.history || [];
    const T = Math.max(H.length, 2);
    if (H.length && H[0].replay) {
      const h = H[0];
      barChart($("#chart-acc"), { categories: ["meta", ...h.nodes.map((n) => n.node)], series: [{ color: C.amber, values: [h.meta_accuracy, ...h.nodes.map((n) => n.local_acc)] }], yMax: 1, yFmt: (v) => pct(v, 0) });
      legend($("#legend-acc"), [{ name: "accuracy (ResNet checkpoints; node fine-tuning ran offline on GPU)", color: C.amber }]);
    } else {
      lineChart($("#chart-acc"), { xMin: 1, xMax: T, yMin: 0, yMax: 1, xFmt: (v) => Math.round(v), yFmt: (v) => pct(v, 0), xTicks: Math.min(T - 1, 10),
        series: [{ color: C.amber, width: 2.5, points: H.map((h) => [h.round, h.meta_accuracy]) }, { color: C.violet, dash: [3, 3], dots: false, points: H.map((h) => [h.round, h.meta_f1]) },
          ...(H[0] ? H[0].nodes.map((n, i) => ({ color: NODE_COLORS[i], width: 1.2, dots: false, points: H.map((h) => [h.round, h.nodes[i].local_acc]) })) : [])] });
      legend($("#legend-acc"), [{ name: "meta-learner test accuracy", color: C.amber }, { name: "meta-learner macro-F1", color: C.violet }, ...(H[0] ? H[0].nodes.map((n, i) => ({ name: `${n.node} local accuracy`, color: NODE_COLORS[i] })) : [])]);
    }
    if (H.length) {
      const last = H[H.length - 1], T2 = last.replay ? last.rounds_configured : H.length;
      const pts = last.replay ? [[0, 0], [T2, last.cumulative_bytes]] : [[0, 1], ...H.map((h) => [h.round, h.cumulative_bytes])];
      const fa = last.replay ? [[0, 0], [T2, last.cumulative_fedavg_bytes]] : [[0, 1], ...H.map((h) => [h.round, h.cumulative_fedavg_bytes])];
      const ref = last.payload.reference_thesis, K = last.payload.per_node.length;
      const refP = [[0, 1], [T2, ref.proposed_per_node * K * T2]], refF = [[0, 1], [T2, ref.fedavg_per_node * K * T2]];
      const ymax = Math.pow(10, Math.ceil(Math.log10(Math.max(...fa.map((p) => p[1]), ...refF.map((p) => p[1])))));
      lineChart($("#chart-traffic"), { xMin: 0, xMax: T2, yMin: 1e3, yMax: ymax, yLog: true, xFmt: (v) => Math.round(v), yFmt: fmtBytes, xTicks: Math.min(T2, 10), yTicks: 5,
        series: [{ color: C.cyan, width: 2.5, points: pts }, { color: C.red, width: 2, points: fa }, { color: C.cyan, dash: [5, 4], dots: false, points: refP }, { color: C.red, dash: [5, 4], dots: false, points: refF }] });
      legend($("#legend-traffic"), [{ name: "proposed, measured (INT8 weights + predictions)", color: C.cyan }, { name: "FedAvg FP32, same nodes (up + down)", color: C.red }, { name: `dashed: thesis reference ${ref.backbone} · proposed ${fmtBytes(ref.proposed_total)} vs FedAvg ${fmtBytes(ref.fedavg_total)} (K=${K}, T=${T2})`, color: C.muted }]);
      renderPayloadTable(last.payload, T2);
    } else if (st.payload_now) { renderPayloadTable(st.payload_now, STATUS ? STATUS.config.federation.rounds : 20); }
  }
  function renderPayloadTable(p, T) {
    const rows = p.per_node.map((n) => `<tr><td>${esc(n.node)}</td><td>${n.params.toLocaleString()}</td><td>${fmtBytes(n.weights_fp32)}</td><td>${fmtBytes(n.weights_bytes)}</td><td>${n.n_val} × ${STATUS ? STATUS.n_families : 28} × 4 B = ${fmtBytes(n.predictions_bytes)}</td><td>${fmtBytes(n.upload_bytes)}</td><td>${fmtBytes(n.download_bytes)}</td><td>${fmtBytes(2 * n.weights_fp32)}</td></tr>`).join("");
    const r = p.reference_thesis;
    $("#payload-table").innerHTML = `<tr><th>node</th><th>params</th><th>weights FP32</th><th>weights INT8 (sent)</th><th>prediction matrix (sent)</th><th>upload / round</th><th>download / round</th><th>FedAvg FP32 / round</th></tr>${rows}
      <tr class="hl"><td>all nodes · per round</td><td></td><td></td><td></td><td></td><td>${fmtBytes(p.proposed_total_bytes)}</td><td>0</td><td>${fmtBytes(p.fedavg_total_bytes)} (${p.reduction_x ? p.reduction_x.toFixed(1) + "× more" : ""})</td></tr>
      <tr class="hl"><td>all nodes · T = ${T} rounds</td><td></td><td></td><td></td><td></td><td>${fmtBytes(p.proposed_total_bytes * T)}</td><td>0</td><td>${fmtBytes(p.fedavg_total_bytes * T)}</td></tr>
      <tr class="ref"><td>thesis reference: ${esc(r.backbone)} nodes, 700 × 28 predictions</td><td>${r.params.toLocaleString()}</td><td>${fmtBytes(r.fp32_bytes)}</td><td>${fmtBytes(r.int8_bytes)}</td><td>${fmtBytes(700 * 28 * 4)}</td><td>${fmtBytes(r.proposed_per_node)} / node</td><td>0</td><td>${fmtBytes(r.fedavg_per_node)} / node → ${fmtBytes(r.fedavg_total)} total vs ${fmtBytes(r.proposed_total)} (Table 3.12)</td></tr>`;
    $("#payload-note").textContent = STATUS && STATUS.mode === "resnet" ? "measured from the loaded checkpoints" : "measured on the surrogate nodes · thesis row is the ResNet-50 configuration";
  }

  // =====================================================================
  // Zero-day
  // =====================================================================
  async function runZeroDay() {
    $("#zd-run").disabled = true; $("#zd-summary").textContent = "Evaluating unseen families…";
    try { renderZeroDay(await postJSON("/api/zeroday/run", {})); } catch (e) { $("#zd-summary").textContent = e.message; } finally { $("#zd-run").disabled = false; }
  }
  function renderZeroDay(z) {
    if (z.status !== "ok") { $("#zd-summary").textContent = z.status === "not_ready" ? "Run the federated training first." : "Not run yet."; return; }
    $("#zd-summary").innerHTML = `threshold <b>${z.threshold.toFixed(3)}</b> (${esc(z.method)}) · unseen images flagged unknown <b>${pct(z.unseen.flagged_unknown)}</b> of ${z.unseen.n} · known test images wrongly flagged <b>${pct(z.known.false_unknown_rate)}</b> of ${z.known.n}<br>mean confidence known <b>${z.known.mean_confidence.toFixed(3)}</b> vs unseen <b>${z.unseen.mean_confidence.toFixed(3)}</b> · best balanced accuracy <b>${pct(z.suggested_balanced_accuracy)}</b> at ${z.suggested_threshold.toFixed(2)} · ${z.ms} ms`;
    $("#zd-families").innerHTML = z.families.map((f) => `<div class="zdf"><div class="h"><span>${esc(pretty(f.family))} <span class="muted small">n=${f.n}</span></span><b>${pct(f.flagged_unknown, 0)} flagged</b></div><div class="bar"><i data-w="${f.flagged_unknown * 100}"></i></div><div class="near">mean confidence ${f.mean_confidence.toFixed(3)} · nearest known: ${f.nearest_known.map((n) => `${esc(n.pretty)} (${pct(n.share, 0)})`).join(", ") || "—"}</div></div>`).join("");
    requestAnimationFrame(() => $$(".zdf .bar i").forEach((b) => b.style.width = b.dataset.w + "%"));
    const b = z.histogram.bins, cats = b.slice(0, -1).map((x, i) => `${x.toFixed(2)}`);
    const norm = (a) => { const s = a.reduce((x, y) => x + y, 0) || 1; return a.map((v) => v / s); };
    barChart($("#chart-hist"), { categories: cats, series: [{ color: C.green, values: norm(z.histogram.known) }, { color: C.red, values: norm(z.histogram.unseen) }], yFmt: (v) => pct(v, 0), xFmt: (v) => v });
    legend($("#legend-hist"), [{ name: "known families (test split)", color: C.green }, { name: "unseen families", color: C.red }, { name: `x = confidence · threshold ${z.threshold.toFixed(2)}`, color: C.muted }]);
    lineChart($("#chart-sweep"), { xMin: 0, xMax: 1, yMin: 0, yMax: 1, xFmt: (v) => v.toFixed(1), yFmt: (v) => pct(v, 0), xTicks: 5,
      series: [{ color: C.amber, width: 2.5, points: z.sweep.map((s) => [s.threshold, s.balanced]) }, { color: C.red, dash: [3, 3], dots: false, points: z.sweep.map((s) => [s.threshold, s.unseen_flagged]) }, { color: C.green, dash: [3, 3], dots: false, points: z.sweep.map((s) => [s.threshold, s.known_accepted]) }],
      vlines: [{ x: z.threshold, color: C.cyan, label: "in use" }] });
    legend($("#legend-sweep"), [{ name: "balanced accuracy", color: C.amber }, { name: "unseen flagged", color: C.red }, { name: "known accepted", color: C.green }, { name: "threshold in use", color: C.cyan }]);
  }

  // =====================================================================
  // boot
  // =====================================================================
  async function boot() {
    setupDropzone();
    $("#fed-run").onclick = runFederation;
    $("#zd-run").onclick = runZeroDay;
    $("#watch-toggle").onchange = async (e) => { await postJSON("/api/watch", { enabled: e.target.checked }); refreshStatus(); };
    await refreshStatus();
    if (STATUS) { $("#fed-rounds").value = STATUS.config.federation.rounds; $("#fed-epochs").value = STATUS.config.federation.local_epochs; }
    loadSamples().then(() => { if (new URLSearchParams(location.search).has("autoscan")) { const b = $("#sample-strip button"); if (b) b.click(); } }).catch(() => {});
    api("/api/federation/state").then(renderFederation).catch(() => {});
    api("/api/zeroday/state").then((z) => { if (z.status === "ok") renderZeroDay(z); }).catch(() => {});
    pollEvents(); setInterval(pollEvents, 900); setInterval(refreshStatus, 2500);
    window.addEventListener("resize", () => { api("/api/federation/state").then(renderFederation).catch(() => {}); api("/api/zeroday/state").then((z) => { if (z.status === "ok") renderZeroDay(z); }).catch(() => {}); });
  }
  document.addEventListener("DOMContentLoaded", boot);
})();
