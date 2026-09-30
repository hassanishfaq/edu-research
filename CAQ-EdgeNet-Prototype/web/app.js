/* CAQ-EdgeNet demo — dashboard logic (no build step, no framework) */
(() => {
  "use strict";
  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => Array.from(el.querySelectorAll(s));
  const REDUCED = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const C = { accent: "#a3e635", sky: "#5cc8f2", amber: "#f5b342", rose: "#ff6b8a", violet: "#b9a6ff", green: "#5ed49a", red: "#ff6767", muted: "#8fa89c", line: "#26362f", text: "#ecf5f0" };
  const ACT_COLOR = { local: C.accent, partial: C.amber, edge: C.sky };
  const CLASS_COLOR = ["#5ed49a", "#ff6b8a", "#f5b342", "#5cc8f2", "#b9a6ff", "#f9a8d4", "#fde68a", "#c4b5fd"];

  async function api(path, opts = {}) {
    const r = await fetch(path, opts);
    if (!r.ok) throw new Error(`${path}: ${r.status} ${await r.text()}`);
    return r.json();
  }
  const postJSON = (path, body) => api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const fmtBytes = (b) => b == null ? "—" : b < 1e3 ? `${b} B` : b < 1e6 ? `${(b / 1e3).toFixed(1)} KB` : `${(b / 1e6).toFixed(2)} MB`;
  const pct = (x, d = 1) => x == null ? "—" : `${(x * 100).toFixed(d)}%`;
  const f3 = (x) => x == null ? "—" : x.toFixed(3);
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  let STATUS = null, lastEventId = 0, dePoll = null, ctPoll = null, flPoll = null, SAMPLES = null, SEL = null, INSPECT_TIMER = null;

  // =====================================================================
  // Canvas charts
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
  function lineChart(canvas, { series, xMin, xMax, yMin = 0, yMax = 1, yFmt = (v) => v, xFmt = (v) => v, xTicks = 5, yTicks = 4 }) {
    const { ctx, w, h } = setupCanvas(canvas);
    const pad = { l: 54, r: 12, t: 12, b: 28 };
    const X = (x) => pad.l + (x - xMin) / (xMax - xMin || 1) * (w - pad.l - pad.r);
    const Y = (y) => h - pad.b - (y - yMin) / (yMax - yMin || 1) * (h - pad.t - pad.b);
    ctx.font = "10.5px IBM Plex Mono, monospace"; ctx.fillStyle = C.muted; ctx.strokeStyle = C.line; ctx.lineWidth = 1;
    for (let i = 0; i <= yTicks; i++) { const v = yMin + (yMax - yMin) * i / yTicks, y = Y(v); ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke(); ctx.textAlign = "right"; ctx.fillText(yFmt(v), pad.l - 6, y + 3); }
    for (let i = 0; i <= xTicks; i++) { const v = xMin + (xMax - xMin) * i / xTicks; ctx.textAlign = "center"; ctx.fillText(xFmt(v), X(v), h - pad.b + 16); }
    for (const s of series) {
      if (!s.points.length) continue;
      ctx.strokeStyle = s.color; ctx.lineWidth = s.width || 2; ctx.setLineDash(s.dash || []); ctx.beginPath();
      s.points.forEach(([x, y], i) => i ? ctx.lineTo(X(x), Y(y)) : ctx.moveTo(X(x), Y(y))); ctx.stroke(); ctx.setLineDash([]);
      if (s.dots) { ctx.fillStyle = s.color; for (const [x, y] of s.points) { ctx.beginPath(); ctx.arc(X(x), Y(y), 2.4, 0, Math.PI * 2); ctx.fill(); } }
    }
  }
  function barChart(canvas, { categories, series, yMax, yFmt = (v) => v, xFmt = (v) => v, labelEvery = 1 }) {
    const { ctx, w, h } = setupCanvas(canvas);
    const pad = { l: 44, r: 8, t: 10, b: 26 };
    const n = categories.length, gw = (w - pad.l - pad.r) / n, bw = gw / (series.length + 0.6);
    const ymax = yMax || Math.max(1e-9, ...series.flatMap((s) => s.values));
    const Y = (v) => h - pad.b - v / ymax * (h - pad.t - pad.b);
    ctx.font = "10.5px IBM Plex Mono, monospace"; ctx.fillStyle = C.muted; ctx.strokeStyle = C.line;
    for (let i = 0; i <= 4; i++) { const v = ymax * i / 4, y = Y(v); ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke(); ctx.textAlign = "right"; ctx.fillText(yFmt(v), pad.l - 6, y + 3); }
    categories.forEach((c, i) => { if (i % labelEvery === 0) { ctx.textAlign = "center"; ctx.fillText(xFmt(c), pad.l + gw * (i + 0.5), h - pad.b + 16); } });
    series.forEach((s, si) => { ctx.fillStyle = s.color; s.values.forEach((v, i) => { const x = pad.l + gw * i + bw * (si + 0.3); ctx.fillRect(x, Y(v), bw, h - pad.b - Y(v)); }); });
  }
  const legend = (el, items) => { el.innerHTML = items.map((i) => `<span><i style="background:${i.color}"></i>${esc(i.name)}</span>`).join(""); };

  // =====================================================================
  // Status / chips / system
  // =====================================================================
  async function refreshStatus() {
    try { STATUS = await api("/api/status"); } catch (e) { return; }
    const s = STATUS, cfg = s.config;
    $("#chip-data").textContent = s.dataset ? `data ${s.dataset.source} · ${s.dataset.n.toLocaleString()} flows` : "data —";
    $("#chip-data").className = "chip " + (s.dataset && s.dataset.source.startsWith("cic") ? "on" : "warn");
    const m = s.metrics_summary;
    $("#chip-det").textContent = m ? `detector ${pct(m.accuracy)} acc · F1 ${pct(m.f1_macro)}` : "detector untrained";
    $("#chip-det").className = "chip " + (m ? "on" : "warn");
    const q = (s.deploy_summary || []).find((r) => r.precision === "int8_hybrid");
    $("#chip-q").textContent = q ? `export INT8 hybrid ${fmtBytes(q.size_bytes)} · ${q.device_ms.toFixed(1)} ms on device` : "export —";
    $("#chip-pol").textContent = s.policy_trained ? `controller DQN ${s.versions_model.policy}` : "controller untrained (rule-based fallback)";
    $("#chip-pol").className = "chip " + (s.policy_trained ? "on" : "warn");
    const r = s.resources || {};
    $("#chip-res").textContent = `cpu ${r.cpu_percent != null ? r.cpu_percent.toFixed(0) + "%" : "—"} · ram ${r.rss_mb != null ? r.rss_mb.toFixed(0) + " MB" : "—"}`;
    $("#not-ready").hidden = !!s.trained;
    $("#fleet-toggle").checked = !!s.fleet_running;
    $("#in-source").textContent = s.dataset ? s.dataset.source : "";
    renderSystem(s);
    if (s.training && !dePoll) startDetectorPolling();
    if (s.ctl_training && !ctPoll) startControllerPolling();
    if (s.fleet_running && !flPoll) startFleetPolling();
    if (!ARCH.built) buildArch(s);
    if (s.cost_model) renderPathsTable(s.cost_model);
  }

  function renderSystem(s) {
    const cfg = s.config;
    $("#sys-models").innerHTML = `<div class="mf">
      <div class="row"><span class="${cfg.cic_iot_dir_exists ? "ok" : "miss"}">${cfg.cic_iot_dir_exists ? "●" : "○"}</span><div><b>CIC-IoT-2023 flows</b> ${cfg.cic_iot_dir_exists ? "CSV folder found" : "folder missing → synthetic stand-in flows (same 46 features, Table 5.2 counts)"}<div class="path">${esc(cfg.cic_iot_dir)}</div></div></div>
      <div class="row"><span class="${cfg.context_csv ? "ok" : "miss"}">${cfg.context_csv ? "●" : "○"}</span><div><b>device context (MDF-style)</b> ${cfg.context_csv ? "CSV" : "built-in simulator"}<div class="path">${esc(cfg.context_csv || "caqedge/data.py · DeviceContextSim")}</div></div></div>
      <div class="row"><span class="${cfg.checkpoint_exists ? "ok" : "miss"}">${cfg.checkpoint_exists ? "●" : "○"}</span><div><b>detector checkpoint</b> ${cfg.checkpoint_exists ? s.versions_model.detector : "not trained yet"}<div class="path">${esc(cfg.checkpoint)}</div></div></div>
      <div class="row"><span class="${cfg.policy_exists ? "ok" : "miss"}">${cfg.policy_exists ? "●" : "○"}</span><div><b>controller policy</b> ${cfg.policy_exists ? s.versions_model.policy : "not trained yet"}<div class="path">${esc(cfg.policy)}</div></div></div>
      <div class="row"><span class="ok">●</span><div><b>config</b><div class="path">${esc(cfg.config_file)}</div></div></div></div>`;
    const r = s.resources || {};
    $("#sys-uptime").textContent = r.uptime_s != null ? `up ${Math.floor(r.uptime_s / 60)} min` : "";
    const g = (name, val, max, fmt) => `<div class="gauge">${name}<b>${fmt}</b><div class="g"><i style="width:${Math.min(100, val / max * 100)}%"></i></div></div>`;
    $("#gauges").innerHTML = g("process CPU", r.cpu_percent || 0, 100 * (r.cpu_count || 1), `${(r.cpu_percent || 0).toFixed(0)}% of ${r.cpu_count || "?"} cores`) + g("process RSS", r.rss_mb || 0, 2048, `${(r.rss_mb || 0).toFixed(0)} MB`)
      + g("system CPU", r.system_cpu_percent || 0, 100, `${(r.system_cpu_percent || 0).toFixed(0)}%`) + g("system RAM", r.system_ram_percent || 0, 100, `${(r.system_ram_percent || 0).toFixed(0)}%`) + `<div class="gauge">threads<b>${r.threads ?? "—"}</b></div>`;
    $("#layer-table").innerHTML = `<tr><th>layer</th><th>role</th><th>in → out</th><th>params</th></tr>` + (s.model.layers || []).map((l) => `<tr><td>${esc(l.layer)}</td><td style="text-align:left;white-space:normal">${esc(l.role)}</td><td>${l.in} → ${l.out}</td><td>${l.params.toLocaleString()}</td></tr>`).join("")
      + `<tr class="hl"><td>total</td><td></td><td></td><td>${(s.model.n_params || 0).toLocaleString()}</td></tr>` + (s.policy_params ? `<tr class="ref"><td>controller Q-network</td><td>7 features → 3 actions</td><td></td><td>${s.policy_params.toLocaleString()}</td></tr>` : "");
    const v = s.versions || {}, cm = s.cost_model || {};
    $("#sys-env").innerHTML = Object.entries({ python: v.python, numpy: v.numpy, "scikit-learn": v.sklearn, torch: v.torch, cuda: v.cuda ? "yes" : "no (CPU)", "detector version": s.versions_model.detector, "quantised export": s.versions_model.export, "policy version": s.versions_model.policy,
      "device profile": s.config.device_profile.name, "compute factor": cm.compute_factor ? `${cm.compute_factor}× (anchored to ${s.config.device_profile.reference_fp32_ms} ms FP32)` : "—", "reward weights λ": cm.lambdas ? cm.lambdas.join(" / ") : "—", "telemetry records": s.telemetry_count })
      .map(([k, val]) => `<span>${esc(k)}</span><span>${esc(val)}</span>`).join("");
  }

  // =====================================================================
  // Event log
  // =====================================================================
  async function pollEvents() {
    try {
      const { events } = await api(`/api/events?since=${lastEventId}`);
      if (events.length) {
        const log = $("#log");
        for (const e of events) { lastEventId = e.id; const d = document.createElement("div"); d.className = `e ${e.level}`; d.innerHTML = `<span>${new Date(e.t * 1000).toLocaleTimeString()}</span><span class="lv">${esc(e.level)}</span><span>${esc(e.msg)}</span>`; log.appendChild(d); }
        while (log.children.length > 400) log.removeChild(log.firstChild);
        log.scrollTop = log.scrollHeight;
        $("#log-count").textContent = `${lastEventId} events`;
      }
    } catch (e) { /* server restarting */ }
  }

  // =====================================================================
  // Inspect
  // =====================================================================
  async function loadSamples() {
    SAMPLES = await api("/api/samples");
    const sel = $("#in-class");
    sel.innerHTML = SAMPLES.classes.map((c) => `<option value="${esc(c)}">${esc(c)}</option>`).join("");
    sel.onchange = renderSampleStrip; renderSampleStrip();
  }
  function renderSampleStrip() {
    const c = $("#in-class").value;
    const items = SAMPLES.samples.filter((s) => s.class === c);
    $("#in-samples").innerHTML = items.map((s) => `<button type="button" data-i="${s.index}" class="${SEL === s.index ? "on" : ""}">#${s.index}</button>`).join("");
    $$("#in-samples button").forEach((b) => b.onclick = () => { SEL = parseInt(b.dataset.i, 10); renderSampleStrip(); runInspect(); });
  }
  function stateFromSliders() {
    return { battery: parseInt($("#st-battery").value, 10) / 100, cpu: parseInt($("#st-cpu").value, 10) / 100, rtt_ms: parseInt($("#st-rtt").value, 10), bandwidth_mbps: parseInt($("#st-bw").value, 10) / 10, edge_load: parseInt($("#st-edge").value, 10) / 100, link_up: $("#st-link").checked };
  }
  function setupSliders() {
    const upd = () => { $("#st-battery-v").textContent = `${$("#st-battery").value}%`; $("#st-cpu-v").textContent = `${$("#st-cpu").value}%`; $("#st-rtt-v").textContent = `${$("#st-rtt").value} ms`; $("#st-bw-v").textContent = `${(parseInt($("#st-bw").value, 10) / 10).toFixed(1)} Mbps`; $("#st-edge-v").textContent = `${$("#st-edge").value}%`; };
    ["#st-battery", "#st-cpu", "#st-rtt", "#st-bw", "#st-edge"].forEach((id) => $(id).oninput = () => { upd(); if (SEL != null) { clearTimeout(INSPECT_TIMER); INSPECT_TIMER = setTimeout(runInspect, 120); } });
    $("#st-link").onchange = () => { if (SEL != null) runInspect(); };
    upd();
  }
  async function runInspect() {
    if (SEL == null) return;
    let res; try { res = await postJSON("/api/inspect", { index: SEL, state: stateFromSliders() }); } catch (e) { return; }
    if (res.status !== "ok") { $("#verdict-empty").hidden = false; $("#verdict-empty").textContent = res.message; return; }
    renderInspect(res);
  }
  function strip(el, values, colorFn) {
    const n = values.length;
    if (el.children.length !== n) el.innerHTML = values.map(() => "<i></i>").join("");
    const max = Math.max(1e-6, ...values.map(Math.abs));
    Array.from(el.children).forEach((c, i) => { c.style.background = colorFn(values[i], max); c.title = `[${i}] ${values[i].toFixed(4)}`; });
  }
  function renderInspect(res) {
    const e = res.encoder, v = res.verdict, ctl = res.controller;
    $("#tr-dims").textContent = `h_d ∈ ℝ^${e.h_flow} (46 flow features) · h_s ∈ ℝ^${e.h_ctx} (7 context features) → z ∈ ℝ^${e.d_z} → gate a(S_t) → ẑ (${e.bits} bit)`;
    $("#tr-caption").textContent = `S_t = (${res.state.battery.toFixed(2)}, ${res.state.cpu.toFixed(2)}, ${res.state.latency_norm.toFixed(2)})`;
    strip($("#strip-gate"), e.gate, (x) => `rgba(163,230,53,${0.15 + 0.85 * x})`);
    strip($("#strip-z"), e.z, (x, m) => `rgba(92,200,242,${0.1 + 0.9 * Math.abs(x) / m})`);
    strip($("#strip-zq"), e.z_hat, (x, m) => `rgba(245,179,66,${0.1 + 0.9 * Math.abs(x) / m})`);
    const levels = new Set(e.z_hat.map((x) => x.toFixed(6))).size;
    $("#tr-quant").textContent = `Eq. (5.4): s = (max z′ − min z′)/(2^${e.bits} − 1) = ${e.scale != null ? e.scale.toExponential(2) : "—"} · ẑ takes ${levels} distinct levels · ${e.z_bytes} B on the wire (partial) vs ${e.feature_bytes} B of raw features (edge)`;
    $("#busy-bar").style.width = `${v.busy_forecast * 100}%`; $("#busy-now").style.left = `${res.state.cpu * 100}%`; $("#busy-v").textContent = `${v.busy_forecast.toFixed(3)} (now ${res.state.cpu.toFixed(2)})`;
    const bdColors = { collect: C.muted, inference: C.accent, contention: C.rose, encoder: C.amber, serialise: C.violet, transmit: C.sky, rtt: "#3b82f6", edge: "#22d3ee" };
    $("#paths").innerHTML = res.paths.map((p) => { const tot = Object.values(p.breakdown).reduce((a, b) => a + b, 0) || 1;
      return `<div class="path ${p.action === ctl.chosen ? "chosen" : ""} ${p.action === ctl.oracle ? "oracle" : ""} ${p.feasible ? "" : "infeasible"}"><div class="n ${p.action}">${p.action}</div><div class="v">${p.latency_ms.toFixed(1)} ms · ${(p.energy_j * 1000).toFixed(1)} mJ · ${p.bytes} B<br>${esc(p.prediction)} <b>${pct(p.confidence, 0)}</b><br>r = <b>${p.reward.toFixed(3)}</b>${p.feasible ? "" : " · masked"}</div><div class="bd" title="${esc(Object.entries(p.breakdown).map(([k, x]) => k + " " + x.toFixed(1) + " ms").join(" · "))}">${Object.entries(p.breakdown).map(([k, x]) => `<i style="width:${x / tot * 100}%;background:${bdColors[k] || C.line}"></i>`).join("")}</div></div>`; }).join("");
    // verdict
    $("#verdict-empty").hidden = true; $("#verdict").hidden = false;
    $("#in-time").textContent = new Date().toLocaleTimeString();
    const ring = $("#ring-fg"); ring.style.stroke = v.malicious ? C.rose : C.green;
    requestAnimationFrame(() => { ring.style.strokeDashoffset = 326.7 * (1 - v.confidence); });
    $("#ring-txt").textContent = pct(v.confidence, 0);
    const badge = $("#verdict-badge"); badge.textContent = v.malicious ? "malicious traffic" : "benign traffic"; badge.className = "badge " + (v.malicious ? "" : "benign");
    $("#verdict-class").textContent = v.class;
    $("#verdict-sub").textContent = `${res.label ? "true class " + res.label + (res.label === v.class ? " ✓" : " ✗") + " · " : ""}INT8 on device · edge FP32 says ${v.edge_class} (${pct(v.edge_confidence, 0)})`;
    const top = Object.entries(v.probs).sort((a, b) => b[1] - a[1]).slice(0, 5);
    $("#top5").innerHTML = top.map(([c, p]) => `<div class="row"><div><div class="lbl">${esc(c)}</div><div class="bar"><i data-w="${p * 100}"></i></div></div><div class="pct">${pct(p)}</div></div>`).join("");
    const q = ctl.q;
    const qmax = q ? Math.max(...q) : 1, qmin = q ? Math.min(...q) : 0;
    $("#qvals").innerHTML = ["local", "partial", "edge"].map((a, i) => `<div class="qv ${a === ctl.chosen ? "chosen" : ""} ${ctl.mask[i] ? "" : "masked"}"><div class="n">${a}</div><div class="w">${q ? "Q = " + q[i].toFixed(3) : "rule"}${ctl.mask[i] ? "" : " · masked"}</div><div class="p" style="width:${q ? ((q[i] - qmin) / (qmax - qmin || 1) * 100) : 0}%;background:${ACT_COLOR[a]}"></div></div>`).join("");
    requestAnimationFrame(() => { $$("#top5 .bar i").forEach((b) => b.style.width = b.dataset.w + "%"); });
    $("#decision-line").textContent = `a_t = argmax_a Q(S_t, a) over the feasible set → ${ctl.chosen} (${ctl.source}) · clairvoyant oracle: ${ctl.oracle} · busy forecast ${v.busy_forecast.toFixed(2)} feeds the policy`;
    $("#timings").innerHTML = `<span>device INT8 <b>${res.timings_ms.device_int8} ms</b> here</span><span>edge FP32 <b>${res.timings_ms.edge_fp32} ms</b> here</span><span>ẑ <b>${e.z_bytes} B</b></span><span>features <b>${e.feature_bytes} B</b></span><span>versions <b>${esc(res.versions.detector)} · ${esc(res.versions.export)} · ${esc(res.versions.policy)}</b></span>`;
    animateArch(ctl.chosen, res);
  }

  // ---------------------------------------------------------------- architecture map
  const ARCH = { built: false, pos: {} };
  const svgEl = (tag, attrs = {}, text) => { const el = document.createElementNS("http://www.w3.org/2000/svg", tag); for (const [k, val] of Object.entries(attrs)) el.setAttribute(k, val); if (text != null) el.textContent = text; return el; };
  function box(svg, x, y, w, h, cls, title, sub, id) {
    const g = svgEl("g", { id });
    g.appendChild(svgEl("rect", { x, y, width: w, height: h, class: `box ${cls}`, rx: 10 }));
    g.appendChild(svgEl("text", { x: x + 10, y: y + 19, class: "t" }, title));
    (Array.isArray(sub) ? sub : [sub]).forEach((line, i) => g.appendChild(svgEl("text", { x: x + 10, y: y + 36 + i * 13, class: "s" }, line)));
    svg.appendChild(g);
    return { x, y, w, h, cx: x + w / 2, cy: y + h / 2 };
  }
  function buildArch(s) {
    const svg = $("#archmap"); svg.innerHTML = "";
    svg.appendChild(svgEl("rect", { x: 12, y: 10, width: 700, height: 200, class: "box dev", rx: 14, opacity: .5 }));
    svg.appendChild(svgEl("text", { x: 24, y: 30, class: "t" }, "Device agent"));
    const mods = [["Collectors", ["static · flow", "context sampler"]], ["Encoders", ["h_d · h_s", "Eq. (5.2)"]], ["Fusion", ["z = σ(W_f[h_s;h_d])", "Eq. (5.3)"]], ["Cond. attention", ["a(S_t) gate", "FusionCAB"]], ["QAE", ["ẑ = round(z′/s)·s", `b = ${s.model.cfg ? s.model.cfg.bits : 8} bits`]], ["Controller", ["a_t ∈ {local,", "partial, edge}"]], ["Local classifier", ["INT8 heads", "busy-state head"]]];
    ARCH.pos.mods = mods.map(([t, sub], i) => box(svg, 24 + i * 98, 44, 90, 70, "mod", t, sub, `am-${i}`));
    ARCH.pos.mods.forEach((m, i) => { if (i) svg.insertBefore(svgEl("path", { d: `M${m.x - 8},${m.cy} L${m.x},${m.cy}`, class: "link" }), svg.firstChild); });
    box(svg, 24, 128, 290, 70, "mod", "Device state sampler", ["S_t = (battery, CPU, latency) · Eq. (5.5)", "telemetry: state, action, latency, energy, verdict"], "am-state");
    box(svg, 330, 128, 370, 70, "mod", "Transport", ["MQTT / gRPC over TLS · per-device credentials", "ẑ (partial) · 53 features (edge) · verdicts · telemetry"], "am-tx");
    ARCH.pos.edge = box(svg, 760, 24, 400, 90, "edge", "Edge server", ["FP32 model on full feature vectors (edge path)", "FP32 heads on ẑ (partial path)", "telemetry & reward log → Eq. (5.7)"], "am-edge");
    ARCH.pos.mgmt = box(svg, 760, 130, 400, 70, "mgmt", "Management plane", ["training · quantisation-aware export · policy trainer (shadow mode)", "model registry: detector · export · policy versions"], "am-mgmt");
    svg.insertBefore(svgEl("path", { d: `M712,120 C740,120 740,69 760,69`, class: "link", id: "al-edge" }), svg.firstChild);
    svg.insertBefore(svgEl("path", { d: `M712,165 L760,165`, class: "link" }), svg.firstChild);
    svg.appendChild(svgEl("text", { x: 24, y: 232, class: "s", id: "arch-note" }, "pick a flow window: the chosen path lights up and its packet crosses the transport"));
    ARCH.built = true;
  }
  function movePacket(svg, from, to, { cls = "", label = "", dur = 600 } = {}) {
    return new Promise((resolve) => {
      if (REDUCED) return resolve();
      const c = svgEl("circle", { r: 6, class: `pk ${cls}` }), t = svgEl("text", { class: "pk-l" }, label);
      svg.appendChild(c); svg.appendChild(t);
      const t0 = performance.now(); let done = false;
      const finish = () => { if (done) return; done = true; c.remove(); t.remove(); resolve(); };
      const guard = setTimeout(finish, dur + 300);
      const step = (now) => {
        if (done) return;
        const p = Math.min(1, (now - t0) / dur), e = p < .5 ? 2 * p * p : -1 + (4 - 2 * p) * p;
        const cx = from.x + (to.x - from.x) * e, cy = from.y + (to.y - from.y) * e + Math.sin(p * Math.PI) * -16;
        c.setAttribute("cx", cx); c.setAttribute("cy", cy); t.setAttribute("x", cx + 9); t.setAttribute("y", cy - 8);
        if (p < 1) requestAnimationFrame(step); else { clearTimeout(guard); finish(); }
      };
      requestAnimationFrame(step);
    });
  }
  async function animateArch(action, res) {
    const svg = $("#archmap"); const M = ARCH.pos.mods;
    $$("#archmap .box").forEach((b) => b.classList.remove("active"));
    const upto = action === "edge" ? 1 : 5;
    for (let i = 0; i <= Math.min(upto, M.length - 1); i++) { $(`#am-${i} rect`).classList.add("active"); await sleep(REDUCED ? 0 : 90); }
    if (action === "local") { $("#am-6 rect").classList.add("active"); $("#arch-note").textContent = `local: everything stays on the device · ${res.paths[0].latency_ms.toFixed(0)} ms · nothing transmitted`; return; }
    const from = action === "edge" ? { x: M[0].cx, y: M[0].cy + 40 } : { x: M[4].cx, y: M[4].cy + 40 };
    $("#am-tx rect").classList.add("active");
    await movePacket(svg, from, { x: ARCH.pos.edge.x, y: ARCH.pos.edge.cy }, { cls: action === "edge" ? "" : "z", label: action === "edge" ? `${res.encoder.feature_bytes} B features` : `ẑ ${res.encoder.z_bytes} B` });
    $("#am-edge rect").classList.add("active");
    await movePacket(svg, { x: ARCH.pos.edge.x, y: ARCH.pos.edge.cy + 20 }, { x: M[5].cx, y: M[5].cy + 30 }, { label: "verdict", dur: 450 });
    const p = res.paths[action === "edge" ? 2 : 1];
    $("#arch-note").textContent = `${action}: ${action === "edge" ? "raw features leave the device, the edge runs the full FP32 model" : "only the quantised embedding leaves the device, the edge runs the heads"} · ${p.latency_ms.toFixed(0)} ms · ${(p.energy_j * 1000).toFixed(0)} mJ`;
  }

  // =====================================================================
  // Fleet
  // =====================================================================
  const FLEET = { built: false, n: 0, pos: [] };
  function buildFleet(n) {
    const svg = $("#fleetmap"); svg.innerHTML = ""; FLEET.pos = [];
    FLEET.edge = { x: 880, y: 100, w: 260, h: 130 };
    const eb = svgEl("g", { id: "fl-edge" });
    eb.appendChild(svgEl("rect", { ...FLEET.edge, width: FLEET.edge.w, height: FLEET.edge.h, class: "edgebox", rx: 14 }));
    eb.appendChild(svgEl("text", { x: FLEET.edge.x + 14, y: FLEET.edge.y + 26, class: "t" }, "Edge server"));
    eb.appendChild(svgEl("text", { x: FLEET.edge.x + 14, y: FLEET.edge.y + 46, class: "s", id: "fl-edge-s1" }, "FP32 model · heads on ẑ"));
    eb.appendChild(svgEl("text", { x: FLEET.edge.x + 14, y: FLEET.edge.y + 62, class: "s", id: "fl-edge-s2" }, "telemetry & reward log"));
    eb.appendChild(svgEl("text", { x: FLEET.edge.x + 14, y: FLEET.edge.y + 78, class: "s", id: "fl-edge-s3" }, ""));
    eb.appendChild(svgEl("text", { x: FLEET.edge.x + 14, y: FLEET.edge.y + 110, class: "s on", id: "fl-edge-s4" }, ""));
    svg.appendChild(eb);
    const cols = Math.ceil(n / 2);
    for (let i = 0; i < n; i++) {
      const x = 40 + (i % cols) * (760 / cols), y = 30 + Math.floor(i / cols) * 150, w = Math.min(230, 760 / cols - 20), h = 118;
      const g = svgEl("g", { id: `fl-dev-${i}` });
      g.appendChild(svgEl("rect", { x, y, width: w, height: h, class: "devnode", rx: 12, id: `fl-rect-${i}` }));
      g.appendChild(svgEl("text", { x: x + 12, y: y + 22, class: "t" }, `Device ${i + 1}`));
      g.appendChild(svgEl("text", { x: x + 12, y: y + 40, class: "s", id: `fl-l1-${i}` }, ""));
      g.appendChild(svgEl("text", { x: x + 12, y: y + 56, class: "s", id: `fl-l2-${i}` }, ""));
      g.appendChild(svgEl("text", { x: x + 12, y: y + 72, class: "s on", id: `fl-l3-${i}` }, ""));
      g.appendChild(svgEl("rect", { x: x + 12, y: y + 86, width: w - 24, height: 6, class: "gauge-bg", rx: 3 }));
      g.appendChild(svgEl("rect", { x: x + 12, y: y + 86, width: 0, height: 6, class: "gauge-bat", rx: 3, id: `fl-bat-${i}` }));
      g.appendChild(svgEl("rect", { x: x + 12, y: y + 98, width: w - 24, height: 6, class: "gauge-bg", rx: 3 }));
      g.appendChild(svgEl("rect", { x: x + 12, y: y + 98, width: 0, height: 6, class: "gauge-cpu", rx: 3, id: `fl-cpu-${i}` }));
      svg.appendChild(g);
      svg.insertBefore(svgEl("path", { d: `M${x + w},${y + h / 2} C${x + w + 60},${y + h / 2} ${FLEET.edge.x - 60},${FLEET.edge.y + FLEET.edge.h / 2} ${FLEET.edge.x},${FLEET.edge.y + FLEET.edge.h / 2}`, class: "link" }), svg.firstChild);
      FLEET.pos.push({ x, y, w, h, cx: x + w, cy: y + h / 2 });
    }
    FLEET.built = true; FLEET.n = n;
  }
  const seenTel = new Set();
  function startFleetPolling() { if (flPoll) return; flPoll = setInterval(pollFleet, 700); pollFleet(); }
  async function pollFleet() {
    let st; try { st = await api("/api/fleet/state"); } catch (e) { return; }
    if (!st.running) { clearInterval(flPoll); flPoll = null; }
    renderFleet(st);
  }
  function renderFleet(st) {
    const devs = st.devices || [];
    if (!devs.length) return;
    if (!FLEET.built || FLEET.n !== devs.length) buildFleet(devs.length);
    const svg = $("#fleetmap");
    devs.forEach((d, i) => {
      const p = FLEET.pos[i];
      $(`#fl-rect-${i}`).setAttribute("class", `devnode ${d.link_up ? d.last_action : "down"}`);
      $(`#fl-l1-${i}`).textContent = `bat ${pct(d.battery, 0)}${d.charging ? " ⚡" : ""} · cpu ${pct(d.cpu, 0)} · fc ${d.busy_forecast.toFixed(2)}`;
      $(`#fl-l2-${i}`).textContent = d.link_up ? `rtt ${d.rtt_ms} ms · ${d.bandwidth_mbps} Mbps${d.congested ? " · congested" : ""}` : "link down → local only";
      $(`#fl-l3-${i}`).textContent = `${d.last_action} · ${d.last_latency_ms} ms · ${(d.last_energy_j * 1000).toFixed(0)} mJ · ${d.label}${d.correct ? "" : " ✗"}`;
      $(`#fl-bat-${i}`).setAttribute("width", (p.w - 24) * d.battery); $(`#fl-cpu-${i}`).setAttribute("width", (p.w - 24) * d.cpu);
    });
    for (const r of (st.recent || []).slice(0, 8).reverse()) {
      const key = `${r.t}-${r.device}`; if (seenTel.has(key)) continue; seenTel.add(key);
      const p = FLEET.pos[r.device]; if (!p) continue;
      if (r.action === "edge") movePacket(svg, { x: p.cx, y: p.cy }, { x: FLEET.edge.x, y: FLEET.edge.y + FLEET.edge.h / 2 }, { label: `${r.bytes} B`, dur: 550 });
      else if (r.action === "partial") movePacket(svg, { x: p.cx, y: p.cy }, { x: FLEET.edge.x, y: FLEET.edge.y + FLEET.edge.h / 2 }, { cls: "z", label: `ẑ ${r.bytes} B`, dur: 550 });
      else { const rect = $(`#fl-rect-${r.device}`); rect.classList.add("active"); setTimeout(() => rect.classList.remove("active"), 350); }
    }
    if (seenTel.size > 2000) seenTel.clear();
    const s = st.stats || {};
    const first = devs[0];
    $("#fl-edge-s3").textContent = `edge load ${pct(first.edge_load, 0)} (shared)`;
    $("#fl-edge-s4").textContent = s.requests ? `${s.requests} requests · ${fmtBytes(s.bytes)} uplink · detection ${pct(s.detection_accuracy)}` : "";
    $("#fleet-req").textContent = s.requests ? `${s.requests} requests · ${s.fallbacks} fallbacks · window ${s.window}` : "";
    $("#device-table").innerHTML = `<tr><th>device</th><th>battery</th><th>cpu</th><th>forecast</th><th>rtt</th><th>link</th><th>action</th><th>shadow</th><th>latency</th><th>energy</th><th>Q local/partial/edge</th></tr>` + devs.map((d) => `<tr><td>${d.id + 1}</td><td>${pct(d.battery, 0)}</td><td>${pct(d.cpu, 0)}</td><td>${d.busy_forecast.toFixed(2)}</td><td>${d.rtt_ms}</td><td class="${d.link_up ? "ok" : "bad"}">${d.link_up ? "up" : "down"}</td><td class="${d.last_action === "local" ? "ok" : d.last_action === "partial" ? "amber" : "sky"}">${d.last_action}</td><td>${d.shadow_action}</td><td>${d.last_latency_ms} ms</td><td>${(d.last_energy_j * 1000).toFixed(0)} mJ</td><td>${d.q ? d.q.map((x) => x.toFixed(2)).join(" / ") : "—"}</td></tr>`).join("");
    const kp = (name, val, sub, cls = "") => `<div class="kpi">${name}<b class="${cls}">${val}</b><small>${sub}</small></div>`;
    const better = (a, b, lowerBetter = true) => a == null || b == null ? "" : ((lowerBetter ? a <= b : a >= b) ? "good" : "bad");
    $("#fleet-kpis").innerHTML = kp("mean reward · controller", s.reward != null ? s.reward.toFixed(3) : "—", `shadow rule-based ${s.shadow_reward != null ? s.shadow_reward.toFixed(3) : "—"}`, better(s.reward, s.shadow_reward, false))
      + kp("latency", s.latency_ms != null ? s.latency_ms.toFixed(0) + " ms" : "—", `shadow ${s.shadow_latency_ms != null ? s.shadow_latency_ms.toFixed(0) + " ms" : "—"}`, better(s.latency_ms, s.shadow_latency_ms))
      + kp("energy / request", s.energy_j != null ? (s.energy_j * 1000).toFixed(0) + " mJ" : "—", `shadow ${s.shadow_energy_j != null ? (s.shadow_energy_j * 1000).toFixed(0) + " mJ" : "—"}`, better(s.energy_j, s.shadow_energy_j))
      + kp("policy", s.policy_active ? "DQN acting" : "rule-based acting", s.policy_active ? "rule-based in shadow mode" : "train the controller to promote the DQN");
    $("#actbars").innerHTML = ["local", "partial", "edge"].map((a, i) => `<div class="r"><span>${a}</span><div class="bar"><i style="width:${(s.action_share ? s.action_share[i] : 0) * 100}%;background:${ACT_COLOR[a]}"></i></div><span class="v">${pct(s.action_share ? s.action_share[i] : 0, 0)}</span></div>`).join("");
    const collapse = s.action_share && Math.max(...s.action_share) > 0.97 && s.window > 60;
    $("#drift-kpis").innerHTML = kp("busy-forecast MAE", s.busy_mae != null ? s.busy_mae.toFixed(3) : "—", `test-split MAE ${STATUS && STATUS.metrics_summary ? STATUS.metrics_summary.busy_mae.toFixed(3) : "—"} · rising ⇒ device drift`, s.busy_mae != null && STATUS && STATUS.metrics_summary && s.busy_mae > 2 * STATUS.metrics_summary.busy_mae ? "bad" : "")
      + kp("action distribution", collapse ? "collapsed" : "mixed", collapse ? "one action only ⇒ masked link or degenerate policy" : "no collapse to a single action", collapse ? "bad" : "good");
    if (s.benign_conf_hist) { barChart($("#chart-benign"), { categories: Array.from({ length: 10 }, (_, i) => (i / 10).toFixed(1)), series: [{ color: C.green, values: s.benign_conf_hist }], labelEvery: 2 }); legend($("#legend-benign"), [{ name: "confidence on benign windows (shift ⇒ traffic drift ⇒ retrain)", color: C.green }]); }
    $("#tel-table").innerHTML = `<tr><th>time</th><th>device</th><th>state</th><th>action</th><th>oracle</th><th>shadow</th><th>latency</th><th>energy</th><th>bytes</th><th>reward</th><th>label</th><th>versions</th></tr>` + (st.telemetry || []).map((r) => `<tr><td>${new Date(r.t * 1000).toLocaleTimeString()}</td><td>${r.device + 1}</td><td>b ${pct(r.state.battery, 0)} · c ${pct(r.state.cpu, 0)} · ${r.state.rtt_ms} ms${r.state.link_up ? "" : " · down"}</td><td class="${r.action === "local" ? "ok" : r.action === "partial" ? "amber" : "sky"}">${r.action}</td><td>${r.oracle}</td><td>${r.shadow}</td><td>${r.latency_ms} ms</td><td>${(r.energy_j * 1000).toFixed(0)} mJ</td><td>${r.bytes}</td><td>${r.reward.toFixed(3)}</td><td class="${r.correct ? "" : "bad"}">${esc(r.label)}${r.correct ? "" : " ✗"}</td><td>${esc(r.versions.detector)} · ${esc(r.versions.export)} · ${esc(r.versions.policy)}</td></tr>`).join("");
  }

  // =====================================================================
  // Detector
  // =====================================================================
  async function runDetector() {
    const r = await postJSON("/api/detector/train", { epochs: parseInt($("#de-epochs").value, 10) });
    if (r.started) { $("#de-run").disabled = true; $("#de-progress").hidden = false; startDetectorPolling(); }
  }
  async function runAblation() {
    const r = await postJSON("/api/ablation/run", {});
    if (r.started) { $("#ab-run").disabled = true; $("#ab-note").textContent = "running…"; startDetectorPolling(); }
  }
  function startDetectorPolling() { if (dePoll) return; dePoll = setInterval(pollDetector, 900); pollDetector(); }
  async function pollDetector() {
    let st; try { st = await api("/api/detector/state"); } catch (e) { return; }
    renderDetector(st);
    if (!st.training && !st.ablation_running) { clearInterval(dePoll); dePoll = null; $("#de-run").disabled = false; $("#ab-run").disabled = false; $("#de-progress").hidden = true; $("#de-status").textContent = st.trained ? `trained · ${st.version}` : "idle"; refreshStatus(); }
    else if (st.training) { const p = st.progress || {}; $("#de-status").textContent = "training"; $("#de-bar").style.setProperty("--p", `${p.epoch ? p.epoch / p.epochs * 100 : 5}%`); $("#de-progress-txt").textContent = p.epoch ? `epoch ${p.epoch}/${p.epochs}` : (p.phase || "starting"); }
  }
  function renderDetector(st) {
    const H = st.history || [];
    if (H.length) {
      const E = Math.max(H.length, 2);
      lineChart($("#chart-train"), { xMin: 1, xMax: E, yMin: 0, yMax: 1, xFmt: (v) => Math.round(v), yFmt: (v) => pct(v, 0), xTicks: Math.min(E - 1, 10),
        series: [{ color: C.accent, width: 2.5, dots: true, points: H.map((h) => [h.epoch, h.val_accuracy]) }, { color: C.sky, dash: [3, 3], points: H.map((h) => [h.epoch, h.val_f1]) }, { color: C.rose, points: H.map((h) => [h.epoch, Math.min(1, h.loss)]) }, { color: C.amber, dash: [2, 3], points: H.map((h) => [h.epoch, h.val_busy_mae * 5]) }] });
      legend($("#legend-train"), [{ name: "validation accuracy", color: C.accent }, { name: "validation macro-F1", color: C.sky }, { name: "training loss (CE + λ·MSE)", color: C.rose }, { name: "busy-state MAE × 5", color: C.amber }]);
    }
    const m = st.metrics;
    if (m && m.test) {
      const t = m.test;
      const kp = (n, v, sub) => `<div class="kpi">${n}<b>${v}</b><small>${sub}</small></div>`;
      $("#test-kpis").innerHTML = kp("accuracy", pct(t.accuracy, 2), `n = ${t.n.toLocaleString()} test flows`) + kp("macro-F1", pct(t.f1_macro, 2), `precision ${pct(t.precision_macro)} · recall ${pct(t.recall_macro)}`) + kp("AUC-ROC (ovr)", t.auc != null ? pct(t.auc, 2) : "—", "one-vs-rest") + kp("busy-state MAE", t.busy_mae.toFixed(3), `RMSE ${t.busy_rmse.toFixed(3)} · R² ${t.busy_r2.toFixed(3)}`);
      $("#test-note").textContent = `${m.n_params.toLocaleString()} parameters · test split`;
      const cm = t.confusion, K = m.classes.length;
      $("#class-table").innerHTML = `<tr><th>class</th><th>P</th><th>R</th><th>F1</th><th>n</th>${m.classes.map((c) => `<th title="${esc(c)}">${esc(c.slice(0, 3))}</th>`).join("")}</tr>` + m.classes.map((c, i) => { const pc = t.per_class[i]; const row = cm[i]; const tot = row.reduce((a, b) => a + b, 0) || 1;
        return `<tr><td>${esc(c)}</td><td>${f3(pc.precision)}</td><td>${f3(pc.recall)}</td><td>${f3(pc.f1)}</td><td>${pc.support}</td>${row.map((x, j) => `<td><span class="heat" style="background:rgba(${i === j ? "94,212,154" : "255,107,138"},${0.08 + 0.7 * x / tot})">${x}</span></td>`).join("")}</tr>`; }).join("");
      const bb = m.busy_baselines || [];
      $("#busy-table").innerHTML = `<tr><th>method</th><th>MAE</th><th>RMSE</th><th>MAPE %</th><th>R²</th><th>EVS</th></tr>` + bb.map((r) => `<tr class="${r.proposed ? "hl" : ""}"><td>${esc(r.method)}</td><td>${r.mae.toFixed(4)}</td><td>${r.rmse.toFixed(4)}</td><td>${r.mape.toFixed(1)}</td><td>${r.r2.toFixed(3)}</td><td>${r.evs.toFixed(3)}</td></tr>`).join("");
    }
    const d = st.deploy;
    if (d && d.rows) {
      $("#deploy-table").innerHTML = `<tr><th>precision</th><th>model size</th><th>device part</th><th>size reduction</th><th>measured here (numpy)</th><th>device estimate</th><th>latency reduction</th><th>accuracy</th><th>bits per layer</th></tr>` + d.rows.map((r) => `<tr class="${r.precision === "int8_hybrid" ? "hl" : ""}"><td>${esc(r.label)}</td><td>${fmtBytes(r.size_bytes)}</td><td>${fmtBytes(r.device_bytes)}</td><td>${pct(r.size_reduction)}</td><td>${r.measured_ms.toFixed(3)} ms</td><td>${r.device_ms.toFixed(2)} ms</td><td>${pct(r.latency_reduction)}</td><td>${pct(r.accuracy, 2)}</td><td class="muted">${Object.values(r.bits).join("/")}</td></tr>`).join("")
        + `<tr class="ref"><td>torch state dict</td><td>${fmtBytes(d.torch_bytes_fp32)} FP32 · ${fmtBytes(d.torch_bytes_fp16)} FP16</td><td colspan="7">${esc(d.note)}</td></tr>`;
      $("#deploy-note").textContent = `FP32 base accuracy ${pct(d.base_accuracy, 2)} · device anchored to ${STATUS ? STATUS.config.device_profile.reference_fp32_ms : 12.6} ms FP32 (Table 5.7)`;
      $("#sens-table").innerHTML = `<tr><th>layer</th><th>params</th><th>drop @ 8 bit</th><th>drop @ 4 bit</th><th>hybrid</th><th>why</th></tr>` + d.sensitivity.map((s) => `<tr><td>${esc(s.layer)}</td><td>${s.params.toLocaleString()}</td><td class="${s.drop8_pts > d.threshold_pts ? "bad" : ""}">${s.drop8_pts >= 0 ? "+" : ""}${s.drop8_pts.toFixed(2)} pts</td><td class="${s.drop4_pts > d.threshold_pts ? "bad" : ""}">${s.drop4_pts >= 0 ? "+" : ""}${s.drop4_pts.toFixed(2)} pts</td><td class="${s.hybrid_bits >= 16 ? "sky" : s.hybrid_bits === 4 ? "amber" : "ok"}">${s.hybrid_bits} bit</td><td class="muted" style="text-align:left">${esc(s.reason)}</td></tr>`).join("");
      $("#sens-note").textContent = `threshold ${d.threshold_pts} pts · protected: ${d.protected.join(", ")}`;
    }
    const ab = st.ablation;
    if (ab && ab.rows && ab.rows.length) {
      $("#ab-table").innerHTML = `<tr><th>variant</th><th>F1</th><th>accuracy</th><th>INT8 accuracy</th><th>busy MAE</th><th>offloading accuracy</th><th>latency</th><th>energy</th><th>cost</th><th>params</th></tr>` + ab.rows.map((r, i) => `<tr class="${i === 0 ? "hl" : ""}"><td>${esc(r.variant)}</td><td>${pct(r.f1)}</td><td>${pct(r.accuracy)}</td><td>${pct(r.int8_accuracy)}</td><td>${r.busy_mae != null ? r.busy_mae.toFixed(3) : "—"}</td><td>${r.offload_accuracy != null ? pct(r.offload_accuracy) : "train the controller"}</td><td>${r.latency_ms != null ? r.latency_ms.toFixed(0) + " ms" : "—"}</td><td>${r.energy_j != null ? (r.energy_j * 1000).toFixed(0) + " mJ" : "—"}</td><td>${r.cost != null ? r.cost.toFixed(3) : "—"}</td><td>${r.params.toLocaleString()}</td></tr>`).join("");
      $("#ab-note").textContent = ab.status === "running" ? `running · ${ab.current}` : `${ab.epochs} epochs per variant · offloading accuracy = released controller re-evaluated with each variant's forecast and detections`;
    }
  }
  async function loadDataset() {
    let d; try { d = await api("/api/dataset"); } catch (e) { return; }
    $("#dataset-table").innerHTML = `<tr><th>class</th><th>category</th><th>samples</th><th>share</th><th>ratio</th></tr>` + d.classes.map((c) => `<tr><td>${esc(c.name)}</td><td class="muted">${esc(c.group)}</td><td>${c.n.toLocaleString()}</td><td>${pct(c.share)}</td><td>${c.ratio.toFixed(2)}</td></tr>`).join("") + `<tr class="hl"><td>total</td><td>${esc(d.source)}</td><td>${d.n.toLocaleString()}</td><td>100%</td><td>${d.imbalance.toFixed(2)}:1</td></tr><tr class="ref"><td>split</td><td colspan="4">train ${d.split[0].toLocaleString()} · val ${d.split[1].toLocaleString()} · test ${d.split[2].toLocaleString()} · 46 flow + 7 context features</td></tr>`;
  }

  // =====================================================================
  // Controller
  // =====================================================================
  async function runController() {
    const r = await postJSON("/api/controller/train", { episodes: parseInt($("#ct-episodes").value, 10) });
    if (r.started) { $("#ct-run").disabled = true; $("#ct-progress").hidden = false; startControllerPolling(); } else alert(r.reason);
  }
  function startControllerPolling() { if (ctPoll) return; ctPoll = setInterval(pollController, 900); pollController(); }
  async function pollController() {
    let st; try { st = await api("/api/controller/state"); } catch (e) { return; }
    renderController(st);
    if (!st.training) { clearInterval(ctPoll); ctPoll = null; $("#ct-run").disabled = false; $("#ct-progress").hidden = true; $("#ct-status").textContent = st.evaluation ? `trained · ${st.version}` : "idle"; refreshStatus(); }
    else { const p = st.progress || {}; $("#ct-status").textContent = "training"; $("#ct-bar").style.setProperty("--p", `${p.episode ? p.episode / p.episodes * 70 : (p.phase && p.phase.startsWith("training the tabular") ? 78 : p.phase ? 92 : 3)}%`); $("#ct-progress-txt").textContent = p.episode ? `episode ${p.episode}/${p.episodes} · ${p.phase}` : (p.phase || "starting"); }
  }
  function renderController(st) {
    const H = st.history || [];
    if (H.length) {
      const E = Math.max(H.length, 2);
      const ymin = Math.min(...H.map((h) => h.reward)) - 0.05, ymax = Math.max(...H.map((h) => h.reward)) + 0.05;
      lineChart($("#chart-reward"), { xMin: 1, xMax: E, yMin: ymin, yMax: ymax, xFmt: (v) => Math.round(v), yFmt: (v) => v.toFixed(2), xTicks: Math.min(E - 1, 10),
        series: [{ color: C.accent, width: 2.2, points: H.map((h) => [h.episode, h.reward]) }, { color: C.sky, dash: [3, 3], points: H.map((h) => [h.episode, ymin + (ymax - ymin) * h.decision_accuracy]) }, { color: C.muted, dash: [2, 4], points: H.map((h) => [h.episode, ymin + (ymax - ymin) * h.epsilon]) }] });
      legend($("#legend-reward"), [{ name: "mean reward per step", color: C.accent }, { name: "decisions = oracle (scaled)", color: C.sky }, { name: "ε (scaled)", color: C.muted }]);
    }
    if (st.cost_model) { const cm = st.cost_model; $("#ct-cost").innerHTML = Object.entries({ "λ1 / λ2 / λ3": cm.lambdas.join(" / "), "latency ref": `${cm.latency_ref_ms} ms`, "energy ref": `${(cm.energy_ref_j * 1000).toFixed(0)} mJ`, "device INT8 full": `${cm.device_full_int8_ms} ms`, "device INT8 encoder": `${cm.device_encoder_int8_ms} ms`, "edge full (nominal load)": `${cm.edge_full_ms_nominal} ms`, "edge heads (nominal load)": `${cm.edge_heads_ms_nominal} ms`, "measured here FP32": `${cm.measured_ms.fp32_full_ms.toFixed(3)} ms` }).map(([k, v]) => `<span>${esc(k)}</span><span>${esc(v)}</span>`).join(""); }
    const ev = st.evaluation;
    if (ev) {
      $("#policy-table").innerHTML = `<tr><th>method</th><th>decision accuracy</th><th>precision</th><th>recall</th><th>F1</th><th>latency</th><th>energy</th><th>total cost</th><th>throughput</th><th>efficiency</th><th>local / partial / edge</th></tr>` + ev.rows.map((r) => `<tr class="${r.kind === "proposed" ? "hl" : ""}"><td>${esc(r.policy)}</td><td>${pct(r.decision_accuracy)}</td><td>${pct(r.precision)}</td><td>${pct(r.recall)}</td><td>${pct(r.f1)}</td><td>${r.latency_ms.toFixed(1)} ms</td><td>${(r.energy_j * 1000).toFixed(1)} mJ</td><td>${(0.6 * r.latency_ms / (st.cost_model ? st.cost_model.latency_ref_ms : 150) + 0.6 * r.energy_j / (st.cost_model ? st.cost_model.energy_ref_j : 0.15)).toFixed(3)}</td><td>${r.throughput.toFixed(1)} req/s</td><td>${r.efficiency != null ? r.efficiency.toFixed(3) : "—"}</td><td>${r.actions.map((a) => pct(a, 0)).join(" / ")}</td></tr>`).join("")
        + `<tr class="ref"><td>clairvoyant oracle</td><td>100%</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1.000</td><td>mean reward ${ev.oracle_reward.toFixed(3)}</td></tr>`;
      $("#pol-note").textContent = `${ev.episodes} held-out episodes × ${ev.steps} steps · decision accuracy = share of steps whose action equals the oracle's · efficiency = mean reward / oracle reward · total cost = λ2·lat/ref + λ3·E/ref`;
    }
  }
  function renderPathsTable(cm) {
    $("#paths-table").innerHTML = `<tr><th>path</th><th>on the device</th><th>transmitted</th><th>at the edge</th></tr>
      <tr><td class="ok">local</td><td>collect · INT8 model (${cm.device_full_int8_ms} ms)</td><td>nothing (telemetry only)</td><td>—</td></tr>
      <tr><td class="amber">partial</td><td>collect · INT8 encoders + attention + QAE (${cm.device_encoder_int8_ms} ms)</td><td>ẑ: ${cm.z_bytes} B</td><td>FP32 heads on ẑ (${cm.edge_heads_ms_nominal} ms)</td></tr>
      <tr><td class="sky">edge</td><td>collect only</td><td>${cm.feature_bytes} B (53 × FP32)</td><td>full FP32 model (${cm.edge_full_ms_nominal} ms)</td></tr>`;
  }

  // =====================================================================
  // boot
  // =====================================================================
  async function boot() {
    setupSliders();
    $("#de-run").onclick = runDetector; $("#ab-run").onclick = runAblation; $("#ct-run").onclick = runController;
    $("#fleet-toggle").onchange = async (e) => { await postJSON("/api/fleet", { enabled: e.target.checked }); if (e.target.checked) startFleetPolling(); refreshStatus(); };
    await refreshStatus();
    if (STATUS) { $("#de-epochs").value = STATUS.config.training.epochs; $("#ct-episodes").value = STATUS.config.controller.episodes; }
    loadDataset();
    loadSamples().then(() => { if (STATUS && STATUS.trained) { const b = $("#in-samples button"); if (b) b.click(); } }).catch(() => {});
    api("/api/detector/state").then(renderDetector).catch(() => {});
    api("/api/controller/state").then(renderController).catch(() => {});
    api("/api/fleet/state").then(renderFleet).catch(() => {});
    pollEvents(); setInterval(pollEvents, 900); setInterval(refreshStatus, 2500);
    window.addEventListener("resize", () => { api("/api/detector/state").then(renderDetector).catch(() => {}); api("/api/controller/state").then(renderController).catch(() => {}); });
  }
  document.addEventListener("DOMContentLoaded", boot);
})();
