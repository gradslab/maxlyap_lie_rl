(function () {
  "use strict";
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));

  /* math */
  function renderMath() {
    if (!window.katex) return setTimeout(renderMath, 80);
    $$(".eq").forEach(el => katex.render(el.dataset.tex, el, { displayMode: true, throwOnError: false }));
    $$(".m").forEach(el => katex.render(el.dataset.tex, el, { displayMode: false, throwOnError: false }));
  }
  renderMath();

  /* scroll helpers */
  $$("[data-scroll]").forEach(b => b.addEventListener("click", () => $(b.dataset.scroll).scrollIntoView({ behavior: "smooth" })));
  const io = new IntersectionObserver(es => es.forEach(e => {
    if (!e.isIntersecting) return;
    e.target.classList.add("in");
    if (e.target.classList.contains("stats")) countUp(e.target);
    io.unobserve(e.target);
  }), { threshold: 0.15 });
  $$(".section .headline, .card, .step, .stats, .explorer, .dcard, .honest, .play, .ticks li, .lie-demo").forEach(el => { el.classList.add("reveal"); io.observe(el); });

  function countUp(root) {
    $$(".num", root).forEach(el => {
      const to = parseFloat(el.dataset.to), dec = +(el.dataset.dec || 0), t0 = performance.now();
      const tick = t => {
        const k = Math.min(1, (t - t0) / 1400), e = 1 - Math.pow(1 - k, 3);
        el.textContent = (to * e).toFixed(dec);
        if (k < 1) requestAnimationFrame(tick);
      };
      requestAnimationFrame(tick);
    });
  }

  /* bibtex */
  $("#copyBib").addEventListener("click", e => {
    navigator.clipboard.writeText($("#bibtex").textContent).then(() => {
      e.target.textContent = "copied"; setTimeout(() => (e.target.textContent = "copy"), 1400);
    });
  });

  /* toy playground */
  const cv = $("#roa"), ctx = cv.getContext("2d");
  const N = 150, L = 3.6, H = 0.08, MAXIT = 500;
  const A = [0.55, 1.15, -0.85, 0.25], K = 1.7, UMAX = [1.0, 0.7];
  const off = document.createElement("canvas"); off.width = off.height = N;
  const octx = off.getContext("2d"), img = octx.createImageData(N, N);
  let nominal = null, current = null, trajs = [];

  function step(x, y, g, c, s) {
    let ux = -K * x, uy = -K * y;
    ux = Math.max(-UMAX[0], Math.min(UMAX[0], ux));
    uy = Math.max(-UMAX[1], Math.min(UMAX[1], uy));
    const rx = g * (c * ux - s * uy), ry = g * (s * ux + c * uy);
    return [x + H * (A[0] * x + A[1] * y + rx), y + H * (A[2] * x + A[3] * y + ry)];
  }
  function costField(g, phi) {
    const c = Math.cos(phi), s = Math.sin(phi), W = new Float64Array(N * N);
    for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
      let x = -L + (2 * L * (i + 0.5)) / N, y = L - (2 * L * (j + 0.5)) / N, w = 0, k = 0, ok = false;
      for (; k < MAXIT; k++) {
        const r2 = x * x + y * y;
        w += H * r2;
        if (r2 < 1e-6) { ok = true; break; }
        if (r2 > 400) break;
        [x, y] = step(x, y, g, c, s);
      }
      W[j * N + i] = ok ? w : Infinity;
    }
    return W;
  }
  function region(W) {
    let rho = Infinity;
    for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
      const w = W[j * N + i];
      const edge = i === 0 || j === 0 || i === N - 1 || j === N - 1;
      if (!isFinite(w)) {
        for (const [a, b] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
          const ii = i + a, jj = j + b;
          if (ii >= 0 && jj >= 0 && ii < N && jj < N && isFinite(W[jj * N + ii])) rho = Math.min(rho, W[jj * N + ii]);
        }
      } else if (edge) rho = Math.min(rho, w);
    }
    rho *= 0.985;
    const M = new Uint8Array(N * N);
    for (let n = 0; n < N * N; n++) M[n] = W[n] <= rho ? 1 : 0;
    return { W, M, rho };
  }
  function colormap(t) {
    const stops = [[253, 230, 138], [249, 115, 22], [124, 58, 237], [30, 27, 75], [11, 16, 38]];
    const x = Math.max(0, Math.min(0.9999, t)) * (stops.length - 1), k = Math.floor(x), f = x - k;
    return stops[k].map((v, d) => v + (stops[k + 1][d] - v) * f);
  }
  function contour(M, color, dashed, width) {
    const s = cv.width / N;
    ctx.save(); ctx.strokeStyle = color; ctx.lineWidth = width; ctx.lineCap = "round";
    if (dashed) ctx.setLineDash([7, 6]);
    ctx.beginPath();
    for (let j = 0; j < N - 1; j++) for (let i = 0; i < N - 1; i++) {
      const a = M[j * N + i], b = M[j * N + i + 1], c = M[(j + 1) * N + i + 1], d = M[(j + 1) * N + i];
      const code = a * 8 + b * 4 + c * 2 + d;
      if (code === 0 || code === 15) continue;
      const x0 = (i + 0.5) * s, y0 = (j + 0.5) * s;
      const T = [x0 + s / 2, y0], R = [x0 + s, y0 + s / 2], B = [x0 + s / 2, y0 + s], Lf = [x0, y0 + s / 2];
      const seg = { 1: [Lf, B], 2: [B, R], 3: [Lf, R], 4: [T, R], 5: [Lf, T, B, R], 6: [T, B], 7: [Lf, T],
        8: [Lf, T], 9: [T, B], 10: [T, R, Lf, B], 11: [T, R], 12: [Lf, R], 13: [B, R], 14: [Lf, B] }[code];
      for (let q = 0; q < seg.length; q += 2) { ctx.moveTo(...seg[q]); ctx.lineTo(...seg[q + 1]); }
    }
    ctx.stroke(); ctx.restore();
  }
  function draw() {
    const showCost = $("#showCost").checked, showOld = $("#showOld").checked;
    const W = current.W, fin = Array.from(W).filter(isFinite), lo = Math.log(Math.min(...fin) + 1e-3), hi = Math.log(Math.max(...fin));
    for (let n = 0; n < N * N; n++) {
      let rgb;
      if (!isFinite(W[n])) rgb = [5, 8, 18];
      else if (!showCost) rgb = current.M[n] ? [24, 32, 58] : [12, 18, 36];
      else rgb = colormap((Math.log(W[n]) - lo) / (hi - lo + 1e-9));
      const bad = showOld && nominal.M[n] && !isFinite(W[n]);
      if (bad) rgb = [244, 63, 94];
      img.data.set([rgb[0], rgb[1], rgb[2], 255], 4 * n);
    }
    octx.putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = true; ctx.drawImage(off, 0, 0, cv.width, cv.height);
    ctx.strokeStyle = "rgba(255,255,255,.12)"; ctx.lineWidth = 1; ctx.beginPath();
    ctx.moveTo(cv.width / 2, 0); ctx.lineTo(cv.width / 2, cv.height); ctx.moveTo(0, cv.height / 2); ctx.lineTo(cv.width, cv.height / 2); ctx.stroke();
    contour(current.M, "#a3e635", false, 3);
    if (showOld) contour(nominal.M, "#ffffff", true, 2);
    ctx.fillStyle = "#fff"; ctx.beginPath(); ctx.arc(cv.width / 2, cv.height / 2, 5, 0, 7); ctx.fill();
    trajs.forEach(t => {
      ctx.strokeStyle = t.ok ? "#e0f2fe" : "#f43f5e"; ctx.lineWidth = 2; ctx.beginPath();
      t.pts.slice(0, t.n).forEach(([x, y], k) => { const p = toPx(x, y); k ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]); });
      ctx.stroke();
      const p = toPx(...t.pts[Math.min(t.n, t.pts.length) - 1]);
      ctx.fillStyle = t.ok ? "#e0f2fe" : "#f43f5e"; ctx.beginPath(); ctx.arc(p[0], p[1], 4, 0, 7); ctx.fill();
    });
    let now = 0, old = 0, bad = 0;
    for (let n = 0; n < N * N; n++) { now += current.M[n]; old += nominal.M[n]; bad += nominal.M[n] && !isFinite(W[n]) ? 1 : 0; }
    $("#areaNow").textContent = (100 * now / (N * N)).toFixed(1) + "%";
    $("#areaBad").textContent = (100 * bad / Math.max(1, old)).toFixed(1) + "%";
  }
  const toPx = (x, y) => [((x + L) / (2 * L)) * cv.width, ((L - y) / (2 * L)) * cv.height];
  let pending = false;
  function recompute() {
    if (pending) return; pending = true;
    requestAnimationFrame(() => {
      const g = +$("#gain").value, phi = (+$("#mis").value * Math.PI) / 180;
      $("#gOut").textContent = g.toFixed(2); $("#mOut").textContent = $("#mis").value + "°";
      current = region(costField(g, phi)); trajs = []; draw(); pending = false;
    });
  }
  cv.addEventListener("click", e => {
    const r = cv.getBoundingClientRect(), x = -L + (2 * L * (e.clientX - r.left)) / r.width, y = L - (2 * L * (e.clientY - r.top)) / r.height;
    const g = +$("#gain").value, phi = (+$("#mis").value * Math.PI) / 180, c = Math.cos(phi), s = Math.sin(phi);
    let p = [x, y]; const pts = [p]; let ok = false;
    for (let k = 0; k < MAXIT; k++) { p = step(p[0], p[1], g, c, s); pts.push(p); const r2 = p[0] ** 2 + p[1] ** 2; if (r2 < 1e-4) { ok = true; break; } if (r2 > 60) break; }
    const t = { pts, n: 1, ok }; trajs.push(t); if (trajs.length > 8) trajs.shift();
    const anim = () => { t.n = Math.min(pts.length, t.n + 3); draw(); if (t.n < pts.length) requestAnimationFrame(anim); };
    anim();
  });
  ["gain", "mis"].forEach(id => $("#" + id).addEventListener("input", recompute));
  ["showOld", "showCost"].forEach(id => $("#" + id).addEventListener("change", draw));
  $("#reset").addEventListener("click", () => { $("#gain").value = 1; $("#mis").value = 0; recompute(); });
  nominal = region(costField(1, 0)); current = nominal; draw();

  /* results explorer */
  const SERIES = [
    { key: "rV", name: "ours (learned maximal Lyapunov function)", color: "#7c3aed", r: 5.5 },
    { key: "rVh", name: "neural Lyapunov baseline", color: "#14b8a6", r: 4 },
    { key: "rVP", name: "linearisation-based", color: "#f97316", r: 4 },
    { key: "rVd", name: "squared norm", color: "#f59e0b", r: 4 },
  ];
  const on = Object.fromEntries(SERIES.map(s => [s.key, true]));
  let rows = [], fGroup = "all", fErr = "all";
  const tip = $("#tip");
  const ser = $("#series");
  SERIES.forEach(s => {
    const b = document.createElement("button"); b.innerHTML = `<i style="background:${s.color}"></i>${s.name}`;
    b.onclick = () => { on[s.key] = !on[s.key]; b.classList.toggle("off", !on[s.key]); chart(); };
    ser.appendChild(b);
  });
  function seg(id, set) {
    $$("#" + id + " button").forEach(b => b.addEventListener("click", () => {
      $$("#" + id + " button").forEach(x => x.classList.remove("on")); b.classList.add("on"); set(b.dataset.v); chart();
    }));
  }
  seg("fGroup", v => (fGroup = v)); seg("fErr", v => (fErr = v));

  function chart() {
    const data = rows.filter(r => (fGroup === "all" || r.group === fGroup) && (fErr === "all" || r.error.startsWith(fErr)))
      .sort((a, b) => Math.max(a.rVd, a.rVP) - Math.max(b.rVd, b.rVP));
    const box = $("#chart"), W = Math.max(box.clientWidth, data.length * 16 + 70), Hh = 360, ml = 46, mb = 30, mt = 10;
    const x = i => ml + (i + 0.5) * ((W - ml - 10) / data.length), y = v => mt + (1 - v / 100) * (Hh - mt - mb);
    let s = `<svg width="${W}" height="${Hh}" viewBox="0 0 ${W} ${Hh}" font-family="Inter" font-size="11">`;
    for (let v = 0; v <= 100; v += 25) s += `<line x1="${ml}" x2="${W - 10}" y1="${y(v)}" y2="${y(v)}" stroke="#eef0f4"/><text x="${ml - 8}" y="${y(v) + 4}" text-anchor="end" fill="#9aa1b2">${v}%</text>`;
    data.forEach((r, i) => {
      const vals = SERIES.filter(sr => on[sr.key]).map(sr => r[sr.key]);
      if (vals.length) s += `<line x1="${x(i)}" x2="${x(i)}" y1="${y(Math.min(...vals))}" y2="${y(Math.max(...vals))}" stroke="#e3e5ec" stroke-width="3" stroke-linecap="round"/>`;
      SERIES.slice().reverse().forEach(sr => { if (on[sr.key]) s += `<circle cx="${x(i)}" cy="${y(r[sr.key])}" r="${sr.r}" fill="${sr.color}" ${sr.key === "rV" ? 'stroke="#fff" stroke-width="1.5"' : 'opacity=".9"'}/>`; });
      s += `<rect class="hit" data-i="${i}" x="${x(i) - 8}" y="${mt}" width="16" height="${Hh - mt - mb}" fill="transparent"/>`;
    });
    s += `<text x="${ml}" y="${Hh - 8}" fill="#9aa1b2">closed loops, sorted by the larger quadratic baseline →</text></svg>`;
    box.innerHTML = s;
    $$(".hit", box).forEach(h => {
      h.addEventListener("mousemove", e => {
        const r = data[+h.dataset.i], pr = $(".explorer").getBoundingClientRect();
        tip.innerHTML = `<b>${r.group} · ${r.algo} seed ${r.seed} · ${r.error}</b><br>ours ${r.rV.toFixed(1)}% · neural ${r.noVh ? "no region" : r.rVh.toFixed(1) + "%"}<br>linearisation ${r.rVP.toFixed(1)}% · squared norm ${r.rVd.toFixed(1)}%`;
        tip.style.left = Math.min(e.clientX - pr.left + 14, pr.width - 260) + "px"; tip.style.top = e.clientY - pr.top + 14 + "px"; tip.style.opacity = 1;
      });
      h.addEventListener("mouseleave", () => (tip.style.opacity = 0));
    });
  }
  fetch("assets/data/actuator_loops.json").then(r => r.json()).then(d => { rows = d; chart(); });
  window.addEventListener("resize", () => rows.length && chart());

  /* SE(2) goal invariance */
  (function () {
    const W = $("#world"), Ec = $("#errf"); if (!W) return;
    const wx = W.getContext("2d"), ex = Ec.getContext("2d");
    const COL = ["#ff7a45", "#8b5cf6", "#22c5b4", "#38bdf8"], AMAX = 0.12, STEPS = 140;
    const wrap = a => Math.atan2(Math.sin(a), Math.cos(a));
    const compose = (a, b) => [a[0] + Math.cos(a[2]) * b[0] - Math.sin(a[2]) * b[1], a[1] + Math.sin(a[2]) * b[0] + Math.cos(a[2]) * b[1], wrap(a[2] + b[2])];
    const inv = a => [-(Math.cos(a[2]) * a[0] + Math.sin(a[2]) * a[1]), Math.sin(a[2]) * a[0] - Math.cos(a[2]) * a[1], -a[2]];
    function Exp(t) {
      const [vx, vy, w] = t;
      if (Math.abs(w) < 1e-9) return [vx, vy, 0];
      const s = Math.sin(w) / w, c = (1 - Math.cos(w)) / w;
      return [s * vx - c * vy, c * vx + s * vy, w];
    }
    function Log(g) {
      const w = g[2];
      if (Math.abs(w) < 1e-9) return [g[0], g[1], 0];
      const s = Math.sin(w) / w, c = (1 - Math.cos(w)) / w, d = s * s + c * c;
      return [(s * g[0] + c * g[1]) / d, (-c * g[0] + s * g[1]) / d, w];
    }
    function policy(e) {
      const t = Log(e).map(v => -v), n = Math.hypot(...t);
      return n > AMAX ? t.map(v => (v * AMAX) / n) : t;
    }
    let runs = [], k = 0, err0;
    function setup(seed) {
      let r = seed * 9301 + 49297;
      const rnd = () => ((r = (r * 9301 + 49297) % 233280) / 233280);
      err0 = [2.0 + rnd() * 0.6, -1.3 + rnd() * 0.6, 2.4 + rnd() * 0.5];
      runs = [0, 1, 2, 3].map(i => {
        const goal = [(i % 2 ? 1 : -1) * (1.3 + rnd() * 0.6), (i < 2 ? 1 : -1) * (1.2 + rnd() * 0.6), (rnd() - 0.5) * 6];
        let g = compose(goal, err0); const path = [g], epath = [err0];
        for (let s = 0; s < STEPS; s++) {
          const e = compose(inv(goal), g); g = compose(g, Exp(policy(e)));
          path.push(g); epath.push(compose(inv(goal), g));
        }
        return { goal, path, epath };
      });
      k = 0;
    }
    function body(c, p, sc, o, col, alpha) {
      c.save(); c.translate(o[0] + p[0] * sc, o[1] - p[1] * sc); c.rotate(-p[2]); c.globalAlpha = alpha;
      c.fillStyle = col; c.beginPath(); c.moveTo(16, 0); c.lineTo(-10, 9); c.lineTo(-5, 0); c.lineTo(-10, -9); c.closePath(); c.fill();
      c.restore();
    }
    function ghost(c, p, sc, o, col) {
      c.save(); c.translate(o[0] + p[0] * sc, o[1] - p[1] * sc); c.rotate(-p[2]); c.strokeStyle = col; c.lineWidth = 1.5; c.setLineDash([4, 3]);
      c.beginPath(); c.moveTo(16, 0); c.lineTo(-10, 9); c.lineTo(-5, 0); c.lineTo(-10, -9); c.closePath(); c.stroke(); c.restore();
    }
    function grid(c) {
      c.fillStyle = "#0b1224"; c.fillRect(0, 0, c.canvas.width, c.canvas.height);
      c.strokeStyle = "rgba(255,255,255,.06)"; c.lineWidth = 1;
      for (let x = 0; x <= c.canvas.width; x += 35) { c.beginPath(); c.moveTo(x, 0); c.lineTo(x, c.canvas.height); c.stroke(); }
      for (let y = 0; y <= c.canvas.height; y += 35) { c.beginPath(); c.moveTo(0, y); c.lineTo(c.canvas.width, y); c.stroke(); }
    }
    function trail(c, pts, sc, o, col, n) {
      c.strokeStyle = col; c.lineWidth = 2.2; c.globalAlpha = 0.85; c.beginPath();
      pts.slice(0, n + 1).forEach((p, i) => (i ? c.lineTo(o[0] + p[0] * sc, o[1] - p[1] * sc) : c.moveTo(o[0] + p[0] * sc, o[1] - p[1] * sc)));
      c.stroke(); c.globalAlpha = 1;
    }
    function fit(pts, c) {
      const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]);
      const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
      const sc = 0.78 * Math.min(c.width / (x1 - x0 + 1e-9), c.height / (y1 - y0 + 1e-9));
      return [sc, [c.width / 2 - sc * (x0 + x1) / 2, c.height / 2 + sc * (y0 + y1) / 2]];
    }
    function frame() {
      const [sw, ow] = fit(runs.flatMap(r => [...r.path, r.goal]), W), [se, oe] = fit([...runs[0].epath, [0, 0, 0]], Ec);
      grid(wx); grid(ex);
      runs.forEach((r, i) => { ghost(wx, r.goal, sw, ow, COL[i]); trail(wx, r.path, sw, ow, COL[i], k); body(wx, r.path[k], sw, ow, COL[i], 1); });
      ghost(ex, [0, 0, 0], se, oe, "#ffffff");
      ex.fillStyle = "rgba(255,255,255,.55)"; ex.font = "12px JetBrains Mono, monospace"; ex.fillText("goal = identity", oe[0] - 48, oe[1] + 32);
      runs.forEach((r, i) => { trail(ex, r.epath, se, oe, COL[i], k); });
      runs.forEach((r, i) => body(ex, r.epath[k], se, oe, COL[i], i === 0 ? 1 : 0.35));
      if (k < STEPS) { k++; requestAnimationFrame(frame); }
    }
    let seed = 3;
    const vis = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) { setup(seed); frame(); vis.disconnect(); } }), { threshold: 0.3 });
    vis.observe(W);
    $("#lieRun").addEventListener("click", () => { k = 0; frame(); });
    $("#lieShuffle").addEventListener("click", () => { seed += 1; setup(seed); frame(); });
  })();
})();
