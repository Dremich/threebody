const MANIFEST_URL = "data/manifest.json";

const COLORS = [
  { fill: "#ff4d4d", stroke: "#ff4d4d" },
  { fill: "#4da3ff", stroke: "#4da3ff" },
  { fill: "#ffd84d", stroke: "#ffd84d" },
];

const state = {
  name: null,
  t: [],
  x: [],
  y: [],
  energy: null,
  i: 0,
  playing: false,
  speed: 1.0,
  lastTs: 0,
  limits: null,
  showFull: true,
  // Playback clock in *simulation time* (not frames).
  playT: 0,
};

const el = {
  dataset: document.getElementById("dataset"),
  play: document.getElementById("play"),
  slower: document.getElementById("slower"),
  faster: document.getElementById("faster"),
  speedLabel: document.getElementById("speedLabel"),
  scrub: document.getElementById("scrub"),
  stats: document.getElementById("stats"),
  traj: document.getElementById("traj"),
  energy: document.getElementById("energy"),
};

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function setSpeed(mult) {
  state.speed = clamp(mult, 0.1, 20.0);
  el.speedLabel.textContent = `${state.speed.toFixed(1)}×`;
}

function setPlaying(p) {
  const next = Boolean(p);
  // When starting playback, sync playback clock to current frame.
  if (next && !state.playing) {
    state.lastTs = 0;
    if (state.t.length) state.playT = state.t[state.i];
  }
  state.playing = next;
  el.play.textContent = state.playing ? "Pause" : "Play";
}

function parseQuery() {
  const params = new URLSearchParams(window.location.search);
  const data = params.get("data");
  return { data };
}

function setCanvasResolution(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const cssWidth = canvas.clientWidth;
  const cssHeight = canvas.clientHeight;
  const w = Math.max(2, Math.floor(cssWidth * dpr));
  const h = Math.max(2, Math.floor(cssHeight * dpr));
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
  return { w, h, dpr };
}

function computeLimits() {
  let xmin = Infinity;
  let xmax = -Infinity;
  let ymin = Infinity;
  let ymax = -Infinity;

  for (let k = 0; k < state.x.length; k++) {
    const xs = state.x[k];
    const ys = state.y[k];
    for (let j = 0; j < 3; j++) {
      const x = xs[j];
      const y = ys[j];
      if (x < xmin) xmin = x;
      if (x > xmax) xmax = x;
      if (y < ymin) ymin = y;
      if (y > ymax) ymax = y;
    }
  }

  if (!Number.isFinite(xmin) || !Number.isFinite(xmax) || !Number.isFinite(ymin) || !Number.isFinite(ymax)) {
    return { xmin: -1, xmax: 1, ymin: -1, ymax: 1 };
  }

  const dx = xmax - xmin;
  const dy = ymax - ymin;
  const span = Math.max(dx, dy) || 1;
  const cx = 0.5 * (xmin + xmax);
  const cy = 0.5 * (ymin + ymax);
  const half = 0.5 * span * 1.12;
  return { xmin: cx - half, xmax: cx + half, ymin: cy - half, ymax: cy + half };
}

function worldToCanvas(x, y, w, h, lim) {
  const sx = w / (lim.xmax - lim.xmin);
  const sy = h / (lim.ymax - lim.ymin);
  const s = Math.min(sx, sy);

  // Centered scale.
  const cx = 0.5 * (lim.xmin + lim.xmax);
  const cy = 0.5 * (lim.ymin + lim.ymax);

  const px = (x - cx) * s + w / 2;
  const py = (y - cy) * s + h / 2;

  // Canvas y-axis is down.
  return { px, py: h - py };
}

function drawTrajectory() {
  const ctx = el.traj.getContext("2d");
  const { w, h } = setCanvasResolution(el.traj);

  ctx.clearRect(0, 0, w, h);

  // Axes/background border subtly via canvas fill.
  ctx.save();
  ctx.globalAlpha = 0.02;
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, w, h);
  ctx.restore();

  const lim = state.limits;
  const i = state.i;
  const trail = 40;

  // Optional full trajectory (faint background)
  if (state.showFull && state.x.length) {
    for (let body = 0; body < 3; body++) {
      ctx.save();
      ctx.strokeStyle = COLORS[body].stroke;
      ctx.lineWidth = 1;
      ctx.globalAlpha = 0.18;
      ctx.beginPath();
      for (let k = 0; k < state.x.length; k++) {
        const { px, py } = worldToCanvas(state.x[k][body], state.y[k][body], w, h, lim);
        if (k === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.stroke();
      ctx.restore();
    }
  }

  // Trails
  for (let body = 0; body < 3; body++) {
    ctx.strokeStyle = COLORS[body].stroke;
    ctx.lineWidth = 2;
    ctx.globalAlpha = 0.65;
    ctx.beginPath();

    let started = false;
    const start = Math.max(0, i - trail);
    for (let k = start; k <= i; k++) {
      const { px, py } = worldToCanvas(state.x[k][body], state.y[k][body], w, h, lim);
      if (!started) {
        ctx.moveTo(px, py);
        started = true;
      } else {
        ctx.lineTo(px, py);
      }
    }
    ctx.stroke();
  }

  // Bodies (markers)
  for (let body = 0; body < 3; body++) {
    const { px, py } = worldToCanvas(state.x[i][body], state.y[i][body], w, h, lim);

    ctx.fillStyle = COLORS[body].fill;
    ctx.globalAlpha = 0.95;
    ctx.beginPath();
    ctx.arc(px, py, 6, 0, 2 * Math.PI);
    ctx.fill();
  }

  ctx.globalAlpha = 1.0;
}

function drawEnergy() {
  const ctx = el.energy.getContext("2d");
  const { w, h } = setCanvasResolution(el.energy);
  ctx.clearRect(0, 0, w, h);

  ctx.save();
  ctx.globalAlpha = 0.02;
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, w, h);
  ctx.restore();

  if (!state.energy || state.energy.length !== state.t.length) {
    ctx.save();
    ctx.fillStyle = "rgba(127,127,127,0.9)";
    ctx.font = "14px system-ui";
    ctx.fillText("Energy not available for this dataset.", 14, 26);
    ctx.restore();
    return;
  }

  const e = state.energy;
  let emin = Infinity;
  let emax = -Infinity;
  for (let i = 0; i < e.length; i++) {
    const v = e[i];
    if (v < emin) emin = v;
    if (v > emax) emax = v;
  }
  if (!(Number.isFinite(emin) && Number.isFinite(emax)) || emin === emax) {
    emin -= 1;
    emax += 1;
  }

  const pad = 18;
  const x0 = pad;
  const x1 = w - pad;
  const y0 = pad;
  const y1 = h - pad;

  // Plot line
  ctx.save();
  ctx.strokeStyle = "rgba(160,160,160,0.9)";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  for (let i = 0; i < e.length; i++) {
    const x = x0 + (i / (e.length - 1)) * (x1 - x0);
    const yn = (e[i] - emin) / (emax - emin);
    const y = y1 - yn * (y1 - y0);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.restore();

  // Current time cursor
  ctx.save();
  ctx.strokeStyle = "rgba(255,80,80,0.95)";
  ctx.lineWidth = 2;
  const t0 = state.t[0];
  const tEnd = state.t[state.t.length - 1];
  const duration = tEnd - t0;
  const alpha = duration > 0 ? (state.playT - t0) / duration : 0;
  const xi = x0 + clamp(alpha, 0, 1) * (x1 - x0);
  ctx.beginPath();
  ctx.moveTo(xi, y0);
  ctx.lineTo(xi, y1);
  ctx.stroke();
  ctx.restore();

  // Labels
  ctx.save();
  ctx.fillStyle = "rgba(127,127,127,0.85)";
  ctx.font = "12px system-ui";
  ctx.fillText("Energy", x0, 14);
  ctx.restore();
}

function updateStats() {
  const t = state.playT;
  const n = state.t.length;
  const name = state.name ?? "";
  el.stats.textContent = `${name} | frame ${state.i + 1}/${n} | t=${t.toFixed(4)}`;
}

function setFrame(i, { syncPlayT = false } = {}) {
  const n = state.t.length;
  state.i = clamp(i | 0, 0, n - 1);
  if (syncPlayT && state.t.length) {
    state.playT = state.t[state.i];
  }
  el.scrub.value = String(state.playT);
  updateStats();
  drawTrajectory();
  drawEnergy();
}

function lowerBound(arr, x) {
  // First index i such that arr[i] >= x
  let lo = 0;
  let hi = arr.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (arr[mid] < x) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

async function loadManifest() {
  const res = await fetch(MANIFEST_URL, { cache: "no-cache" });
  if (!res.ok) {
    throw new Error(`Failed to load manifest: ${res.status}`);
  }
  return await res.json();
}

async function loadDataset(url) {
  const res = await fetch(url, { cache: "no-cache" });
  if (!res.ok) {
    throw new Error(`Failed to load dataset: ${res.status}`);
  }
  return await res.json();
}

function populateDatasetSelect(entries, preferredKey) {
  el.dataset.innerHTML = "";
  for (const entry of entries) {
    const opt = document.createElement("option");
    opt.value = entry.key;
    opt.textContent = entry.label;
    el.dataset.appendChild(opt);
  }

  const keys = entries.map((e) => e.key);
  const key = preferredKey && keys.includes(preferredKey) ? preferredKey : entries[0]?.key;
  if (key) el.dataset.value = key;
  return key;
}

async function selectDataset(entries, key) {
  const entry = entries.find((e) => e.key === key);
  if (!entry) return;

  setPlaying(false);

  let payload;
  try {
    payload = await loadDataset(entry.url);
  } catch (err) {
    console.error(err);
    setPlaying(false);
    el.stats.textContent = `Missing dataset: ${entry.url}`;
    return;
  }
  state.name = payload.name || entry.label || entry.key;
  state.t = payload.t;
  state.x = payload.x;
  state.y = payload.y;
  state.energy = payload.energy || null;

  state.limits = computeLimits();

  const t0 = state.t[0];
  const tEnd = state.t[state.t.length - 1];
  const duration = tEnd - t0;
  el.scrub.min = String(t0);
  el.scrub.max = String(tEnd);
  // Pick a reasonable scrub resolution; user can still click energy plot / use arrows.
  const step = duration > 0 ? Math.max(duration / 5000, 1e-6) : 1;
  el.scrub.step = String(step);

  state.playT = state.t.length ? state.t[0] : 0;
  setFrame(0, { syncPlayT: true });
}

function tick(ts) {
  if (!state.lastTs) state.lastTs = ts;
  const dt = (ts - state.lastTs) / 1000;
  state.lastTs = ts;

  if (state.playing && state.t.length >= 2) {
    // Advance linearly in *simulation time*.
    // At 1.0×, 1 second wall-time advances 1 unit of simulation time.
    const dtSim = dt * state.speed;
    if (dtSim > 0) {
      state.playT += dtSim;
      const t0 = state.t[0];
      const tEnd = state.t[state.t.length - 1];
      const duration = tEnd - t0;

      // Loop playback.
      if (duration > 0 && state.playT > tEnd) {
        state.playT = t0 + ((state.playT - t0) % duration);
      }
      state.playT = clamp(state.playT, t0, tEnd);

      const j = lowerBound(state.t, state.playT);
      setFrame(clamp(j, 0, state.t.length - 1));
    }
  }

  requestAnimationFrame(tick);
}

function bindUI(entries) {
  el.play.addEventListener("click", () => setPlaying(!state.playing));
  el.slower.addEventListener("click", () => setSpeed(state.speed / 1.25));
  el.faster.addEventListener("click", () => setSpeed(state.speed * 1.25));

  el.scrub.addEventListener("input", (e) => {
    setPlaying(false);
    const tWanted = Number(e.target.value);
    if (!state.t.length) return;
    const t0 = state.t[0];
    const tEnd = state.t[state.t.length - 1];
    state.playT = clamp(tWanted, t0, tEnd);
    const j = lowerBound(state.t, state.playT);
    setFrame(clamp(j, 0, state.t.length - 1));
  });

  el.dataset.addEventListener("change", async () => {
    const key = el.dataset.value;
    const url = new URL(window.location.href);
    url.searchParams.set("data", key);
    window.history.replaceState({}, "", url.toString());
    await selectDataset(entries, key);
  });

  // Keyboard controls
  window.addEventListener("keydown", (e) => {
    if (e.target && (e.target.tagName === "INPUT" || e.target.tagName === "SELECT")) return;

    if (e.key === " ") {
      e.preventDefault();
      setPlaying(!state.playing);
      return;
    }

    if (e.key === "t" || e.key === "T") {
      state.showFull = !state.showFull;
      drawTrajectory();
      return;
    }

    if (e.key === "ArrowLeft") {
      e.preventDefault();
      setPlaying(false);
      setFrame(state.i - 1, { syncPlayT: true });
      return;
    }

    if (e.key === "ArrowRight") {
      e.preventDefault();
      setPlaying(false);
      setFrame(state.i + 1, { syncPlayT: true });
      return;
    }

    if (e.key === "-" || e.key === "_") {
      setSpeed(state.speed / 1.25);
      return;
    }

    if (e.key === "+" || e.key === "=") {
      setSpeed(state.speed * 1.25);
      return;
    }
  });

  // Energy plot click-to-scrub
  el.energy.addEventListener("pointerdown", (e) => {
    if (!state.t.length) return;
    const rect = el.energy.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width;
    const t0 = state.t[0];
    const tEnd = state.t[state.t.length - 1];
    state.playT = clamp(t0 + x * (tEnd - t0), t0, tEnd);
    const i = lowerBound(state.t, state.playT);
    setPlaying(false);
    setFrame(i, { syncPlayT: true });
  });

  window.addEventListener("resize", () => {
    drawTrajectory();
    drawEnergy();
  });
}

async function main() {
  setSpeed(2.0);
  setPlaying(false);

  const manifest = await loadManifest();
  const entries = manifest.entries || [];
  if (!entries.length) {
    throw new Error("Manifest has no entries");
  }

  const { data } = parseQuery();
  const initialKey = populateDatasetSelect(entries, data ?? "moth_I_BDF");

  bindUI(entries);
  await selectDataset(entries, initialKey);

  // Autoplay on load.
  setPlaying(true);

  requestAnimationFrame(tick);
}

main().catch((err) => {
  console.error(err);
  el.stats.textContent = String(err?.message || err);
});
