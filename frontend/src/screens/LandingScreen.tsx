import { useEffect, useRef, useState } from 'react';
import { APP_NAME } from '../branding';
import { NorthstarMark } from '../overlays/Wordmark';

const STAR_COUNT = 860;
const FLIGHT_MS = 3000;
const HOLD_S = 0.42;
const NORTH_X = 0.56;
const NORTH_Y = 0.14;

// Rough blackbody sequence, hot (blue) to cool (orange-red).
const TINT = [
  [170, 196, 255],
  [202, 220, 255],
  [236, 240, 255],
  [255, 250, 240],
  [255, 236, 204],
  [255, 213, 166],
  [255, 184, 142],
] as const;

// Spectral mix: mostly white / blue-white with a warm tail.
const TINT_CDF = [0.07, 0.22, 0.52, 0.72, 0.87, 0.96, 1] as const;

type Star = {
  x: number;
  y: number;
  z: number;
  r: number;
  phase: number;
  phase2: number;
  tw: number;
  tw2: number;
  scint: number;
  vx: number;
  vy: number;
  tint: number;
  chroma: number;
  spike: number;
};

function mulberry32(seed: number) {
  let state = seed;
  return () => {
    state |= 0;
    state = (state + 0x6d2b79f5) | 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function smoothstep(edge0: number, edge1: number, value: number) {
  const t = Math.min(1, Math.max(0, (value - edge0) / (edge1 - edge0)));
  return t * t * (3 - 2 * t);
}

function approachOf(t: number) {
  if (t <= 0) return 0;
  if (t >= 1) return 1;
  return t ** 1.7;
}

function pickTint(roll: number) {
  for (let i = 0; i < TINT_CDF.length; i++) if (roll <= TINT_CDF[i]) return i;
  return TINT.length - 1;
}

function rgb(tint: number, alpha: number) {
  const c = TINT[tint];
  return `rgba(${c[0]},${c[1]},${c[2]},${alpha})`;
}

/** Point-spread sprite: tight core bleeding into a soft Airy-ish halo. */
function makeGlowSprite(tint: number) {
  const size = 64;
  const cvs = document.createElement('canvas');
  cvs.width = size;
  cvs.height = size;
  const g = cvs.getContext('2d');
  if (!g) return cvs;
  const c = size / 2;
  const grad = g.createRadialGradient(c, c, 0, c, c, c);
  grad.addColorStop(0, rgb(tint, 1));
  grad.addColorStop(0.07, rgb(tint, 0.92));
  grad.addColorStop(0.16, rgb(tint, 0.44));
  grad.addColorStop(0.3, rgb(tint, 0.14));
  grad.addColorStop(0.55, rgb(tint, 0.04));
  grad.addColorStop(1, rgb(tint, 0));
  g.fillStyle = grad;
  g.fillRect(0, 0, size, size);
  return cvs;
}

/** Four-way diffraction cross with tapered ends, pre-rendered once per tint. */
function makeSpikeSprite(tint: number) {
  const size = 128;
  const cvs = document.createElement('canvas');
  cvs.width = size;
  cvs.height = size;
  const g = cvs.getContext('2d');
  if (!g) return cvs;
  const c = size / 2;
  g.lineCap = 'round';
  for (const vertical of [false, true]) {
    const grad = vertical
      ? g.createLinearGradient(0, 0, 0, size)
      : g.createLinearGradient(0, 0, size, 0);
    grad.addColorStop(0, rgb(tint, 0));
    grad.addColorStop(0.3, rgb(tint, 0.1));
    grad.addColorStop(0.46, rgb(tint, 0.6));
    grad.addColorStop(0.5, rgb(tint, 1));
    grad.addColorStop(0.54, rgb(tint, 0.6));
    grad.addColorStop(0.7, rgb(tint, 0.1));
    grad.addColorStop(1, rgb(tint, 0));
    g.strokeStyle = grad;
    g.lineWidth = vertical ? 1.6 : 1.6;
    g.beginPath();
    if (vertical) {
      g.moveTo(c, 2);
      g.lineTo(c, size - 2);
    } else {
      g.moveTo(2, c);
      g.lineTo(size - 2, c);
    }
    g.stroke();
  }
  return cvs;
}

/**
 * North-star corona and halo, baked at two wake levels. The live values only
 * change stop alphas, so cross-fading the two bakes matches the gradient to
 * within a pixel of noise while costing nothing per frame.
 */
function makeHaloSprite(stops: Array<[number, string]>) {
  const size = 512;
  const cvs = document.createElement('canvas');
  cvs.width = size;
  cvs.height = size;
  const g = cvs.getContext('2d');
  if (!g) return cvs;
  const c = size / 2;
  const grad = g.createRadialGradient(c, c, 0, c, c, c);
  for (const [at, color] of stops) grad.addColorStop(at, color);
  g.fillStyle = grad;
  g.fillRect(0, 0, size, size);
  return cvs;
}

function makeCoronaSprite(wake: number) {
  return makeHaloSprite([
    [0, `rgba(190,212,255,${0.22 * (0.6 + wake * 0.4)})`],
    [0.35, `rgba(150,180,255,${0.09 * (0.6 + wake * 0.4)})`],
    [1, 'rgba(0,0,0,0)'],
  ]);
}

function makeGlowCoreSprite(wake: number) {
  return makeHaloSprite([
    [0, `rgba(255,255,255,${Math.min(1, 0.5 + wake * 0.5)})`],
    [0.1, `rgba(255,248,232,${Math.min(1, 0.42 + wake * 0.45)})`],
    [0.24, `rgba(255,228,190,${0.26 + wake * 0.42})`],
    [0.52, `rgba(186,206,255,0.16)`],
    [1, 'rgba(0,0,0,0)'],
  ]);
}

/** One diffraction ray pointing +x, tapered in both thickness and alpha. */
function makeRaySprite() {
  const len = 256;
  const thick = 16;
  const cvs = document.createElement('canvas');
  cvs.width = len;
  cvs.height = thick;
  const g = cvs.getContext('2d');
  if (!g) return cvs;
  const mid = thick / 2;
  const grad = g.createLinearGradient(0, 0, len, 0);
  grad.addColorStop(0, 'rgba(255,252,244,0.95)');
  grad.addColorStop(0.2, 'rgba(232,240,255,0.5)');
  grad.addColorStop(0.6, 'rgba(196,214,255,0.14)');
  grad.addColorStop(1, 'rgba(176,200,255,0)');
  g.fillStyle = grad;
  g.beginPath();
  g.moveTo(0, mid - 4);
  g.lineTo(len, mid - 0.35);
  g.lineTo(len, mid + 0.35);
  g.lineTo(0, mid + 4);
  g.closePath();
  g.fill();
  return cvs;
}

// angle, length scale, width scale, strength
const RAYS: Array<[number, number, number, number]> = [
  [0, 1.0, 1.0, 0.95],
  [Math.PI / 2, 0.92, 0.95, 0.95],
  [Math.PI, 1.0, 1.0, 0.95],
  [-Math.PI / 2, 0.92, 0.95, 0.95],
  [Math.PI / 4, 0.4, 0.55, 0.45],
  [(3 * Math.PI) / 4, 0.4, 0.55, 0.45],
  [(-3 * Math.PI) / 4, 0.4, 0.55, 0.45],
  [-Math.PI / 4, 0.4, 0.55, 0.45],
];

/** Faint galactic band + a couple of dust clouds so the sky is never flat black. */
function makeSkyWash(w: number, h: number) {
  const cvs = document.createElement('canvas');
  cvs.width = Math.max(1, Math.round(w));
  cvs.height = Math.max(1, Math.round(h));
  const g = cvs.getContext('2d');
  if (!g) return cvs;
  g.save();
  g.translate(w * 0.58, h * 0.3);
  g.rotate(-0.48);
  g.scale(1, 0.26);
  const band = g.createRadialGradient(0, 0, 0, 0, 0, Math.max(w, h) * 0.78);
  band.addColorStop(0, 'rgba(126,142,196,0.17)');
  band.addColorStop(0.42, 'rgba(96,108,170,0.09)');
  band.addColorStop(0.75, 'rgba(60,70,122,0.035)');
  band.addColorStop(1, 'rgba(0,0,0,0)');
  g.fillStyle = band;
  g.beginPath();
  g.arc(0, 0, Math.max(w, h) * 0.78, 0, Math.PI * 2);
  g.fill();
  g.restore();

  const clouds: Array<[number, number, number, string]> = [
    [0.18, 0.72, 0.3, 'rgba(82,74,150,0.1)'],
    [0.82, 0.62, 0.26, 'rgba(60,96,158,0.085)'],
    [0.38, 0.2, 0.22, 'rgba(110,86,140,0.07)'],
  ];
  for (const [cx, cy, cr, color] of clouds) {
    const r = Math.max(w, h) * cr;
    const grad = g.createRadialGradient(cx * w, cy * h, 0, cx * w, cy * h, r);
    grad.addColorStop(0, color);
    grad.addColorStop(1, 'rgba(0,0,0,0)');
    g.fillStyle = grad;
    g.fillRect(cx * w - r, cy * h - r, r * 2, r * 2);
  }
  return cvs;
}

function makeStars(rand: () => number): Star[] {
  const stars: Star[] = [];
  for (let i = 0; i < STAR_COUNT; i++) {
    const roll = rand();
    const z = rand() ** 0.65;
    let r: number;
    let spike = 0;
    let tint: number;
    if (roll > 0.972) {
      // Rare naked-eye giants: big, spiked, usually hot and blue-white.
      r = 1.3 + rand() * 0.5;
      spike = 8 + rand() * 7;
      tint = pickTint(rand() * 0.55);
    } else if (roll > 0.86) {
      r = 0.85 + rand() * 0.42;
      spike = rand() > 0.72 ? 4 + rand() * 3 : 0;
      tint = pickTint(rand() * 0.9);
    } else {
      r = 0.4 + rand() * 0.45;
      tint = pickTint(rand());
    }
    const drift = rand();
    stars.push({
      x: rand(),
      y: rand(),
      z,
      r,
      phase: rand() * Math.PI * 2,
      phase2: rand() * Math.PI * 2,
      tw: 0.45 + rand() * 1.6,
      tw2: 2.3 + rand() * 4.4,
      // Atmospheric scintillation reads strongest on the brighter points.
      scint: 0.16 + (1 - z) * 0.16 + (spike > 0 ? 0.14 : 0) + drift * 0.1,
      vx: 0.008 + z * 0.022,
      vy: (rand() - 0.5) * 0.004,
      tint,
      chroma: Math.max(0, Math.min(TINT.length - 1, tint + (rand() > 0.5 ? 2 : -2))),
      spike,
    });
  }
  stars.sort((a, b) => a.z - b.z);
  return stars;
}

export function LandingScreen({ onArrive, onFinished }: { onArrive: () => void; onFinished: () => void }) {
  const rootRef = useRef<HTMLElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const whiteRef = useRef<HTMLDivElement>(null);
  const lockupRef = useRef<HTMLDivElement>(null);
  const onArriveRef = useRef(onArrive);
  const onFinishedRef = useRef(onFinished);
  const enterRef = useRef<() => void>(() => {});
  const [departing, setDeparting] = useState(false);

  onArriveRef.current = onArrive;
  onFinishedRef.current = onFinished;

  useEffect(() => {
    const canvas = canvasRef.current;
    const white = whiteRef.current;
    const lockup = lockupRef.current;
    const root = rootRef.current;
    if (!canvas || !white || !lockup || !root) return;
    const ctx = canvas.getContext('2d', { alpha: false });
    if (!ctx) return;

    const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const stars = makeStars(mulberry32(0x4e7374));
    const glowSprites = TINT.map((_, i) => makeGlowSprite(i));
    const spikeSprites = TINT.map((_, i) => makeSpikeSprite(i));
    const coronaIdle = makeCoronaSprite(0.35);
    const coronaWake = makeCoronaSprite(1);
    const glowIdle = makeGlowCoreSprite(0.35);
    const glowWake = makeGlowCoreSprite(1);
    const raySprite = makeRaySprite();
    let wash: HTMLCanvasElement | null = null;
    let washW = 0;
    let washH = 0;

    // Canvas and card geometry are measured on resize, never inside the frame
    // loop -- reading them per frame forces a synchronous layout and is the
    // main source of stutter.
    let viewW = 0;
    let viewH = 0;
    let cardRect: DOMRect | null = null;
    const measure = () => {
      const rect = canvas.getBoundingClientRect();
      viewW = rect.width;
      viewH = rect.height;
      const card = lockup.querySelector('.ns-card');
      cardRect = card ? card.getBoundingClientRect() : null;
    };
    measure();

    let raf = 0;
    let stopped = false;
    let mode: 'idle' | 'flight' | 'hold' | 'done' = 'idle';
    let clock = 0;
    let last = performance.now();
    let flightStart = 0;
    let hold = 0;
    let originX = NORTH_X;
    let originY = NORTH_Y;
    let prevTravel = 0;
    let smoothDt = 1 / 60;
    let lastVeil = -1;
    const flightMs = reduce ? 480 : FLIGHT_MS;

    const northAt = (time: number) => ({
      x: NORTH_X + Math.sin(time * 0.13) * 0.012,
      y: NORTH_Y + Math.cos(time * 0.1) * 0.007,
    });

    const draw = (now: number) => {
      const dt = Math.min(0.05, (now - last) / 1000);
      last = now;
      // Vsync delivers noisy deltas; integrating the smoothed value keeps the
      // drift and twinkle from visibly jittering on an uneven frame.
      smoothDt += (dt - smoothDt) * 0.18;
      const step = reduce ? 0 : smoothDt;
      clock += step;

      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const w = viewW;
      const h = viewH;
      if (w < 1 || h < 1) {
        measure();
        if (!stopped && mode !== 'done') raf = requestAnimationFrame(draw);
        return;
      }

      const bw = Math.round(w * dpr);
      const bh = Math.round(h * dpr);
      if (canvas.width !== bw || canvas.height !== bh) {
        canvas.width = bw;
        canvas.height = bh;
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.globalCompositeOperation = 'source-over';
      ctx.globalAlpha = 1;
      ctx.fillStyle = '#01010a';
      ctx.fillRect(0, 0, w, h);

      if (!wash || washW !== Math.round(w) || washH !== Math.round(h)) {
        washW = Math.round(w);
        washH = Math.round(h);
        wash = makeSkyWash(washW, washH);
      }

      let travel = 0;
      let t = 0;
      if (mode !== 'idle') {
        t = Math.min(1, (now - flightStart) / flightMs);
        travel = reduce ? 0 : approachOf(t);
        const fade = smoothstep(0.26, 0.48, t);
        lockup.style.opacity = String(1 - fade);
        lockup.style.transform = `translate3d(0,${(fade * 16).toFixed(2)}px,0)`;
      }

      ctx.globalAlpha = 1 - travel * 0.65;
      ctx.drawImage(wash, 0, 0, washW, washH, 0, 0, w, h);
      ctx.globalAlpha = 1;

      const idleNorth = northAt(clock);
      const nx = mode === 'idle' ? idleNorth.x : originX;
      const ny = mode === 'idle' ? idleNorth.y : originY;
      const pushK = travel;
      const prevPushK = prevTravel;
      const card = travel < 0.35 ? cardRect : null;
      const cardPad = 36;

      // Stars accumulate light, so overlapping halos bloom instead of flattening.
      ctx.globalCompositeOperation = 'lighter';
      ctx.lineCap = 'round';

      for (let i = 0; i < stars.length; i++) {
        const s = stars[i];
        if (!reduce) {
          s.x -= s.vx * step;
          s.y += s.vy * step;
          if (s.x < 0) s.x += 1;
          else if (s.x > 1) s.x -= 1;
          if (s.y < 0) s.y += 1;
          else if (s.y > 1) s.y -= 1;
        }
        const depth = 0.18 + s.z * 0.62;
        const push = pushK * depth;
        const sx = nx + (s.x - nx) * (1 + push);
        const sy = ny + (s.y - ny) * (1 + push);
        if (sx < -0.1 || sx > 1.1 || sy < -0.1 || sy > 1.1) continue;

        // Two incommensurate frequencies read as air turbulence, not a pulse.
        const slow = Math.sin(clock * s.tw + s.phase);
        const fast = Math.sin(clock * s.tw2 + s.phase2);
        const shimmer = slow * 0.62 + fast * 0.38;
        const twinkle = Math.max(0.1, 1 + shimmer * s.scint);
        const alpha = Math.min(1, (0.34 + s.z * 0.66) * twinkle * (1 + travel * 0.35));
        const px = sx * w;
        const py = sy * h;
        if (
          card &&
          px > card.left - cardPad &&
          px < card.right + cardPad &&
          py > card.top - cardPad &&
          py < card.bottom + cardPad
        ) {
          continue;
        }
        const radius = s.r * (0.65 + s.z) * (1 + travel * s.z * 1.6);

        // Radial motion blur while the camera accelerates toward the star.
        if (prevPushK !== pushK) {
          const prevPush = prevPushK * depth;
          const qx = (nx + (s.x - nx) * (1 + prevPush)) * w;
          const qy = (ny + (s.y - ny) * (1 + prevPush)) * h;
          const dx = px - qx;
          const dy = py - qy;
          if (dx * dx + dy * dy > 2.2) {
            ctx.globalAlpha = alpha * 0.55;
            ctx.strokeStyle = rgb(s.tint, 1);
            ctx.lineWidth = Math.max(0.6, radius * 0.9);
            ctx.beginPath();
            ctx.moveTo(qx, qy);
            ctx.lineTo(px, py);
            ctx.stroke();
          }
        }

        if (s.spike > 0) {
          const len = (s.spike + radius * 2.2) * (0.8 + twinkle * 0.3);
          // Scintillation dispersing into colour, the way bright stars actually flicker.
          const chromaMix = Math.max(0, fast) * 0.5;
          const sprite = spikeSprites[s.tint];
          const span = len * 2;
          ctx.globalAlpha = Math.min(1, alpha * 0.85) * (1 - chromaMix * 0.45);
          ctx.drawImage(sprite, px - len, py - len, span, span);
          if (chromaMix > 0.05) {
            ctx.globalAlpha = alpha * 0.45 * chromaMix;
            ctx.drawImage(spikeSprites[s.chroma], px - len, py - len, span, span);
          }
        }

        const halo = radius * (s.spike > 0 ? 7.5 : 5.2);
        ctx.globalAlpha = alpha;
        ctx.drawImage(glowSprites[s.tint], px - halo, py - halo, halo * 2, halo * 2);
      }

      ctx.globalAlpha = 1;
      prevTravel = travel;

      // --- North star -------------------------------------------------------
      // Layered flicker: a slow breath plus faster seeing-cell jitter.
      const breath = Math.sin(clock * 1.5);
      const jitter = Math.sin(clock * 6.3 + 1.1) * 0.5 + Math.sin(clock * 11.7 + 2.7) * 0.3;
      const pulse = reduce ? 1 : 0.9 + breath * 0.1 + jitter * 0.045;
      const cover = Math.hypot(Math.max(nx, 1 - nx) * w, Math.max(ny, 1 - ny) * h);
      const grown = travel ** 1.55;
      const reach = grown * cover;
      const disc = 2.15 * pulse + reach * 1.15;
      const glowR = 30 * pulse + reach * 2.2;
      const gx = nx * w;
      const gy = ny * h;
      const wake = mode === 'idle' ? 0.35 : Math.min(1, 0.35 + t * 3);

      // Cross-fade the two bakes to land on the current wake level.
      const wakeMix = smoothstep(0.35, 1, wake);

      // Outer corona, cool and wide.
      const coronaR = Math.max(glowR * 2.6, disc + 24);
      ctx.globalAlpha = pulse * (1 - grown * 0.5) * (1 - wakeMix);
      ctx.drawImage(coronaIdle, gx - coronaR, gy - coronaR, coronaR * 2, coronaR * 2);
      ctx.globalAlpha = pulse * (1 - grown * 0.5) * wakeMix;
      ctx.drawImage(coronaWake, gx - coronaR, gy - coronaR, coronaR * 2, coronaR * 2);

      // Inner halo, warm at the throat and cooling outward.
      const haloR = Math.max(glowR, disc);
      ctx.globalAlpha = pulse * (1 - wakeMix);
      ctx.drawImage(glowIdle, gx - haloR, gy - haloR, haloR * 2, haloR * 2);
      ctx.globalAlpha = pulse * wakeMix;
      ctx.drawImage(glowWake, gx - haloR, gy - haloR, haloR * 2, haloR * 2);
      ctx.globalAlpha = 1;

      const spikeFade = 1 - smoothstep(0.04, 0.2, travel);
      if (spikeFade > 0.04) {
        const base = (18 + Math.min(disc, 22) * 2.6) * pulse;
        ctx.save();
        ctx.translate(gx, gy);
        ctx.rotate(Math.sin(clock * 0.08) * 0.05);
        // Four long primaries plus four short diagonals, each tapered to nothing.
        const rayWidth = Math.min(2.4, 0.9 + grown * 4);
        let turned = 0;
        for (const [angle, lenScale, width, strength] of RAYS) {
          ctx.rotate(angle - turned);
          turned = angle;
          const sy = (rayWidth * width) / 4;
          ctx.globalAlpha = spikeFade * strength;
          ctx.drawImage(raySprite, 0, -8 * sy, base * lenScale, 16 * sy);
        }
        ctx.restore();
      }

      ctx.globalAlpha = 1;
      ctx.fillStyle = '#ffffff';
      ctx.beginPath();
      ctx.arc(gx, gy, Math.max(1.65, 1.7 * pulse + reach * 0.62), 0, Math.PI * 2);
      ctx.fill();

      ctx.globalCompositeOperation = 'source-over';
      const veil = reduce ? t : smoothstep(0.5, 0.94, travel);
      if (veil !== lastVeil) {
        lastVeil = veil;
        white.style.opacity = String(veil);
      }

      if (mode === 'flight' && t >= 1) {
        mode = 'hold';
        hold = 0;
        onArriveRef.current();
      }
      if (mode === 'hold') {
        hold += dt;
        if (hold >= (reduce ? 0.06 : HOLD_S)) {
          mode = 'done';
          root.classList.add('is-leaving');
        }
      }

      ctx.globalAlpha = 1;
      const looping = !reduce || mode === 'flight' || mode === 'hold';
      if (!stopped && mode !== 'done' && looping) raf = requestAnimationFrame(draw);
    };

    const enter = () => {
      if (mode !== 'idle') return;
      const here = northAt(clock);
      originX = here.x;
      originY = here.y;
      mode = 'flight';
      flightStart = performance.now();
      last = flightStart;
      prevTravel = 0;
      setDeparting(true);
      lockup.style.willChange = 'opacity, transform';
      white.style.willChange = 'opacity';
      if (reduce) {
        cancelAnimationFrame(raf);
        raf = requestAnimationFrame(draw);
      }
    };
    enterRef.current = enter;

    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Enter') enter();
    };
    window.addEventListener('keydown', onKey);

    const onResize = () => {
      measure();
      if (reduce && mode === 'idle') draw(performance.now());
    };
    window.addEventListener('resize', onResize);
    const observer = new ResizeObserver(onResize);
    observer.observe(canvas);
    // Webfont swap reflows the card, so re-measure once it settles.
    document.fonts?.ready.then(measure).catch(() => {});

    if (reduce) draw(performance.now());
    else raf = requestAnimationFrame(draw);

    return () => {
      stopped = true;
      cancelAnimationFrame(raf);
      window.removeEventListener('keydown', onKey);
      window.removeEventListener('resize', onResize);
      observer.disconnect();
      enterRef.current = () => {};
    };
  }, []);

  return (
    <section
      ref={rootRef}
      className="ns-landing"
      aria-label={APP_NAME}
      aria-hidden={departing || undefined}
      onAnimationEnd={(event) => {
        if (event.target !== event.currentTarget) return;
        if (event.animationName !== 'ns-unveil') return;
        onFinishedRef.current();
      }}
    >
      <canvas ref={canvasRef} className="ns-sky" aria-hidden="true" />
      <div ref={lockupRef} className="ns-lockup">
        <div className="ns-card">
          <div className="ns-mark">
            <NorthstarMark size={128} />
          </div>
          <h1 className="ns-name">{APP_NAME}</h1>
          <span className="ns-rule" aria-hidden="true" />
          <button type="button" className="ns-enter" disabled={departing} onClick={() => enterRef.current()}>
            <span className="ns-corner ns-nw" />
            <span className="ns-corner ns-ne" />
            <span className="ns-corner ns-sw" />
            <span className="ns-corner ns-se" />
            <span className="ns-enter-frame" />
            <NorthstarMark size={18} />
            <span className="ns-enter-label">Enter</span>
          </button>
        </div>
      </div>
      <div ref={whiteRef} className="ns-white" aria-hidden="true" />
    </section>
  );
}
