import { useEffect, useRef, useState } from 'react';
import { APP_NAME } from '../branding';
import { NorthstarMark } from '../overlays/Wordmark';

const STAR_COUNT = 860;
const FLIGHT_MS = 3000;
const HOLD_S = 0.42;
const NORTH_X = 0.56;
const NORTH_Y = 0.14;

const TINT = ['rgb(255,255,255)', 'rgb(198,214,255)', 'rgb(214,206,255)'] as const;

type Tint = 0 | 1 | 2;

type Star = {
  x: number;
  y: number;
  z: number;
  r: number;
  phase: number;
  tw: number;
  vx: number;
  vy: number;
  tint: Tint;
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

function makeStars(rand: () => number): Star[] {
  const stars: Star[] = [];
  for (let i = 0; i < STAR_COUNT; i++) {
    const roll = rand();
    const z = rand() ** 0.65;
    let r: number;
    let spike = 0;
    let tint: Tint = 0;
    if (roll > 0.972) {
      r = 1.25 + rand() * 0.45;
      spike = 7 + rand() * 6;
      tint = rand() > 0.5 ? 1 : 0;
    } else if (roll > 0.86) {
      r = 0.85 + rand() * 0.4;
      tint = rand() > 0.86 ? 2 : rand() > 0.5 ? 1 : 0;
    } else {
      r = 0.4 + rand() * 0.45;
      tint = rand() > 0.93 ? 2 : rand() > 0.74 ? 1 : 0;
    }
    stars.push({
      x: rand(),
      y: rand(),
      z,
      r,
      phase: rand() * Math.PI * 2,
      tw: 0.45 + rand() * 1.6,
      vx: 0.008 + z * 0.022,
      vy: (rand() - 0.5) * 0.004,
      tint,
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
    let raf = 0;
    let stopped = false;
    let mode: 'idle' | 'flight' | 'hold' | 'done' = 'idle';
    let clock = 0;
    let last = performance.now();
    let flightStart = 0;
    let hold = 0;
    let originX = NORTH_X;
    let originY = NORTH_Y;
    const flightMs = reduce ? 480 : FLIGHT_MS;

    const northAt = (time: number) => ({
      x: NORTH_X + Math.sin(time * 0.13) * 0.012,
      y: NORTH_Y + Math.cos(time * 0.1) * 0.007,
    });

    const draw = (now: number) => {
      const dt = Math.min(0.05, (now - last) / 1000);
      last = now;
      if (!reduce) clock += dt;

      const rect = canvas.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const w = rect.width;
      const h = rect.height;
      if (w < 1 || h < 1) {
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
      ctx.fillStyle = '#000';
      ctx.fillRect(0, 0, w, h);
      ctx.lineCap = 'butt';

      let travel = 0;
      let t = 0;
      if (mode !== 'idle') {
        t = Math.min(1, (now - flightStart) / flightMs);
        travel = reduce ? 0 : approachOf(t);
        const fade = smoothstep(0.26, 0.48, t);
        lockup.style.opacity = String(1 - fade);
        lockup.style.transform = `translate3d(0,${(fade * 16).toFixed(2)}px,0)`;
      }

      const idleNorth = northAt(clock);
      const nx = mode === 'idle' ? idleNorth.x : originX;
      const ny = mode === 'idle' ? idleNorth.y : originY;
      const pushK = travel;
      const card = travel < 0.35 ? lockup.querySelector('.ns-card')?.getBoundingClientRect() : undefined;
      const cardPad = 36;

      for (let i = 0; i < stars.length; i++) {
        const s = stars[i];
        if (!reduce) {
          s.x -= s.vx * dt;
          s.y += s.vy * dt;
          if (s.x < 0) s.x += 1;
          else if (s.x > 1) s.x -= 1;
          if (s.y < 0) s.y += 1;
          else if (s.y > 1) s.y -= 1;
        }
        const push = pushK * (0.18 + s.z * 0.62);
        const sx = nx + (s.x - nx) * (1 + push);
        const sy = ny + (s.y - ny) * (1 + push);
        if (sx < -0.08 || sx > 1.08 || sy < -0.08 || sy > 1.08) continue;

        const twinkle = 0.55 + 0.45 * Math.sin(clock * s.tw + s.phase);
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
        ctx.globalAlpha = alpha;
        ctx.fillStyle = TINT[s.tint];
        if (s.spike > 0) {
          const len = s.spike + radius * 1.6;
          ctx.strokeStyle = TINT[s.tint];
          ctx.lineWidth = 0.6;
          ctx.beginPath();
          ctx.moveTo(px - len, py);
          ctx.lineTo(px + len, py);
          ctx.moveTo(px, py - len);
          ctx.lineTo(px, py + len);
          ctx.stroke();
        }
        if (radius < 1.25) {
          const d = Math.max(0.7, radius);
          ctx.fillRect(px - d * 0.5, py - d * 0.5, d, d);
        } else {
          ctx.beginPath();
          ctx.arc(px, py, radius * 0.55, 0, Math.PI * 2);
          ctx.fill();
        }
      }

      const pulse = reduce ? 1 : 0.86 + 0.14 * Math.sin(clock * 1.5);
      const cover = Math.hypot(Math.max(nx, 1 - nx) * w, Math.max(ny, 1 - ny) * h);
      const grown = travel ** 1.55;
      const reach = grown * cover;
      const disc = 2.15 * pulse + reach * 1.15;
      const glowR = 28 * pulse + reach * 2.2;
      const gx = nx * w;
      const gy = ny * h;
      const wake = mode === 'idle' ? 0.35 : Math.min(1, 0.35 + t * 3);
      const glow = ctx.createRadialGradient(gx, gy, disc * 0.04, gx, gy, Math.max(glowR, disc + 8));
      glow.addColorStop(0, `rgba(255,255,255,${Math.min(1, 0.5 + wake * 0.5)})`);
      glow.addColorStop(0.18, `rgba(255,236,206,${(0.28 + wake * 0.45) * pulse})`);
      glow.addColorStop(0.46, `rgba(186,206,255,${0.16 * (1 - grown) * pulse})`);
      glow.addColorStop(1, 'rgba(0,0,0,0)');
      ctx.globalAlpha = 1;
      ctx.fillStyle = glow;
      ctx.beginPath();
      ctx.arc(gx, gy, Math.max(glowR, disc), 0, Math.PI * 2);
      ctx.fill();

      const spikeFade = 1 - smoothstep(0.04, 0.2, travel);
      if (spikeFade > 0.04) {
        const len = (16 + Math.min(disc, 22) * 2.4) * pulse;
        ctx.globalAlpha = 0.95 * spikeFade * pulse;
        ctx.strokeStyle = '#f5f8ff';
        ctx.lineWidth = Math.min(2.2, 0.85 + grown * 4);
        ctx.beginPath();
        ctx.moveTo(gx - len, gy);
        ctx.lineTo(gx + len, gy);
        ctx.moveTo(gx, gy - len);
        ctx.lineTo(gx, gy + len);
        ctx.stroke();
      }

      ctx.globalAlpha = 1;
      ctx.fillStyle = '#ffffff';
      ctx.beginPath();
      ctx.arc(gx, gy, Math.max(1.65, 1.7 * pulse + reach * 0.62), 0, Math.PI * 2);
      ctx.fill();

      white.style.opacity = String(reduce ? t : smoothstep(0.5, 0.94, travel));

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
      if (reduce && mode === 'idle') draw(performance.now());
    };
    window.addEventListener('resize', onResize);

    if (reduce) draw(performance.now());
    else raf = requestAnimationFrame(draw);

    return () => {
      stopped = true;
      cancelAnimationFrame(raf);
      window.removeEventListener('keydown', onKey);
      window.removeEventListener('resize', onResize);
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
