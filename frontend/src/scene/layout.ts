import { HALL, MAT } from './palette';

export type Prop = {
  pos: [number, number, number];
  size: [number, number, number];
  color: string;
  yaw?: number;
};

const { X0, X1, H } = HALL;

const lcg = (seed: number) => {
  let a = seed;
  return () => {
    a = (a * 16807) % 2147483647;
    return a / 2147483647;
  };
};

function rack(out: Prop[], cx: number, cz: number, len: number, rnd: () => number) {
  const hx = 2.3;
  const hz = len / 2;
  const levels = [0.35, 1.5, 2.65, 3.8];

  for (const [qx, qz] of [[-1, -1], [1, -1], [-1, 1], [1, 1], [-1, 0], [1, 0]]) {
    out.push({
      pos: [cx + qx * (hx - 0.06), 2.3, cz + qz * (hz - 0.08)],
      size: [0.12, 4.6, 0.12],
      color: MAT.post,
    });
  }
  out.push({ pos: [cx - hx + 0.2, 0.25, cz - hz - 0.2], size: [0.6, 0.5, 0.6], color: MAT.guard });
  out.push({ pos: [cx + hx - 0.2, 0.25, cz - hz - 0.2], size: [0.6, 0.5, 0.6], color: MAT.guard });

  for (const y of levels) {
    out.push({ pos: [cx - hx + 0.06, y, cz], size: [0.1, 0.14, len], color: MAT.beam });
    out.push({ pos: [cx + hx - 0.06, y, cz], size: [0.1, 0.14, len], color: MAT.beam });
    out.push({ pos: [cx, y, cz], size: [(hx - 0.06) * 2, 0.04, len], color: MAT.steel });
    for (let i = 0; i < 4; i++) {
      if (rnd() < 0.08) continue;
      const sz = (hz / 4) * (0.7 + rnd() * 0.25);
      const sy = 0.3 + rnd() * 0.38;
      const zc = cz - hz + (i + 0.5) * ((2 * hz) / 4);
      for (const dx of [-0.95, 0.95]) {
        out.push({
          pos: [cx + dx + (rnd() - 0.5) * 0.2, y + 0.04 + sy, zc],
          size: [1.7, sy * 2, sz * 2],
          color: rnd() < 0.62 ? MAT.crate : MAT.tote,
        });
      }
    }
  }
}

function pallets(out: Prop[], x: number, z: number, n: number) {
  for (let i = 0; i < n; i++) {
    out.push({ pos: [x, 0.08 + i * 0.2, z], size: [1.24, 0.16, 1.0], color: MAT.wood });
  }
}

function cartons(out: Prop[], x: number, z: number, nx: number, ny: number) {
  for (let i = 0; i < nx; i++) {
    for (let j = 0; j < ny; j++) {
      out.push({
        pos: [x + (i - (nx - 1) / 2) * 0.95, 0.44 + j * 0.46, z],
        size: [0.92, 0.46, 0.84],
        color: MAT.crate,
      });
    }
  }
}

export function buildHallProps(): Prop[] {
  const out: Prop[] = [];
  const rnd = lcg(11);

  for (let k = 0; k < 8; k++) {
    const z = 8 + 12 * k;
    for (const x of [-8, 8]) {
      out.push({ pos: [x, H / 2, z], size: [1.1, H, 1.1], color: MAT.conc });
      out.push({ pos: [x, 0.1, z], size: [1.4, 0.2, 1.4], color: MAT.slab });
      out.push({ pos: [x, H - 0.6, z], size: [1.4, 1.2, 1.4], color: MAT.conc });
    }
  }

  for (const x of [-8, 0, 8]) {
    out.push({ pos: [x, H - 0.3, 52], size: [0.6, 0.6, 92], color: MAT.steel });
  }
  for (let k = 0; k < 8; k++) {
    out.push({ pos: [0, H - 0.3, 8 + 12 * k], size: [32, 0.6, 0.44], color: MAT.steel });
  }

  for (let k = 0; k < 7; k++) {
    const z = 14 + 12 * k;
    if (!(k === 2 || k === 5)) rack(out, -12.6, z, 9, rnd);
    if (k !== 3) rack(out, 12.6, z, 9, rnd);
  }

  pallets(out, -9.4, 31, 6);
  pallets(out, -10.2, 31, 5);
  pallets(out, -9.6, 32.2, 4);
  pallets(out, 9.4, 49, 5);
  pallets(out, 10.3, 49.2, 6);
  pallets(out, -9.8, 72, 5);
  pallets(out, -10.8, 72.2, 4);
  cartons(out, 9.8, 36.5, 2, 3);
  cartons(out, -9.6, 58, 2, 3);
  cartons(out, 10, 66, 2, 2);
  out.push({ pos: [-10.6, 0.5, 40], size: [1.6, 1.0, 0.9], color: MAT.tote });

  return out;
}

export type TrussBar = { pos: [number, number, number]; len: number; angle: number; thick: number };

export function buildTrusses(): TrussBar[] {
  const bars: TrussBar[] = [];
  const A = H + 3.2;
  const bar = (z: number, p1: [number, number], p2: [number, number], thick: number) => {
    const dx = p2[0] - p1[0];
    const dy = p2[1] - p1[1];
    bars.push({
      pos: [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2, z],
      len: Math.hypot(dx, dy),
      angle: Math.atan2(dy, dx),
      thick,
    });
  };
  for (let k = 0; k < 8; k++) {
    const z = 8 + 12 * k;
    bar(z, [X0, H], [0, A], 0.36);
    bar(z, [X1, H], [0, A], 0.36);
    for (const x of [-8, 0, 8]) bar(z, [x, H], [x, H + 3.2 * (1 - Math.abs(x) / 16)], 0.22);
    bar(z, [X0, H], [-8, H + 1.6], 0.2);
    bar(z, [-8, H + 1.6], [0, H], 0.2);
    bar(z, [0, H], [8, H + 1.6], 0.2);
    bar(z, [8, H + 1.6], [X1, H], 0.2);
  }
  return bars;
}

export function buildCeilingLights(): [number, number, number][] {
  const out: [number, number, number][] = [];
  for (let k = 0; k < 8; k++) {
    const z = 14 + 12 * k;
    for (const x of [-8, 0, 8]) out.push([x, H - 1.2, z]);
  }
  return out;
}

export const WINDOW_Z = Array.from({ length: 8 }, (_, k) => 14 + 12 * k);

export const LANE_EDGES = [-6.5, 6.5];
export const LANE_DASHES: { x: number; color: string }[] = [
  { x: -2.6, color: '#FFE2B0' },
  { x: 2.6, color: '#F2A33A' },
];

export { HALL };
