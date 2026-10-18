export type NetShape = {
  d: string;
  fill: string;
  fo: number;
  stroke: string;
  sw: number;
  so: number;
  cls: string;
  style?: { animationDuration?: string; animationDelay?: string };
};

export type NetLabel = { x: number; y: number; text: string };

export type NetOutput = {
  top: number;
  name: string;
  pct: number;
  w: number;
  bar: string;
  color: string;
  weight: number;
};

export type NetworkMode = 'track' | 'avoid';

export type NetworkGraph = {
  shapes: NetShape[];
  heads: NetLabel[];
  subs: NetLabel[];
  outputs: NetOutput[];
  why: string;
  foot: string;
};

const clamp = (v: number, a: number, b: number) => Math.max(a, Math.min(b, v));
const f1 = (v: number) => Math.round(v * 10) / 10;

const item = (
  d: string,
  fill?: string | null,
  fo?: number | null,
  stroke?: string | null,
  sw?: number | null,
  so?: number | null,
  cls?: string,
  style?: NetShape['style'],
): NetShape => ({
  d,
  fill: fill || 'none',
  fo: fo == null ? 1 : fo,
  stroke: stroke || 'none',
  sw: sw || 0,
  so: so == null ? 1 : so,
  cls: cls || '',
  style,
});

const circle = (cx: number, cy: number, r: number) =>
  'M' +
  f1(cx - r) +
  ' ' +
  f1(cy) +
  'a' +
  r +
  ' ' +
  r +
  ' 0 1 0 ' +
  f1(2 * r) +
  ' 0a' +
  r +
  ' ' +
  r +
  ' 0 1 0 ' +
  f1(-2 * r) +
  ' 0Z';

const makeRng = (s: number) => {
  let a = s;
  return () => {
    a = (a * 16807) % 2147483647;
    return a / 2147483647;
  };
};

const AMBER = ['#1c2229', '#2a2622', '#5a3a1f', '#a2621f', '#e08f2b', '#F2A33A', '#FFE2B0'];
const BLUE = ['#1c2229', '#212c3e', '#2a4270', '#3f65b6', '#5B8DEF', '#9dbaf5', '#E2ECFF'];

function grid(
  out: NetShape[],
  x0: number,
  y0: number,
  cols: number,
  rows: number,
  cell: number,
  gap: number,
  fn: (c: number, r: number) => number,
  scale: string[],
) {
  const buckets: string[][] = scale.map(() => []);
  const s = cell - gap;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const v = clamp(fn(c, r), 0, 1);
      buckets[Math.round(v * (scale.length - 1))].push(
        'M' + f1(x0 + c * cell) + ' ' + f1(y0 + r * cell) + 'h' + s + 'v' + s + 'h' + -s + 'Z',
      );
    }
  }
  buckets.forEach((arr, i) => {
    if (!arr.length) return;
    out.push(
      item(
        arr.join(''),
        scale[i],
        1,
        null,
        0,
        null,
        i >= 2 ? 'tw' : '',
        i >= 2
          ? {
              animationDuration: f1(2.2 + ((i * 0.7 + x0 * 0.013) % 2.6)) + 's',
              animationDelay: '-' + f1((i * 0.9 + y0 * 0.05 + x0 * 0.01) % 3) + 's',
            }
          : undefined,
      ),
    );
  });
}

function bundle(
  out: NetShape[],
  x1: number,
  ya: number,
  yb: number,
  x2: number,
  yc: number,
  yd: number,
  n: number,
  col: string,
  op: number,
  w: number,
  rng: () => number,
) {
  for (let i = 0; i < n; i++) {
    const y1 = ya + (yb - ya) * rng();
    const y2 = yc + (yd - yc) * rng();
    const xm = (x1 + x2) / 2;
    out.push(
      item(
        'M' +
          f1(x1) +
          ' ' +
          f1(y1) +
          'C' +
          f1(xm) +
          ' ' +
          f1(y1) +
          ' ' +
          f1(xm) +
          ' ' +
          f1(y2) +
          ' ' +
          f1(x2) +
          ' ' +
          f1(y2),
        'none',
        1,
        col,
        w,
        op,
      ),
    );
  }
}

const MODES = {
  track: {
    sel: 2,
    pct: [4, 9, 52, 14, 6, 11, 4],
    why: 'Path 3 has the widest clearance and a steady heading to the goal. Paths 4 and 5 pass close to the pallet stack.',
    foot: 'Reaches goal in 21 s',
    cluster: { c0: 8, c1: 11, r0: 1, r1: 6 },
    seed: 2719,
  },
  avoid: {
    sel: 5,
    pct: [5, 8, 22, 7, 3, 46, 9],
    why: 'Slowing down: a forklift is crossing inside the 4 m envelope. Path 3 stays available once it clears.',
    foot: 'Safety filter on · 4.2 ms',
    cluster: { c0: 4, c1: 8, r0: 1, r1: 4 },
    seed: 4173,
  },
} as const;

const NAMES = ['Path 1', 'Path 2', 'Path 3', 'Path 4', 'Path 5', 'Slow down', 'Stop'];

export function buildNetworkGraph(mode: NetworkMode): NetworkGraph {
  const M = MODES[mode] ?? MODES.track;
  const rng = makeRng(M.seed);
  const shapes: NetShape[] = [];
  const cl = M.cluster;

  const lr = makeRng(97);
  bundle(shapes, 294, 28, 136, 360, 28, 100, 14, '#E8E6E1', 0.07, 0.8, lr);
  bundle(shapes, 294, 28, 136, 360, 128, 164, 8, '#E8E6E1', 0.07, 0.8, lr);
  bundle(shapes, 620, 28, 100, 700, 28, 168, 12, '#E8E6E1', 0.08, 0.8, lr);
  bundle(shapes, 620, 128, 164, 700, 28, 168, 8, '#E8E6E1', 0.08, 0.8, lr);
  bundle(shapes, 840, 28, 168, 900, 28, 168, 18, '#F2A33A', 0.16, 0.8, lr);

  const inVals: number[][] = [];
  for (let r = 0; r < 12; r++) {
    inVals.push([]);
    for (let c = 0; c < 12; c++) inVals[r].push(rng());
  }
  const inFn = (c: number, r: number) => {
    const inside = c >= cl.c0 && c <= cl.c1 && r >= cl.r0 && r <= cl.r1;
    const edge = c === cl.c0 || c === cl.c1 || r === cl.r0 || r === cl.r1;
    if (inside) return (edge ? 0.45 : 0.7) + 0.3 * inVals[r][c];
    return 0.04 + 0.2 * Math.pow(inVals[r][c], 2.4) + (r > 8 ? 0.05 : 0);
  };
  grid(shapes, 22, 28, 12, 12, 13, 2, inFn, BLUE);
  shapes.push(
    item(
      'M' + (22 + 5 * 13 - 1.5) + ' ' + (28 + 10 * 13 - 1.5) + 'h' + 29 + 'v' + 16 + 'h-29Z',
      'none',
      1,
      '#E8E6E1',
      1.4,
      0.9,
    ),
  );

  const chans: [number, number, (c: number, r: number) => number][] = [
    [200, 28, (_c, r) => 0.1 + 0.55 * (r / 7) + 0.25 * rng()],
    [
      254,
      28,
      (c, r) => {
        const i = ((c * 12) / 8) | 0;
        const j = ((r * 12) / 8) | 0;
        return inFn(i, j) > 0.4 ? 0.8 + 0.2 * rng() : 0.05 * rng();
      },
    ],
    [
      200,
      96,
      (c, r) => {
        const i = ((c * 12) / 8) | 0;
        const j = ((r * 12) / 8) | 0;
        return inFn(i, j) > 0.4 ? 0.06 : 0.35 + 0.4 * rng();
      },
    ],
    [
      254,
      96,
      (c, r) => {
        const d = Math.hypot(c - 3.5, r - 0.8);
        return Math.max(0, 1 - d / 4.2) * (0.7 + 0.3 * rng());
      },
    ],
  ];
  chans.forEach((ch) => grid(shapes, ch[0], ch[1], 8, 8, 5, 1, ch[2], BLUE));

  const sparse = (k: number) => () => Math.pow(rng(), k);
  grid(shapes, 360, 28, 18, 8, 9, 1.5, sparse(2.4), AMBER);
  grid(shapes, 360, 128, 18, 4, 9, 1.5, sparse(2.6), AMBER);
  grid(shapes, 548, 28, 8, 8, 9, 1.5, sparse(2.2), AMBER);
  grid(shapes, 548, 128, 8, 4, 9, 1.5, sparse(2.6), AMBER);

  grid(shapes, 700, 28, 14, 14, 10, 1.6, sparse(2.0), AMBER);
  grid(shapes, 900, 28, 14, 14, 10, 1.6, sparse(1.6), AMBER);

  const outputs: NetOutput[] = [];
  const oy = (i: number) => 30 + 26 * i;
  for (let i = 0; i < 7; i++) {
    const isSel = i === M.sel;
    const pct = M.pct[i];
    const n = isSel ? 12 : 5 + Math.round(pct / 4);
    for (let k = 0; k < n; k++) {
      const y1 = 28 + 140 * rng();
      const xm = 1040 + (1176 - 1040) * 0.5;
      const y2 = oy(i);
      shapes.push(
        item(
          'M1040 ' +
            f1(y1) +
            'C' +
            f1(xm) +
            ' ' +
            f1(y1) +
            ' ' +
            f1(xm) +
            ' ' +
            f1(y2) +
            ' 1176 ' +
            f1(y2),
          'none',
          1,
          isSel ? '#F2A33A' : '#5B8DEF',
          isSel ? 1 : 0.8,
          isSel ? 0.7 : 0.08 + pct / 200,
          isSel ? 'fl' : '',
        ),
      );
    }
  }
  for (let i = 0; i < 7; i++) {
    const isSel = i === M.sel;
    const isPath = i < 5;
    const col = isSel ? '#F2A33A' : isPath ? '#5B8DEF' : '#E8E6E1';
    shapes.push(item(circle(1176, oy(i), isSel ? 5.5 : 4), col, isSel ? 1 : 0.5));
    if (isSel) {
      shapes.push(item(circle(1176, oy(i), 9), 'none', 1, '#E8E6E1', 1.4, 0.9));
      shapes.push(item(circle(1176, oy(i), 9), 'none', 1, '#F2A33A', 1.4, 0.9, 'rg'));
    }
    outputs.push({
      top: oy(i) - 9,
      name: NAMES[i],
      pct: M.pct[i],
      w: Math.round(clamp(M.pct[i] / 60, 0.04, 1) * 130),
      bar: isSel ? '#F2A33A' : 'rgba(232,230,225,0.34)',
      color: isSel ? '#E8E6E1' : 'rgba(232,230,225,0.72)',
      weight: isSel ? 600 : 400,
    });
  }

  const heads: NetLabel[] = [
    { x: 22, y: 6, text: 'Input · what it sees' },
    { x: 200, y: 6, text: 'Channels' },
    { x: 360, y: 6, text: 'Encoders' },
    { x: 700, y: 6, text: 'Decision trunk' },
    { x: 1168, y: 6, text: 'Output · 7 actions' },
  ];
  const subs: NetLabel[] = [
    { x: 22, y: 190, text: 'Local view · 12 × 12 around the rover' },
    { x: 200, y: 70, text: 'Depth' },
    { x: 254, y: 70, text: 'Obstacle' },
    { x: 200, y: 138, text: 'Clearance' },
    { x: 254, y: 138, text: 'Goal' },
    { x: 360, y: 102, text: 'Perception · 512 → 256' },
    { x: 360, y: 166, text: 'Goal and status · 64' },
    { x: 700, y: 172, text: 'Layer 1 · 256' },
    { x: 900, y: 172, text: 'Layer 2 · 256' },
  ];

  return { shapes, heads, subs, outputs, why: M.why, foot: M.foot };
}
