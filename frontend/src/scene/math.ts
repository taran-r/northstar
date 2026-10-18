
export type Pt = { x: number; z: number };

export const DEG = Math.PI / 180;

export const clamp = (v: number, a: number, b: number) => Math.max(a, Math.min(b, v));

const h2r = (h: string): [number, number, number] => [
  parseInt(h.slice(1, 3), 16),
  parseInt(h.slice(3, 5), 16),
  parseInt(h.slice(5, 7), 16),
];

const r2h = (r: number, g: number, b: number) =>
  '#' +
  [r, g, b]
    .map((v) => Math.round(clamp(v, 0, 255)).toString(16).padStart(2, '0'))
    .join('');

export const mix = (a: string, b: string, t: number) => {
  const A = h2r(a);
  const B = h2r(b);
  return r2h(A[0] + (B[0] - A[0]) * t, A[1] + (B[1] - A[1]) * t, A[2] + (B[2] - A[2]) * t);
};

export const smooth = (t: number) => t * t * (3 - 2 * t);

export const f1 = (v: number) => Math.round(v * 10) / 10;

export const pd = (pts: number[][]) =>
  'M' + pts.map((p) => f1(p[0]) + ' ' + f1(p[1])).join('L') + 'Z';

export const ell = (cx: number, cy: number, rx: number, ry: number) =>
  'M' +
  f1(cx - rx) +
  ' ' +
  f1(cy) +
  'a' +
  f1(rx) +
  ' ' +
  f1(ry) +
  ' 0 1 0 ' +
  f1(2 * rx) +
  ' 0a' +
  f1(rx) +
  ' ' +
  f1(ry) +
  ' 0 1 0 ' +
  f1(-2 * rx) +
  ' 0Z';

const cr = (a: number, b: number, c: number, d: number, t: number) =>
  0.5 *
  (2 * b + (-a + c) * t + (2 * a - 5 * b + 4 * c - d) * t * t + (-a + 3 * b - 3 * c + d) * t * t * t);

export const spline = (w: Pt[], seg: number): Pt[] => {
  const p = [w[0]].concat(w, [w[w.length - 1]]);
  const out: Pt[] = [];
  for (let i = 1; i < p.length - 2; i++) {
    for (let s = 0; s < seg; s++) {
      const t = s / seg;
      out.push({
        x: cr(p[i - 1].x, p[i].x, p[i + 1].x, p[i + 2].x, t),
        z: cr(p[i - 1].z, p[i].z, p[i + 1].z, p[i + 2].z, t),
      });
    }
  }
  out.push(w[w.length - 1]);
  return out;
};

export const curve = (a: Pt, g: Pt, off: number, upto?: number | null, nn?: number): Pt[] => {
  const n = nn || 48;
  const m = Math.max(2, Math.round(n * (upto == null ? 1 : upto)));
  const out: Pt[] = [];
  for (let i = 0; i <= m; i++) {
    const t = i / n;
    out.push({
      x: a.x + (g.x - a.x) * smooth(t) + off * Math.pow(Math.sin(Math.PI * t), 2),
      z: a.z + (g.z - a.z) * t,
    });
  }
  return out;
};
