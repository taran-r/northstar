import { useCallback, useMemo, useState } from 'react';
import type { Pt } from '../scene/math';
import { OBSTACLE_R } from '../scene/palette';
import type { Obstacle } from '../scene/Obstacles';

export type ToolId = 'pallet' | 'crates' | 'barrier' | 'forklift';
export type Density = 'sparse' | 'medium' | 'dense';

export const TOOL_NAMES: Record<ToolId, string> = {
  pallet: 'Pallet stack',
  crates: 'Crate wall',
  barrier: 'Barrier',
  forklift: 'Forklift',
};

const COUNT: Record<Density, number> = { sparse: 8, medium: 14, dense: 22 };
const START = { x: -1.5, z: 8 };
const GOAL = { x: 3, z: 86 };
const smooth = (t: number) => t * t * (3 - 2 * t);

function generate(seed: number, n: number): Obstacle[] {
  let a = (seed * 7919 + 13) % 2147483647;
  const r = () => {
    a = (a * 16807) % 2147483647;
    return a / 2147483647;
  };
  const out: Obstacle[] = [];
  let tries = 0;
  while (out.length < n && tries < 600) {
    tries++;
    const q = r();
    const t: ToolId = q < 0.32 ? 'pallet' : q < 0.62 ? 'crates' : q < 0.84 ? 'barrier' : 'forklift';
    const x = -6.4 + r() * 12.8;
    const z = 20 + r() * 58;
    if (Math.hypot(x - GOAL.x, z - GOAL.z) < 8) continue;
    if (out.some((o) => Math.hypot(o.x - x, o.z - z) < 5.4)) continue;
    const quarter = t === 'barrier' || t === 'forklift' ? (r() < 0.5 ? 0 : Math.PI / 2) : 0;
    out.push({
      t,
      x: Math.round(x * 10) / 10,
      z: Math.round(z * 10) / 10,
      yaw: quarter + (r() - 0.5) * 0.5,
    });
  }
  return out;
}

function plan(obs: Obstacle[], bias: number) {
  const N = 40;
  const pts: Pt[] = [];
  for (let i = 0; i <= N; i++) {
    const t = i / N;
    pts.push({
      x: START.x + (GOAL.x - START.x) * smooth(t) + bias * Math.pow(Math.sin(Math.PI * t), 2),
      z: START.z + (GOAL.z - START.z) * t,
    });
  }
  for (let it = 0; it < 40; it++) {
    obs.forEach((o) => {
      const rr = (OBSTACLE_R[o.t] ?? 1.5) + 1.5;
      for (let i = 1; i < N; i++) {
        const p = pts[i];
        const dz = p.z - o.z;
        if (Math.abs(dz) >= rr) continue;
        const need = Math.sqrt(rr * rr - dz * dz);
        const dx = p.x - o.x;
        if (Math.abs(dx) < need) {
          const side = dx === 0 ? (bias >= 0 ? 1 : -1) : Math.sign(dx);
          p.x += (o.x + side * need - p.x) * 0.5;
        }
      }
    });
    for (let i = 1; i < N; i++) {
      pts[i].x = Math.max(-7.4, Math.min(7.4, (pts[i - 1].x + 2 * pts[i].x + pts[i + 1].x) / 4));
    }
  }
  let len = 0;
  let clear = 9.9;
  for (let i = 1; i <= N; i++) len += Math.hypot(pts[i].x - pts[i - 1].x, pts[i].z - pts[i - 1].z);
  obs.forEach((o) =>
    pts.forEach((p) => {
      clear = Math.min(clear, Math.hypot(p.x - o.x, p.z - o.z) - (OBSTACLE_R[o.t] ?? 1.5));
    }),
  );
  let dense = pts;
  for (let k = 0; k < 2; k++) {
    const nx: Pt[] = [dense[0]];
    for (let i = 0; i < dense.length - 1; i++) {
      const a = dense[i];
      const b = dense[i + 1];
      nx.push(
        { x: a.x * 0.75 + b.x * 0.25, z: a.z * 0.75 + b.z * 0.25 },
        { x: a.x * 0.25 + b.x * 0.75, z: a.z * 0.25 + b.z * 0.75 },
      );
    }
    nx.push(dense[dense.length - 1]);
    dense = nx;
  }
  return { pts: dense, len, clear };
}

export type CourseSetup = ReturnType<typeof useCourseSetup>;

export function useCourseSetup() {
  const [seed, setSeed] = useState(427);
  const [density, setDensity] = useState<Density>('medium');
  const [obstacles, setObstacles] = useState<Obstacle[]>(() => generate(427, COUNT.medium));
  const [selected, setSelected] = useState(3);
  const [tool, setTool] = useState<ToolId>('pallet');
  const [routePreview, setRoutePreview] = useState(true);
  const [tilt, setTilt] = useState(52);

  const randomize = useCallback(() => {
    const next = 100 + Math.floor(Math.random() * 8900);
    setSeed(next);
    setObstacles(generate(next, COUNT[density]));
    setSelected(-1);
  }, [density]);

  const pickDensity = useCallback(
    (d: Density) => {
      setDensity(d);
      setObstacles(generate(seed, COUNT[d]));
      setSelected(-1);
    },
    [seed],
  );

  const placeAt = useCallback(
    (pt: Pt) => {
      let hit = -1;
      let best = 1e9;
      obstacles.forEach((o, i) => {
        const d = Math.hypot(o.x - pt.x, o.z - pt.z);
        if (d < (OBSTACLE_R[o.t] ?? 1.5) + 0.8 && d < best) {
          best = d;
          hit = i;
        }
      });
      if (hit >= 0) {
        setSelected(hit);
        return;
      }
      if (pt.x < -8 || pt.x > 8 || pt.z < 14 || pt.z > 82) {
        setSelected(-1);
        return;
      }
      const next = obstacles.concat([
        { t: tool, x: Math.round(pt.x * 10) / 10, z: Math.round(pt.z * 10) / 10, yaw: 0 },
      ]);
      setObstacles(next);
      setSelected(next.length - 1);
    },
    [obstacles, tool],
  );

  const rotate = useCallback(() => {
    setObstacles((list) =>
      list.map((o, i) => (i === selected ? { ...o, yaw: (o.yaw ?? 0) + Math.PI / 2 } : o)),
    );
  }, [selected]);

  const remove = useCallback(() => {
    setObstacles((list) => list.filter((_, i) => i !== selected));
    setSelected(-1);
  }, [selected]);

  const clear = useCallback(() => {
    setObstacles([]);
    setSelected(-1);
  }, []);

  const routing = useMemo(() => {
    const plans = [-6, -3, 0, 3, 6].map((b) => plan(obstacles, b));
    let bi = 0;
    let bs = 1e9;
    plans.forEach((p, i) => {
      const score = p.len + (p.clear < 0.8 ? 80 : 0) - Math.min(p.clear, 2) * 2;
      if (score < bs) {
        bs = score;
        bi = i;
      }
    });
    const best = plans[bi];
    const others = plans.filter((_, i) => i !== bi).map((p) => p.pts);
    return {
      plannedRoute: best.pts,
      plannedCandidates: others,
      route: routePreview ? best.pts : null,
      candidates: routePreview ? others : [],
      length: String(Math.round(best.len)),
      clearance: Math.max(0, best.clear).toFixed(1),
    };
  }, [obstacles, routePreview]);

  const selectedObstacle = selected >= 0 ? obstacles[selected] : null;

  return {
    seed,
    seedText: String(seed).padStart(4, '0'),
    obstacles,
    selected,
    setSelected,
    selectedObstacle,
    tool,
    setTool,
    density,
    pickDensity,
    randomize,
    clear,
    placeAt,
    rotate,
    remove,
    routePreview,
    toggleRoutePreview: () => setRoutePreview((r) => !r),
    tilt,
    setTilt,
    ...routing,
  };
}
