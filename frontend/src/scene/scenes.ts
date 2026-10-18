import { curve, type Pt } from './math';

export type PathSpec = {
  k: 'cand' | 'sel' | 'dim' | 'fb';
  off?: number;
  upto?: number;
  a?: number;
  pts?: Pt[];
  wps?: number[][];
  label?: string;
  at?: number;
};

export type WorldState = {
  rov: { x: number; z: number; yaw: number };
  goal: Pt | null;
  ring: 'ok' | 'warn' | 'stop';
  beacon: string;
  fork: { x: number; z: number; yaw: number } | null;
  hazard?: { z: number; x0: number; x1: number };
  contact?: Pt;
  cursor?: boolean;
  speed: number;
  paths: PathSpec[];
};

export type WorldStateName =
  | 'hero'
  | 'goal'
  | 'intervention'
  | 'estop'
  | 'replan'
  | 'replayOff'
  | 'replayOn';

export const SCENES: Record<WorldStateName, WorldState> = {
  hero: {
    rov: { x: 0.2, z: 30, yaw: 0.02 },
    goal: { x: 2.6, z: 60 },
    ring: 'ok',
    beacon: '#5B8DEF',
    fork: null,
    speed: 1.4,
    paths: [
      { k: 'cand', off: -6.5 },
      { k: 'cand', off: -3.4 },
      { k: 'cand', off: 3.2 },
      { k: 'cand', off: 6.2 },
      { k: 'sel', off: -1.2 },
    ],
  },
  goal: {
    rov: { x: -1.0, z: 26, yaw: 0.05 },
    goal: { x: 4.0, z: 54 },
    ring: 'ok',
    beacon: '#5B8DEF',
    fork: null,
    cursor: true,
    speed: 1.1,
    paths: [
      { k: 'cand', off: -6.0, upto: 0.45, a: 0.4 },
      { k: 'cand', off: -3.2, upto: 0.6, a: 0.5 },
      { k: 'cand', off: -0.8, upto: 0.72, a: 0.6 },
      { k: 'cand', off: 2.2, upto: 0.85, a: 0.7 },
      { k: 'cand', off: 5.6, upto: 1, a: 0.8 },
    ],
  },
  intervention: {
    rov: { x: 0.5, z: 40, yaw: -0.07 },
    goal: { x: 2.2, z: 70 },
    ring: 'warn',
    beacon: '#F2A33A',
    fork: { x: -0.4, z: 51.2, yaw: Math.PI / 2 },
    hazard: { z: 51.2, x0: -0.4, x1: 10 },
    speed: 0.6,
    paths: [
      { k: 'cand', off: -5.5 },
      { k: 'cand', off: 4.2 },
      { k: 'sel', off: 0.3 },
    ],
  },
  estop: {
    rov: { x: 0.4, z: 44.6, yaw: -0.04 },
    goal: { x: 2.2, z: 70 },
    ring: 'stop',
    beacon: '#E5484D',
    fork: { x: 0.9, z: 50.6, yaw: Math.PI / 2 },
    speed: 0,
    paths: [
      { k: 'dim', off: -5.5 },
      { k: 'dim', off: 4.2 },
      { k: 'dim', off: 0.3 },
    ],
  },
  replan: {
    rov: { x: 0.4, z: 44.6, yaw: -0.04 },
    goal: { x: 2.6, z: 66 },
    ring: 'ok',
    beacon: '#5B8DEF',
    fork: { x: 1.8, z: 50.6, yaw: Math.PI / 2 },
    speed: 0,
    paths: [
      {
        k: 'fb',
        wps: [[0.4, 44.6], [-1.6, 45.8], [-6.4, 48.4], [-7, 53], [-5, 58], [-0.6, 63], [2.6, 66]],
        label: 'fallback',
        at: 0.6,
      },
      {
        k: 'fb',
        wps: [[0.4, 44.6], [3, 45.6], [9, 47.8], [9.6, 53], [7, 58.4], [4, 63], [2.6, 66]],
        label: 'fallback',
        at: 0.58,
      },
      { k: 'cand', wps: [[0.4, 44.6], [2.4, 46.2], [6.4, 48.6], [6.8, 52], [5.2, 56], [3.4, 61], [2.6, 66]] },
      { k: 'sel', wps: [[0.4, 44.6], [0, 46.4], [-2.8, 48.8], [-3, 51.6], [-1.2, 55], [1.4, 60], [2.6, 66]] },
    ],
  },
  replayOff: {
    rov: { x: 0.35, z: 48.95, yaw: -0.03 },
    goal: { x: 2.6, z: 70 },
    ring: 'stop',
    beacon: '#E5484D',
    fork: { x: 0.6, z: 50.8, yaw: Math.PI / 2 },
    contact: { x: 0.5, z: 50.1 },
    speed: 0,
    paths: [{ k: 'sel', off: 0.2 }],
  },
  replayOn: {
    rov: { x: 0.2, z: 46.6, yaw: -0.05 },
    goal: { x: 2.6, z: 70 },
    ring: 'warn',
    beacon: '#F2A33A',
    fork: { x: 0.6, z: 50.8, yaw: Math.PI / 2 },
    speed: 0.3,
    paths: [
      { k: 'cand', off: 4.0 },
      { k: 'sel', wps: [[0.2, 46.6], [-0.8, 47.8], [-3.2, 49.4], [-3.4, 52.4], [-1.2, 56], [1.6, 62], [2.6, 70]] },
    ],
  },
};

export const LOOP_MS = 44000;
export const HOLD = 0.1;
export const RAMP = 0.12;

export const DRIVE = { start: { x: 0.2, z: 14 }, goal: { x: 2.6, z: 62 }, off: -1.2 };

export const DRIVE_PTS = curve(DRIVE.start, DRIVE.goal, DRIVE.off, 1, 240);

export type Track = {
  pts: Pt[];
  len: number;
  at: (s: number) => { x: number; z: number; i: number };
};

export function buildTrack(pts: Pt[]): Track {
  const cum = [0];
  for (let i = 1; i < pts.length; i++) {
    cum.push(cum[i - 1] + Math.hypot(pts[i].x - pts[i - 1].x, pts[i].z - pts[i - 1].z));
  }
  const len = cum[cum.length - 1] || 1;
  const at = (s: number) => {
    const d = Math.max(0, Math.min(len, s));
    let lo = 0;
    let hi = cum.length - 1;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (cum[mid] <= d) lo = mid;
      else hi = mid;
    }
    const t = (d - cum[lo]) / Math.max(1e-6, cum[hi] - cum[lo]);
    return {
      x: pts[lo].x + (pts[hi].x - pts[lo].x) * t,
      z: pts[lo].z + (pts[hi].z - pts[lo].z) * t,
      i: lo,
    };
  };
  return { pts, len, at };
}

export const DRIVE_TRACK = buildTrack(DRIVE_PTS);

export const driveDist = (p: number, len: number) => {
  const u = Math.min(1, p / (1 - HOLD));
  const a = RAMP;
  const vm = 1 / (1 - a);
  const e =
    u < a
      ? (vm * u * u) / (2 * a)
      : u > 1 - a
        ? 1 - (vm * (1 - u) * (1 - u)) / (2 * a)
        : vm * (a / 2 + (u - a));
  return len * e;
};

export const liveAt = (t: number) => {
  const p = (t % LOOP_MS) / LOOP_MS;
  if (p >= 1 - HOLD) return { chip: 'goal' as const, speed: '0.0' };
  const u = p / (1 - HOLD);
  const prof = u < RAMP ? u / RAMP : u > 1 - RAMP ? (1 - u) / RAMP : 1;
  return { chip: 'tracking' as const, speed: (1.4 * prof).toFixed(1) };
};
