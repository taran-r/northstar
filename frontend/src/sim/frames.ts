import type { Obstacle } from '../scene/Obstacles';
import { OBSTACLE_R } from '../scene/palette';
import type { Pt } from '../scene/math';
import { SCENES, type WorldStateName } from '../scene/scenes';
import {
  SCHEMA_VERSION,
  type ControllerMode,
  type ControllerState,
  type ObstacleState,
  type PathPoint,
  type RunMetrics,
  type StateFrame,
} from '../types/sim';

type ScenePreset = {
  controller: ControllerState;
  metrics: RunMetrics;
};

const EMPTY_METRICS: RunMetrics = {
  collisions: null,
  min_clearance: null,
  replans: null,
  interventions: null,
  path_efficiency: null,
  goal_reached: null,
};

const PRESETS: Record<string, ScenePreset> = {
  tracking: {
    controller: {
      mode: 'tracking',
      nominal: { a: 0.62, steer: 0.03 },
      actual: { a: 0.62, steer: 0.03 },
      barrier_min: 1.84,
      latency_ms: null,
      source: 'learned',
    },
    metrics: EMPTY_METRICS,
  },
  intervention: {
    controller: {
      mode: 'intervention',
      nominal: { a: 0.82, steer: 0.04 },
      actual: { a: 0.31, steer: -0.18 },
      barrier_min: 0.12,
      latency_ms: 4.2,
      source: 'learned',
    },
    metrics: { ...EMPTY_METRICS, min_clearance: 1.1, interventions: 6 },
  },
  estop: {
    controller: {
      mode: 'estop',
      nominal: { a: 0.44, steer: -0.02 },
      actual: { a: -1, steer: 0 },
      barrier_min: 0,
      latency_ms: 3.8,
      source: 'learned',
    },
    metrics: { ...EMPTY_METRICS, min_clearance: 0.9 },
  },
  replanning: {
    controller: {
      mode: 'replanning',
      nominal: { a: 0, steer: 0 },
      actual: { a: 0, steer: 0 },
      barrier_min: 0.4,
      latency_ms: 38,
      source: 'learned',
    },
    metrics: { ...EMPTY_METRICS, replans: 1 },
  },
  goal: {
    controller: {
      mode: 'goal',
      nominal: { a: 0, steer: 0 },
      actual: { a: 0, steer: 0 },
      barrier_min: null,
      latency_ms: null,
      source: 'learned',
    },
    metrics: EMPTY_METRICS,
  },
};

const toPath = (pts: Pt[]): PathPoint[] => pts.map((p) => ({ x: p.x, y: -p.z }));

function obstacleStates(items: Obstacle[]): ObstacleState[] {
  return items.map((o, i) => {
    const r = OBSTACLE_R[o.t] ?? 1.5;
    return {
      id: `${o.t}-${i}`,
      kind: o.t,
      position: [o.x, -o.z, 0],
      velocity: [0, 0, 0],
      shape: 'box',
      size: [r * 2, r * 2, 1.6],
      yaw: o.yaw ?? 0,
    };
  });
}

export type FrameInput = {
  world: WorldStateName;
  mode: ControllerMode;
  seed: number;
  speed: number;
  obstacles: Obstacle[];
  route: Pt[] | null;
  candidates: Pt[][];
  simTime: number;
};

export function frameFor(input: FrameInput): StateFrame {
  const scene = SCENES[input.world] ?? SCENES.hero;
  const preset = PRESETS[input.mode] ?? PRESETS.tracking;
  const rover = scene.rov;
  const selected = input.route ?? [];

  return {
    type: 'state',
    version: SCHEMA_VERSION,
    tick: Math.round(input.simTime * 50),
    sim_time: input.simTime,
    seed: input.seed,
    rover: {
      x: rover.x,
      y: -rover.z,
      z: 0,
      heading: -rover.yaw,
      speed: input.speed,
      steering: preset.controller.actual.steer,
    },
    goal: scene.goal ? [scene.goal.x, -scene.goal.z, 0] : null,
    obstacles: obstacleStates(input.obstacles),
    candidate_paths: input.candidates.map(toPath),
    selected_path: toPath(selected),
    fallback_paths: [],
    executed_trail: [],
    controller: preset.controller,
    metrics: preset.metrics,
    measured: false,
  };
}

export const formatSigned = (v: number) => (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(2);
export const formatUnit = (v: number | null, digits = 2) => {
  if (v == null || Number.isNaN(v)) return '—';
  const text = v.toFixed(digits);
  return text.replace(/^-0(\.0+)?$/, '0$1');
};
