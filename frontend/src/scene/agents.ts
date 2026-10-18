import type { Pt } from './math';
import { HALL } from './palette';

export type PatrolSpec = {
  id: string;
  a: Pt;
  b: Pt;
  speed: number;
  dwell: number;
  yieldGap: number;
  phase: number;
};

export type AgentPose = Pt & { yaw: number; moving: boolean; waiting: boolean };

export const CROSSING_FORKLIFT: PatrolSpec = {
  id: 'forklift-staged',
  a: { x: 12.5, z: 51.2 },
  b: { x: -12.5, z: 51.2 },
  speed: 1.6,
  dwell: 5,
  yieldGap: 7,
  phase: 0,
};

const PATROL_REACH = 5.5;
const EDGE = 0.8;

const clampX = (x: number) => Math.max(HALL.X0 + EDGE, Math.min(HALL.X1 - EDGE, x));
const clampZ = (z: number) => Math.max(2, Math.min(HALL.ZE - 2, z));

export function patrolFor(
  obstacle: { x: number; z: number; yaw?: number },
  index: number,
): PatrolSpec {
  const yaw = obstacle.yaw ?? 0;
  const acrossAisle = Math.abs(Math.sin(yaw)) > 0.5;
  const reach = PATROL_REACH;

  const a = acrossAisle
    ? { x: clampX(obstacle.x - reach), z: obstacle.z }
    : { x: obstacle.x, z: clampZ(obstacle.z - reach) };
  const b = acrossAisle
    ? { x: clampX(obstacle.x + reach), z: obstacle.z }
    : { x: obstacle.x, z: clampZ(obstacle.z + reach) };

  return {
    id: `forklift-${index}`,
    a,
    b,
    speed: 0.9 + ((index * 37) % 7) * 0.08,
    dwell: 2.5 + ((index * 53) % 5) * 0.7,
    yieldGap: 6,
    phase: ((index * 97) % 100) / 100,
  };
}

export function segmentLength(spec: PatrolSpec): number {
  return Math.hypot(spec.b.x - spec.a.x, spec.b.z - spec.a.z) || 1;
}

export function gapTo(agent: Pt, rover: Pt): number {
  return Math.hypot(agent.x - rover.x, agent.z - rover.z);
}
