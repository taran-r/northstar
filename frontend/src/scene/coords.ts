import type { Pt } from './math';
import type { PathPoint, RoverState, Vec3 } from '../types/sim';

export type ScenePoint = { x: number; y: number; z: number };

export function toScene(x: number, y: number, z: number): ScenePoint {
  return { x, y: z, z: -y };
}

export function toSceneVec(v: Vec3): ScenePoint {
  return toScene(v[0], v[1], v[2]);
}

export function toBackend(p: ScenePoint): Vec3 {
  return [p.x, -p.z, p.y];
}

export function toSceneYaw(heading: number): number {
  return -heading;
}

export function toBackendHeading(yaw: number): number {
  return -yaw;
}

export function toSceneGround(x: number, y: number): Pt {
  return { x, z: -y };
}

export function toScenePath(path: PathPoint[]): Pt[] {
  return path.map((p) => toSceneGround(p.x, p.y));
}

export function toSceneRover(rover: RoverState): { x: number; z: number; yaw: number } {
  const p = toScene(rover.x, rover.y, rover.z);
  return { x: p.x, z: p.z, yaw: toSceneYaw(rover.heading) };
}
