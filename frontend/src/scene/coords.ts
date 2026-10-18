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

/**
 * Yaw for a mesh whose nose is local +Z.
 *
 * Backend heading 0 points along +X. In the scene that velocity is (cos ψ, −sin ψ)
 * on (x, z). A +Z-forward mesh needs atan2 of that pair, which is not the axis
 * flip in `toSceneYaw` — that helper only mirrors the ground plane.
 */
export function toSceneMeshYaw(heading: number): number {
  return Math.atan2(Math.cos(heading), -Math.sin(heading));
}

export function toBackendHeading(yaw: number): number {
  return -yaw;
}

/**
 * Backend heading for a mesh whose nose is local +Z (Forklift.tsx).
 *
 * Setup stores scene yaw for that mesh. The bicycle heading that points the
 * same way is a −π/2 shift of that yaw — the inverse of `toSceneMeshYaw`.
 */
export function toBackendMeshHeading(sceneYaw: number): number {
  return sceneYaw - Math.PI / 2;
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
