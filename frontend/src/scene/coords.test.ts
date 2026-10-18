import { describe, expect, it } from 'vitest';
import {
  toBackend,
  toBackendHeading,
  toScene,
  toSceneGround,
  toScenePath,
  toSceneRover,
  toSceneYaw,
} from './coords';

describe('backend to scene coordinates', () => {
  it('maps (x, y, z) z-up onto (x, z, -y) y-up', () => {
    expect(toScene(1, 2, 3)).toEqual({ x: 1, y: 3, z: -2 });
  });

  it('round-trips a point', () => {
    const backend: [number, number, number] = [4.5, -8.25, 1.75];
    expect(toBackend(toScene(...backend))).toEqual(backend);
  });

  it('puts backend +y (forward down the hall) at scene -z', () => {
    expect(toScene(0, 10, 0)).toEqual({ x: 0, y: 0, z: -10 });
  });

  it('keeps the floor plane flat', () => {
    expect(toScene(3, 7, 0).y).toBe(0);
  });

  it('negates heading for the flipped ground axis', () => {
    expect(toSceneYaw(Math.PI / 2)).toBeCloseTo(-Math.PI / 2);
    expect(toBackendHeading(toSceneYaw(0.63))).toBeCloseTo(0.63);
  });

  it('converts ground points and paths', () => {
    expect(toSceneGround(2, 5)).toEqual({ x: 2, z: -5 });
    expect(toScenePath([{ x: 1, y: 2 }, { x: 3, y: 4 }])).toEqual([
      { x: 1, z: -2 },
      { x: 3, z: -4 },
    ]);
  });

  it('converts a rover state', () => {
    expect(
      toSceneRover({ x: 2, y: 9, z: 0.4, heading: 1.2, speed: 1.4, steering: 0 }),
    ).toEqual({ x: 2, z: -9, yaw: -1.2 });
  });
});
