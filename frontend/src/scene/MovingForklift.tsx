import { useRef, useState } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { Forklift } from './Forklift';
import { gapTo, segmentLength, type AgentPose, type PatrolSpec } from './agents';
import type { Pose } from './WorldCanvas';
import type { SimClock } from '../hooks/useSimClock';

export function MovingForklift({
  spec,
  scale = 1.1,
  pose,
  rover,
  clock,
  showLane = false,
}: {
  spec: PatrolSpec;
  scale?: number;
  pose?: React.RefObject<AgentPose>;
  rover?: React.RefObject<Pose> | null;
  clock?: React.RefObject<SimClock>;
  showLane?: boolean;
}) {
  const group = useRef<THREE.Group>(null);
  const lane = useRef<THREE.Mesh>(null);
  const [moving, setMoving] = useState(true);
  const lastState = useRef(true);

  const u = useRef(spec.phase);
  const dir = useRef(1);
  const wait = useRef(0);

  useFrame((_, dt) => {
    const running = clock ? clock.current.playing : true;
    const step = running ? Math.min(dt, 0.1) : 0;
    const len = segmentLength(spec);

    const at = (t: number) => ({
      x: spec.a.x + (spec.b.x - spec.a.x) * t,
      z: spec.a.z + (spec.b.z - spec.a.z) * t,
    });

    const next = Math.max(0, Math.min(1, u.current + (dir.current * spec.speed * step) / len));
    const ahead = at(next);
    const holding = rover?.current ? gapTo(ahead, rover.current) < spec.yieldGap : false;

    if (wait.current > 0) {
      wait.current -= step;
    } else if (!holding) {
      u.current = next;
      if (u.current >= 1) {
        u.current = 1;
        dir.current = -1;
        wait.current = spec.dwell;
      } else if (u.current <= 0) {
        u.current = 0;
        dir.current = 1;
        wait.current = spec.dwell;
      }
    }

    const here = at(u.current);
    const tx = (spec.b.x - spec.a.x) * dir.current;
    const tz = (spec.b.z - spec.a.z) * dir.current;
    const yaw = Math.atan2(tx, tz);
    const isMoving = running && wait.current <= 0 && !holding;

    if (pose) pose.current = { ...here, yaw, moving: isMoving, waiting: holding };

    const g = group.current;
    if (g) {
      g.position.set(here.x, 0, here.z);
      g.rotation.y = yaw;
    }
    if (lane.current) {
      const material = lane.current.material as THREE.MeshBasicMaterial;
      material.opacity = isMoving ? 0.13 : 0.04;
    }
    if (isMoving !== lastState.current) {
      lastState.current = isMoving;
      setMoving(isMoving);
    }
  });

  const mid = { x: (spec.a.x + spec.b.x) / 2, z: (spec.a.z + spec.b.z) / 2 };
  const laneYaw = Math.atan2(spec.b.x - spec.a.x, spec.b.z - spec.a.z);

  return (
    <group>
      {showLane && (
        <mesh ref={lane} position={[mid.x, 0.016, mid.z]} rotation={[-Math.PI / 2, 0, -laneYaw]}>
          <planeGeometry args={[2.6, segmentLength(spec) + 3]} />
          <meshBasicMaterial color="#F2A33A" transparent opacity={0.12} depthWrite={false} />
        </mesh>
      )}
      <group ref={group}>
        <Forklift x={0} z={0} yaw={0} scale={scale} />
        {moving && (
          <pointLight position={[0, 2.4, 0]} color="#F2A33A" intensity={2.4} distance={8} decay={2} />
        )}
      </group>
    </group>
  );
}
