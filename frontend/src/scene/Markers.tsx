import { useMemo, useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { clamp, smooth, type Pt } from './math';
import { useReducedMotion } from '../hooks/useMediaQuery';
import type { PathSpec, WorldState } from './scenes';

const FLAT = [-Math.PI / 2, 0, 0] as [number, number, number];

export function SafetyRing({
  position,
  state,
  radius = 2.1,
}: {
  position: [number, number, number];
  state: WorldState['ring'];
  radius?: number;
}) {
  const ring = useRef<THREE.Group>(null);
  const scan = useRef<THREE.Mesh>(null);
  const still = useReducedMotion();
  const color = state === 'stop' ? '#E5484D' : '#F2A33A';
  const fill = state === 'ok' ? 0.06 : 0.16;
  const edge = state === 'ok' ? 0.45 : 0.95;

  useFrame(({ clock }) => {
    if (still) return;
    const t = clock.elapsedTime;
    if (ring.current) {
      if (state === 'stop') {
        const a = 0.7 + 0.3 * Math.sin(t * 7);
        ring.current.scale.setScalar(1);
        (ring.current.children as THREE.Mesh[]).forEach((m) => {
          const mat = m.material as THREE.Material;
          mat.opacity = (m.userData.base as number) * a;
        });
      } else {
        const s = 1 + 0.035 * Math.sin(t * 2);
        ring.current.scale.setScalar(s);
      }
    }
    if (scan.current) {
      const u = (t % 3.4) / 3.4;
      scan.current.scale.setScalar(0.35 + u * 2.35);
      (scan.current.material as THREE.MeshBasicMaterial).opacity = 0.55 * (1 - u);
    }
  });

  return (
    <group position={position}>
      <group ref={ring}>
        <mesh rotation={FLAT} userData={{ base: fill }}>
          <circleGeometry args={[radius, 64]} />
          <meshBasicMaterial color={color} transparent opacity={fill} depthWrite={false} />
        </mesh>
        <mesh rotation={FLAT} position={[0, 0.002, 0]} userData={{ base: edge }}>
          <ringGeometry args={[radius - (state === 'ok' ? 0.04 : 0.07), radius, 64]} />
          <meshBasicMaterial color={color} transparent opacity={edge} depthWrite={false} />
        </mesh>
      </group>
      {state !== 'stop' && (
        <mesh ref={scan} rotation={FLAT} position={[0, 0.003, 0]}>
          <ringGeometry args={[radius - 0.05, radius, 48]} />
          <meshBasicMaterial color="#9ec3ff" transparent opacity={0.4} depthWrite={false} />
        </mesh>
      )}
      {state !== 'ok' && (
        <mesh rotation={FLAT} position={[0, 0.002, 0]}>
          <ringGeometry args={[radius * 1.3 - 0.03, radius * 1.3, 64]} />
          <meshBasicMaterial color={color} transparent opacity={0.35} depthWrite={false} />
        </mesh>
      )}
    </group>
  );
}

export function GoalMarker({
  position,
  radius = 1.4,
  cursor = false,
}: {
  position: [number, number, number];
  radius?: number;
  cursor?: boolean;
}) {
  const ripple = useRef<THREE.Mesh>(null);
  const still = useReducedMotion();

  useFrame(({ clock }) => {
    if (!ripple.current || still) return;
    const u = (clock.elapsedTime % 2.6) / 2.6;
    ripple.current.scale.setScalar(0.6 + u * 1.3);
    (ripple.current.material as THREE.MeshBasicMaterial).opacity = 0.9 * (1 - u);
  });

  return (
    <group position={position}>
      <mesh rotation={FLAT}>
        <circleGeometry args={[radius, 48]} />
        <meshBasicMaterial color="#4CAF7D" transparent opacity={0.16} depthWrite={false} />
      </mesh>
      <mesh rotation={FLAT} position={[0, 0.002, 0]}>
        <ringGeometry args={[radius - 0.07, radius, 48]} />
        <meshBasicMaterial color="#4CAF7D" transparent opacity={0.95} depthWrite={false} />
      </mesh>
      <mesh rotation={FLAT} position={[0, 0.003, 0]}>
        <circleGeometry args={[radius * 0.45, 32]} />
        <meshBasicMaterial color="#4CAF7D" transparent opacity={0.4} depthWrite={false} />
      </mesh>
      <mesh ref={ripple} rotation={FLAT} position={[0, 0.004, 0]}>
        <ringGeometry args={[radius - 0.04, radius, 48]} />
        <meshBasicMaterial color="#4CAF7D" transparent opacity={0.8} depthWrite={false} />
      </mesh>
      <mesh position={[0, 3, 0]}>
        <cylinderGeometry args={[radius * 0.12, radius * 0.3, 6, 16, 1, true]} />
        <meshBasicMaterial
          color="#4CAF7D"
          transparent
          opacity={0.22}
          side={THREE.DoubleSide}
          depthWrite={false}
          blending={THREE.AdditiveBlending}
        />
      </mesh>
      <pointLight position={[0, 1.2, 0]} color="#4CAF7D" intensity={6} distance={10} decay={2} />
      {cursor && (
        <group>
          <mesh rotation={FLAT} position={[0, 0.005, 0]}>
            <ringGeometry args={[radius * 1.33, radius * 1.35, 48]} />
            <meshBasicMaterial color="#E8E6E1" transparent opacity={0.8} depthWrite={false} />
          </mesh>
          {[0, Math.PI / 2, Math.PI, -Math.PI / 2].map((a, i) => (
            <mesh
              key={i}
              rotation={FLAT}
              position={[Math.cos(a) * radius * 1.57, 0.005, Math.sin(a) * radius * 1.57]}
            >
              <planeGeometry
                args={i % 2 ? [0.08, radius * 0.45] : [radius * 0.45, 0.08]}
              />
              <meshBasicMaterial color="#E8E6E1" transparent opacity={0.9} depthWrite={false} />
            </mesh>
          ))}
        </group>
      )}
    </group>
  );
}

const BURST_RINGS = 3;
const BURST_GAP_S = 0.28;
const BURST_S = 1.4;

/** One-shot arrival burst over the goal. Mount it when the goal is reached. */
export function GoalCelebration({
  position,
  radius = 1.4,
}: {
  position: [number, number, number];
  radius?: number;
}) {
  const rings = useRef<(THREE.Mesh | null)[]>([]);
  const beam = useRef<THREE.Mesh>(null);
  const light = useRef<THREE.PointLight>(null);
  const start = useRef<number | null>(null);
  const still = useReducedMotion();

  useFrame(({ clock }) => {
    if (start.current == null) start.current = clock.elapsedTime;
    const t = still ? 10 : clock.elapsedTime - start.current;
    rings.current.forEach((m, i) => {
      if (!m) return;
      const u = clamp((t - i * BURST_GAP_S) / BURST_S, 0, 1);
      m.visible = u > 0 && u < 1;
      m.scale.setScalar(1 + u * 3);
      (m.material as THREE.MeshBasicMaterial).opacity = 0.9 * (1 - u) * (1 - u);
    });
    // Flash, then settle to a steady "locked in" glow.
    const flash = Math.exp(-t * 1.6);
    if (beam.current) {
      (beam.current.material as THREE.MeshBasicMaterial).opacity = 0.32 + 0.5 * flash;
    }
    if (light.current) light.current.intensity = 8 + 30 * flash;
  });

  return (
    <group position={position}>
      {Array.from({ length: BURST_RINGS }, (_, i) => (
        <mesh
          key={i}
          ref={(m) => {
            rings.current[i] = m;
          }}
          rotation={FLAT}
          position={[0, 0.006 + i * 0.001, 0]}
          visible={false}
        >
          <ringGeometry args={[radius - 0.09, radius, 64]} />
          <meshBasicMaterial color="#7FE0A8" transparent opacity={0} depthWrite={false} />
        </mesh>
      ))}
      <mesh rotation={FLAT} position={[0, 0.005, 0]}>
        <ringGeometry args={[radius * 1.08, radius * 1.16, 64]} />
        <meshBasicMaterial color="#4CAF7D" transparent opacity={0.9} depthWrite={false} />
      </mesh>
      <mesh ref={beam} position={[0, 4, 0]}>
        <cylinderGeometry args={[radius * 0.18, radius * 0.4, 8, 20, 1, true]} />
        <meshBasicMaterial
          color="#7FE0A8"
          transparent
          opacity={0.4}
          side={THREE.DoubleSide}
          depthWrite={false}
          blending={THREE.AdditiveBlending}
        />
      </mesh>
      <pointLight ref={light} position={[0, 1.6, 0]} color="#7FE0A8" intensity={8} distance={14} decay={2} />
    </group>
  );
}

export function HazardBand({ z, x0, x1 }: { z: number; x0: number; x1: number }) {
  return (
    <mesh position={[(x0 + x1) / 2, 0.016, z]} rotation={FLAT}>
      <planeGeometry args={[Math.abs(x1 - x0), 2.4]} />
      <meshBasicMaterial color="#F2A33A" transparent opacity={0.12} depthWrite={false} />
    </mesh>
  );
}

export function ContactMark({ position }: { position: [number, number, number] }) {
  const mesh = useRef<THREE.Mesh>(null);
  const still = useReducedMotion();
  useFrame(({ clock }) => {
    if (!mesh.current || still) return;
    const a = 0.55 + 0.45 * Math.sin(clock.elapsedTime * 7);
    (mesh.current.material as THREE.MeshBasicMaterial).opacity = a;
  });
  return (
    <group position={position}>
      <mesh ref={mesh} rotation={FLAT}>
        <circleGeometry args={[1.1, 32]} />
        <meshBasicMaterial color="#E5484D" transparent opacity={0.45} depthWrite={false} />
      </mesh>
      <mesh rotation={FLAT} position={[0, 0.002, 0]}>
        <circleGeometry args={[0.55, 24]} />
        <meshBasicMaterial color="#E5484D" transparent opacity={0.7} depthWrite={false} />
      </mesh>
    </group>
  );
}

export function AttentionTiles({
  rover,
  goal,
  selected,
  fork,
}: {
  rover: Pt;
  goal: Pt;
  selected?: PathSpec;
  fork?: { x: number; z: number } | null;
}) {
  const tiles = useMemo(() => {
    let ts = 31;
    const rnd = () => {
      ts = (ts * 16807) % 2147483647;
      return ts / 2147483647;
    };
    const pathX = (z: number) => {
      if (selected?.pts) {
        for (let i = 0; i < selected.pts.length - 1; i++) {
          if (selected.pts[i + 1].z >= z) return selected.pts[i].x;
        }
        return goal.x;
      }
      const t = clamp((z - rover.z) / Math.max(1, goal.z - rover.z), 0, 1);
      return (
        rover.x +
        (goal.x - rover.x) * smooth(t) +
        (selected?.off ?? 0) * Math.pow(Math.sin(Math.PI * t), 2)
      );
    };
    const out: { x: number; z: number; w: number }[] = [];
    for (let z = rover.z + 2.5; z < rover.z + 24; z += 1.6) {
      for (let x = -7.8; x < 7.8; x += 1.3) {
        const dx = x - pathX(z);
        let w = Math.exp(-(dx * dx) / (2 * 1.7 * 1.7)) * (1 - (z - rover.z) / 30) * 0.8 + rnd() * 0.12;
        if (fork) {
          const fx = x - fork.x;
          const fz = z - fork.z;
          w += 0.8 * Math.exp(-(fx * fx + fz * fz) / (2 * 2.4 * 2.4));
        }
        if (x > 6.6 && z > rover.z + 6 && z < rover.z + 16) w += 0.35 * rnd();
        if (w < 0.16) continue;
        out.push({ x: x + 0.575, z: z + 0.7, w });
      }
    }
    return out;
  }, [rover.x, rover.z, goal.x, goal.z, selected, fork]);

  return (
    <group>
      {tiles.map((t, i) => (
        <mesh key={i} position={[t.x, 0.018, t.z]} rotation={FLAT}>
          <planeGeometry args={[1.15, 1.4]} />
          <meshBasicMaterial
            color={t.w > 0.6 ? '#F2A33A' : '#5B8DEF'}
            transparent
            opacity={clamp(t.w * 0.5, 0.06, 0.42)}
            depthWrite={false}
          />
        </mesh>
      ))}
    </group>
  );
}
