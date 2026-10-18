import { memo, useEffect, useMemo } from 'react';
import * as THREE from 'three';
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js';
import { HALL, MAT } from './palette';
import {
  LANE_DASHES,
  LANE_EDGES,
  WINDOW_Z,
  buildCeilingLights,
  buildHallProps,
  buildTrusses,
  type Prop,
} from './layout';

const { X0, X1, ZE, H } = HALL;

function mergeByColour(parts: Prop[]) {
  const buckets = new Map<string, THREE.BufferGeometry[]>();
  for (const p of parts) {
    const geo = new THREE.BoxGeometry(p.size[0], p.size[1], p.size[2]);
    if (p.yaw) geo.rotateY(p.yaw);
    geo.translate(p.pos[0], p.pos[1], p.pos[2]);
    const list = buckets.get(p.color);
    if (list) list.push(geo);
    else buckets.set(p.color, [geo]);
  }
  const out: { color: string; geometry: THREE.BufferGeometry }[] = [];
  for (const [color, list] of buckets) {
    const merged = mergeGeometries(list, false);
    list.forEach((g) => g.dispose());
    if (merged) out.push({ color, geometry: merged });
  }
  return out;
}

function Stripe({
  x,
  z,
  w,
  l,
  color,
  opacity,
}: {
  x: number;
  z: number;
  w: number;
  l: number;
  color: string;
  opacity: number;
}) {
  return (
    <mesh position={[x, 0.012, z]} rotation={[-Math.PI / 2, 0, 0]}>
      <planeGeometry args={[w, l]} />
      <meshBasicMaterial color={color} transparent opacity={opacity} depthWrite={false} />
    </mesh>
  );
}

export const Hall = memo(function Hall({ detail = 'hi' }: { detail?: 'hi' | 'lo' }) {
  const solids = useMemo(() => {
    const merged = mergeByColour(buildHallProps());
    const bars = buildTrusses().map((b) => {
      const geo = new THREE.BoxGeometry(b.len, b.thick, b.thick);
      geo.rotateZ(b.angle);
      geo.translate(b.pos[0], b.pos[1], b.pos[2]);
      return geo;
    });
    const trussGeo = mergeGeometries(bars, false);
    bars.forEach((g) => g.dispose());
    if (trussGeo) merged.push({ color: '#1a212c', geometry: trussGeo });
    return merged;
  }, []);

  const lights = useMemo(
    () =>
      mergeByColour(
        buildCeilingLights().map((p) => ({ pos: p, size: [0.7, 0.12, 2.8], color: '#f4f8ff' })),
      )[0],
    [],
  );

  useEffect(
    () => () => {
      solids.forEach((s) => s.geometry.dispose());
      lights?.geometry.dispose();
    },
    [solids, lights],
  );

  const gridZ = useMemo(() => {
    const out: number[] = [];
    for (let z = 6; z < ZE; z += 6) out.push(z);
    return out;
  }, []);

  const dashes = useMemo(() => {
    const out: { x: number; z: number; color: string }[] = [];
    for (const lane of LANE_DASHES) {
      for (let z = 8; z < ZE - 2; z += 8) out.push({ x: lane.x, z: z + 2.75, color: lane.color });
    }
    return out;
  }, []);

  return (
    <group>
      <mesh position={[0, 0, ZE / 2]} rotation={[-Math.PI / 2, 0, 0]} receiveShadow>
        <planeGeometry args={[X1 - X0, ZE]} />
        <meshStandardMaterial color={MAT.floor} roughness={0.52} metalness={0.12} />
      </mesh>

      {gridZ.map((z) => (
        <Stripe key={`g${z}`} x={0} z={z} w={X1 - X0} l={0.05} color="#ffffff" opacity={0.06} />
      ))}

      {LANE_EDGES.map((x) => (
        <Stripe key={`e${x}`} x={x} z={ZE / 2} w={0.16} l={ZE} color="#e0c46a" opacity={0.34} />
      ))}
      {dashes.map((d, i) => (
        <Stripe key={`d${i}`} x={d.x} z={d.z} w={0.08} l={5.5} color={d.color} opacity={0.85} />
      ))}

      {[X0, X1].map((x) => (
        <group key={`w${x}`}>
          <mesh position={[x, 2.1, ZE / 2]} rotation={[0, x < 0 ? Math.PI / 2 : -Math.PI / 2, 0]}>
            <planeGeometry args={[ZE, 4.2]} />
            <meshStandardMaterial color={MAT.wall} roughness={0.95} />
          </mesh>
          <mesh
            position={[x, 4.2 + (H - 4.2) / 2, ZE / 2]}
            rotation={[0, x < 0 ? Math.PI / 2 : -Math.PI / 2, 0]}
          >
            <planeGeometry args={[ZE, H - 4.2]} />
            <meshStandardMaterial color={MAT.wallUpper} roughness={0.95} />
          </mesh>
        </group>
      ))}

      <mesh position={[0, H / 2, ZE]} rotation={[0, Math.PI, 0]}>
        <planeGeometry args={[X1 - X0, H]} />
        <meshStandardMaterial color="#4a5c76" roughness={0.95} />
      </mesh>
      <mesh position={[0, 2.7, ZE - 0.1]} rotation={[0, Math.PI, 0]}>
        <planeGeometry args={[8, 5.4]} />
        <meshBasicMaterial color="#b9cdf0" transparent opacity={0.55} />
      </mesh>

      {WINDOW_Z.map((z) => (
        <mesh key={`win${z}`} position={[X1 - 0.05, 7.7, z]} rotation={[0, -Math.PI / 2, 0]}>
          <planeGeometry args={[8, 3]} />
          <meshBasicMaterial color="#ffc98a" />
        </mesh>
      ))}

      {solids.map((s, i) => (
        <mesh
          key={i}
          geometry={s.geometry}
          castShadow={detail === 'hi'}
          receiveShadow={detail === 'hi'}
        >
          <meshStandardMaterial color={s.color} roughness={0.78} metalness={0.08} />
        </mesh>
      ))}

      {lights && (
        <mesh geometry={lights.geometry}>
          <meshStandardMaterial color="#f4f8ff" emissive="#f4f8ff" emissiveIntensity={2.4} />
        </mesh>
      )}
    </group>
  );
});
