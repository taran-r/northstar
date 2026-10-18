import { Fragment, memo, useMemo } from 'react';
import { MAT, OBSTACLE_R } from './palette';
import { Forklift } from './Forklift';
import { MovingForklift } from './MovingForklift';
import { patrolFor } from './agents';
import type { Pose } from './WorldCanvas';
import type { SimClock } from '../hooks/useSimClock';

export type Obstacle = {
  t: string;
  x: number;
  z: number;
  yaw?: number;
  id?: string;
  length?: number;
  width?: number;
  height?: number;
};

type Part = { pos: [number, number, number]; size: [number, number, number]; color: string };

const OS = 1.2;

const PALLET: Part[] = [
  ...[0.08, 0.24, 0.4, 0.56].map(
    (y): Part => ({ pos: [0, y, 0], size: [1.24, 0.16, 1.0], color: MAT.wood }),
  ),
  { pos: [-0.18, 0.84, 0.05], size: [0.6, 0.4, 0.64], color: MAT.crate },
  { pos: [0.26, 0.8, -0.1], size: [0.48, 0.32, 0.52], color: MAT.tote },
];

const CRATES: Part[] = [[-0.5, -0.45], [0.5, -0.45], [-0.5, 0.45], [0.5, 0.45]].flatMap(
  (q, k): Part[] => {
    const lower: Part = {
      pos: [q[0], 0.25, q[1]],
      size: [0.9, 0.5, 0.8],
      color: k % 3 === 1 ? MAT.tote : MAT.crate,
    };
    if (k === 2) return [lower];
    return [
      lower,
      { pos: [q[0], 0.75, q[1]], size: [0.9, 0.5, 0.8], color: k % 2 ? MAT.crate : MAT.tote },
    ];
  },
);

const BARRIER: Part[] = [
  { pos: [0, 0.32, 0], size: [2.8, 0.6, 0.44], color: MAT.guard },
  ...[-0.95, 0, 0.95].map((x): Part => ({ pos: [x, 0.32, 0], size: [0.36, 0.62, 0.45], color: MAT.dark })),
  ...[-1.1, 1.1].map((x): Part => ({ pos: [x, 0.08, 0], size: [0.16, 0.16, 0.76], color: MAT.dark })),
];

const SHAPES: Record<string, Part[]> = { pallet: PALLET, crates: CRATES, barrier: BARRIER };

const MEASURED_COLOR: Record<string, string> = {
  pallet: MAT.wood,
  crates: MAT.crate,
  barrier: MAT.guard,
  forklift: MAT.fork,
};

function MeasuredBody({ item }: { item: Obstacle }) {
  const length = item.length ?? 1;
  const width = item.width ?? 1;
  const height = item.height ?? 1;
  const color = MEASURED_COLOR[item.t] ?? MAT.steel;
  return (
    <group position={[item.x, 0, item.z]} rotation={[0, item.yaw ?? 0, 0]}>
      <mesh position={[0, height / 2, 0]} castShadow receiveShadow>
        <boxGeometry args={[width, height, length]} />
        <meshStandardMaterial color={color} roughness={0.72} metalness={0.08} />
      </mesh>
      {item.t === 'forklift' && (
        <mesh position={[0, height + 0.08, 0]}>
          <boxGeometry args={[0.12, 0.12, 0.12]} />
          <meshStandardMaterial color="#F2A33A" emissive="#F2A33A" emissiveIntensity={1.4} />
        </mesh>
      )}
    </group>
  );
}

function SelectionRing({ x, z, r }: { x: number; z: number; r: number }) {
  return (
    <group position={[x, 0.04, z]}>
      <mesh rotation={[-Math.PI / 2, 0, 0]}>
        <circleGeometry args={[r * 1.15, 56]} />
        <meshBasicMaterial color="#F2A33A" transparent opacity={0.12} depthWrite={false} />
      </mesh>
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.002, 0]}>
        <ringGeometry args={[r * 1.15 - 0.08, r * 1.15, 56]} />
        <meshBasicMaterial color="#F2A33A" transparent opacity={0.95} depthWrite={false} />
      </mesh>
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.002, 0]}>
        <ringGeometry args={[r * 1.45 - 0.04, r * 1.45, 56]} />
        <meshBasicMaterial color="#F2A33A" transparent opacity={0.45} depthWrite={false} />
      </mesh>
    </group>
  );
}

export const Obstacles = memo(function Obstacles({
  items,
  selected,
  onPick,
  clock,
  rover,
  animate = false,
  runId = 0,
}: {
  items: Obstacle[];
  selected: number;
  onPick?: (index: number) => void;
  clock?: React.RefObject<SimClock>;
  rover?: React.RefObject<Pose> | null;
  animate?: boolean;
  runId?: number;
}) {
  const patrols = useMemo(() => items.map((o, i) => patrolFor(o, i)), [items]);
  return (
    <group>
      {items.map((o, i) => {
        const yaw = o.yaw ?? 0;
        const r = o.length ? Math.max(o.length, o.width ?? 1) * 0.65 : (OBSTACLE_R[o.t] ?? 1.5);
        const measured = o.length != null && o.width != null;
        return (
          <Fragment key={o.id ?? i}>
            {measured ? (
              <group
                onClick={(e) => {
                  e.stopPropagation();
                  onPick?.(i);
                }}
              >
                <MeasuredBody item={o} />
              </group>
            ) : o.t === 'forklift' ? (
              <group
                onClick={(e) => {
                  e.stopPropagation();
                  onPick?.(i);
                }}
              >
                {animate ? (
                  <MovingForklift
                    key={`${runId}-${i}`}
                    spec={patrols[i]}
                    rover={rover}
                    clock={clock}
                    scale={1.3}
                  />
                ) : (
                  <Forklift x={o.x} z={o.z} yaw={yaw} scale={1.3} />
                )}
              </group>
            ) : (
              <group
                position={[o.x, 0, o.z]}
                rotation={[0, yaw, 0]}
                scale={OS}
                onClick={(e) => {
                  e.stopPropagation();
                  onPick?.(i);
                }}
              >
                {(SHAPES[o.t] ?? PALLET).map((p, k) => (
                  <mesh key={k} position={p.pos} castShadow receiveShadow>
                    <boxGeometry args={p.size} />
                    <meshStandardMaterial color={p.color} roughness={0.78} metalness={0.08} />
                  </mesh>
                ))}
              </group>
            )}
            {i === selected && <SelectionRing x={o.x} z={o.z} r={r} />}
          </Fragment>
        );
      })}
    </group>
  );
});
