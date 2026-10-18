import { memo } from 'react';
import { MAT } from './palette';

type Part = {
  pos: [number, number, number];
  size: [number, number, number];
  color: string;
  emissive?: number;
};

const PARTS: Part[] = [
  { pos: [0, 0.55, -0.1], size: [1.2, 0.6, 1.9], color: MAT.fork },
  { pos: [0, 0.7, -1.1], size: [1.2, 0.8, 0.5], color: MAT.fork },
  { pos: [0, 1.0, -0.45], size: [0.5, 0.5, 0.3], color: MAT.dark },
  { pos: [-0.52, 1.5, -0.35], size: [0.08, 1.2, 0.08], color: MAT.dark },
  { pos: [0.52, 1.5, -0.35], size: [0.08, 1.2, 0.08], color: MAT.dark },
  { pos: [-0.52, 1.5, 0.45], size: [0.08, 1.2, 0.08], color: MAT.dark },
  { pos: [0.52, 1.5, 0.45], size: [0.08, 1.2, 0.08], color: MAT.dark },
  { pos: [0, 2.12, 0.05], size: [1.16, 0.06, 1.1], color: MAT.dark },
  { pos: [-0.28, 1.1, 1.05], size: [0.1, 2.2, 0.1], color: MAT.dark },
  { pos: [0.28, 1.1, 1.05], size: [0.1, 2.2, 0.1], color: MAT.dark },
  { pos: [0, 0.7, 1.12], size: [1.1, 0.8, 0.06], color: MAT.dark },
  { pos: [-0.3, 0.1, 1.75], size: [0.1, 0.06, 1.4], color: MAT.dark },
  { pos: [0.3, 0.1, 1.75], size: [0.1, 0.06, 1.4], color: MAT.dark },
  { pos: [0, 0.22, 1.75], size: [1.1, 0.14, 1.2], color: MAT.wood },
  { pos: [0, 0.65, 1.7], size: [0.9, 0.6, 0.9], color: MAT.crate },
  { pos: [-0.58, 0.3, 0.55], size: [0.2, 0.6, 0.6], color: MAT.dark },
  { pos: [0.58, 0.3, 0.55], size: [0.2, 0.6, 0.6], color: MAT.dark },
  { pos: [-0.58, 0.25, -0.8], size: [0.2, 0.5, 0.5], color: MAT.dark },
  { pos: [0.58, 0.25, -0.8], size: [0.2, 0.5, 0.5], color: MAT.dark },
  { pos: [0, 2.25, 0.05], size: [0.12, 0.1, 0.12], color: '#F2A33A', emissive: 2 },
];

type Props = {
  x: number;
  z: number;
  yaw: number;
  scale?: number;
};

export const Forklift = memo(function Forklift({ x, z, yaw, scale = 1.1 }: Props) {
  return (
    <group position={[x, 0, z]} rotation={[0, yaw, 0]} scale={scale}>
      {PARTS.map((p, i) => (
        <mesh key={i} position={p.pos} castShadow receiveShadow>
          <boxGeometry args={p.size} />
          <meshStandardMaterial
            color={p.color}
            roughness={0.72}
            metalness={0.1}
            emissive={p.emissive ? p.color : '#000000'}
            emissiveIntensity={p.emissive ?? 0}
          />
        </mesh>
      ))}
      <pointLight position={[0, 2.4, 0]} color="#F2A33A" intensity={1.2} distance={6} decay={2} />
    </group>
  );
});
