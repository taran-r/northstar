import { memo } from 'react';
import { MAT } from './palette';

type Part = {
  pos: [number, number, number];
  size: [number, number, number];
  color: string;
  emissive?: number;
};

const CHASSIS: Part[] = [
  { pos: [0, 0.34, 0], size: [0.8, 0.24, 1.1], color: MAT.rover },
  { pos: [0, 0.55, -0.02], size: [0.72, 0.2, 1.0], color: MAT.roverTrim },
  { pos: [0, 0.72, -0.1], size: [0.6, 0.14, 0.68], color: MAT.roverTrim },
  { pos: [0, 0.3, 0.57], size: [0.72, 0.16, 0.08], color: MAT.dark },
  { pos: [0, 0.5, -0.515], size: [0.56, 0.044, 0.024], color: '#ffb04a', emissive: 1.4 },
  { pos: [-0.375, 0.5, 0], size: [0.024, 0.04, 0.6], color: '#fff0d0', emissive: 1.1 },
  { pos: [0.375, 0.5, 0], size: [0.024, 0.04, 0.6], color: '#fff0d0', emissive: 1.1 },
  { pos: [0, 0.97, -0.2], size: [0.05, 0.4, 0.05], color: MAT.dark },
  { pos: [0, 1.2, -0.2], size: [0.18, 0.1, 0.14], color: MAT.roverTrim },
  { pos: [0, 1.3, -0.2], size: [0.14, 0.07, 0.14], color: MAT.dark },
  { pos: [0, 1.285, -0.2], size: [0.152, 0.02, 0.152], color: '#F2A33A', emissive: 1.2 },
  { pos: [0.12, 1.2, -0.3], size: [0.016, 0.32, 0.016], color: MAT.dark },
  { pos: [0, 0.82, 0.18], size: [0.2, 0.12, 0.16], color: MAT.dark },
];

const WHEELS: [number, number][] = [
  [-1, -1],
  [1, -1],
  [-1, 1],
  [1, 1],
];

type Props = {
  x: number;
  z: number;
  yaw: number;
  beacon: string;
  scale?: number;
};

export const Rover = memo(function Rover({ x, z, yaw, beacon, scale = 1.5 }: Props) {
  return (
    <group position={[x, 0, z]} rotation={[0, yaw, 0]} scale={scale}>
      {WHEELS.map(([qx, qz], i) => (
        <mesh key={i} position={[qx * 0.5, 0.17, qz * 0.36]} castShadow>
          <boxGeometry args={[0.17, 0.34, 0.34]} />
          <meshStandardMaterial color={MAT.dark} roughness={0.9} />
        </mesh>
      ))}
      {CHASSIS.map((p, i) => (
        <mesh key={i} position={p.pos} castShadow receiveShadow>
          <boxGeometry args={p.size} />
          <meshStandardMaterial
            color={p.color}
            roughness={p.emissive ? 0.4 : 0.62}
            metalness={p.emissive ? 0 : 0.18}
            emissive={p.emissive ? p.color : '#000000'}
            emissiveIntensity={p.emissive ?? 0}
          />
        </mesh>
      ))}
      <mesh position={[0, 1.38, -0.2]}>
        <boxGeometry args={[0.05, 0.05, 0.05]} />
        <meshStandardMaterial color={beacon} emissive={beacon} emissiveIntensity={2.2} />
      </mesh>
      <pointLight position={[0, 1.42, -0.2]} color={beacon} intensity={1.6} distance={5} decay={2} />
      <pointLight position={[0, 0.45, 2.4]} color="#fff0d0" intensity={5} distance={9} decay={2} />
    </group>
  );
});
