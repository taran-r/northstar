import { memo, useMemo } from 'react';
import type { ThreeEvent } from '@react-three/fiber';
import { MAT } from './palette';
import { toSceneGround, toSceneMeshYaw } from './coords';
import type { Pt } from './math';

export type FacilityGeometry = {
  width_x: number;
  depth_y: number;
  height_z: number;
  walls: {
    id: string;
    x1: number;
    y1: number;
    x2: number;
    y2: number;
    thickness: number;
    height: number;
  }[];
  columns: { id: string; x: number; y: number; radius: number; height: number }[];
  permanent: {
    id: string;
    kind: string;
    x: number;
    y: number;
    yaw: number;
    length: number;
    width: number;
    height: number;
  }[];
};

/**
 * The real facility, in the same materials as the canned hall.
 *
 * Incoming coordinates are backend (x, y) horizontal. Conversion goes through
 * `toSceneGround` / `toSceneMeshYaw` and is not re-derived here.
 */
export const FacilityHall = memo(function FacilityHall({
  facility,
  onFloorClick,
}: {
  facility: FacilityGeometry;
  onFloorClick?: (p: Pt) => void;
}) {
  const width = facility.width_x;
  const depth = facility.depth_y;

  const walls = useMemo(
    () =>
      facility.walls.map((wall) => {
        const dx = wall.x2 - wall.x1;
        const dy = wall.y2 - wall.y1;
        const length = Math.hypot(dx, dy);
        const heading = Math.atan2(dy, dx);
        const mid = toSceneGround((wall.x1 + wall.x2) / 2, (wall.y1 + wall.y2) / 2);
        return {
          id: wall.id,
          x: mid.x,
          z: mid.z,
          yaw: toSceneMeshYaw(heading),
          length: Math.max(length, 0.05),
          thickness: wall.thickness,
          height: wall.height,
        };
      }),
    [facility.walls],
  );

  const columns = useMemo(
    () =>
      facility.columns.map((column) => {
        const ground = toSceneGround(column.x, column.y);
        return { ...column, x: ground.x, z: ground.z };
      }),
    [facility.columns],
  );

  const machines = useMemo(
    () =>
      facility.permanent.map((obj) => {
        const ground = toSceneGround(obj.x, obj.y);
        return {
          ...obj,
          x: ground.x,
          z: ground.z,
          yaw: toSceneMeshYaw(obj.yaw),
        };
      }),
    [facility.permanent],
  );

  const handleFloor = (event: ThreeEvent<MouseEvent>) => {
    if (!onFloorClick) return;
    event.stopPropagation();
    onFloorClick({ x: event.point.x, z: event.point.z });
  };

  const stripes = useMemo(() => {
    const out: number[] = [];
    for (let x = 6; x < width; x += 6) out.push(x);
    return out;
  }, [width]);

  return (
    <group>
      <mesh
        position={[width / 2, 0, -depth / 2]}
        rotation={[-Math.PI / 2, 0, 0]}
        receiveShadow
        onClick={handleFloor}
      >
        <planeGeometry args={[width, depth]} />
        <meshStandardMaterial color={MAT.floor} roughness={0.52} metalness={0.12} />
      </mesh>

      {stripes.map((x) => (
        <mesh key={x} position={[x, 0.02, -depth / 2]} rotation={[-Math.PI / 2, 0, 0]}>
          <planeGeometry args={[0.04, depth]} />
          <meshBasicMaterial color="#ffffff" transparent opacity={0.05} depthWrite={false} />
        </mesh>
      ))}

      {walls.map((wall) => (
        <mesh
          key={wall.id}
          position={[wall.x, wall.height / 2, wall.z]}
          rotation={[0, wall.yaw, 0]}
          castShadow
          receiveShadow
        >
          <boxGeometry args={[wall.thickness, wall.height, wall.length]} />
          <meshStandardMaterial color={MAT.wall} roughness={0.8} metalness={0.04} />
        </mesh>
      ))}

      {columns.map((column) => (
        <mesh
          key={column.id}
          position={[column.x, column.height / 2, column.z]}
          castShadow
          receiveShadow
        >
          <cylinderGeometry args={[column.radius, column.radius, column.height, 16]} />
          <meshStandardMaterial color={MAT.conc} roughness={0.7} metalness={0.05} />
        </mesh>
      ))}

      {machines.map((obj) => (
        <mesh
          key={obj.id}
          position={[obj.x, obj.height / 2, obj.z]}
          rotation={[0, obj.yaw, 0]}
          castShadow
          receiveShadow
        >
          <boxGeometry args={[obj.width, obj.height, obj.length]} />
          <meshStandardMaterial color={MAT.steel} roughness={0.64} metalness={0.18} />
        </mesh>
      ))}
    </group>
  );
});
