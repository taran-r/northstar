import { memo, useEffect, useMemo, useRef, useState } from 'react';
import { Canvas, useFrame, useThree, type ThreeEvent } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';
import { Hall } from './Hall';
import { Rover } from './Rover';
import { Forklift } from './Forklift';
import { PathRibbons, WheelTrail, pathPoints } from './Paths';
import {
  AttentionTiles,
  ContactMark,
  GoalCelebration,
  GoalMarker,
  HazardBand,
  SafetyRing,
} from './Markers';
import { useReducedMotion } from '../hooks/useMediaQuery';
import { Obstacles, type Obstacle } from './Obstacles';
import { MovingForklift } from './MovingForklift';
import { CROSSING_FORKLIFT, type AgentPose } from './agents';
import { LabelOverlay, LabelTracker, type WorldLabelSpec } from './LabelLayer';
import { HALL } from './palette';
import { FacilityHall, type FacilityGeometry } from './FacilityHall';
import {

  DRIVE_TRACK,
  LOOP_MS,
  SCENES,
  buildTrack,
  driveDist,
  type PathSpec,
  type Track,
  type WorldState,
  type WorldStateName,
} from './scenes';
import type { Pt } from './math';
import type { SimClock } from '../hooks/useSimClock';

export type CameraMode = 'chase' | 'free' | 'course' | 'setup';

export type QualityLevel = 'High' | 'Balanced' | 'Fast';

const RENDER: Record<QualityLevel, { detail: 'hi' | 'lo'; dpr: [number, number] | number }> = {
  High: { detail: 'hi', dpr: [1, 2] },
  Balanced: { detail: 'hi', dpr: 1 },
  Fast: { detail: 'lo', dpr: 1 },
};

const FOV = (f: number) => (2 * Math.atan(450 / f) * 180) / Math.PI;

const orbitCamera = (tiltDeg: number, yawDeg: number, dist: number, centerZ: number, f: number) => {
  const th = (tiltDeg * Math.PI) / 180;
  const ps = (yawDeg * Math.PI) / 180;
  const fwd = {
    x: Math.sin(ps) * Math.cos(th),
    y: -Math.sin(th),
    z: Math.cos(ps) * Math.cos(th),
  };
  return {
    pos: [-dist * fwd.x, -dist * fwd.y, centerZ - dist * fwd.z] as [number, number, number],
    target: [0, 0, centerZ] as [number, number, number],
    fov: FOV(f),
  };
};

const COURSE = orbitCamera(44, 62, 112, 48, 1500);

const setupCamera = (tiltDeg: number) => orbitCamera(tiltDeg, 66, 92, 48, 1050);

function CameraRig({
  mode,
  rover,
  snap,
  tilt,
  pose,
  focus,
}: {
  mode: CameraMode;
  rover: { x: number; z: number };
  snap: boolean;
  tilt: number;
  pose?: React.RefObject<Pose> | null;
  focus?: Pt | null;
}) {
  const { camera, size } = useThree();
  const target = useRef(new THREE.Vector3());
  const want = useRef(new THREE.Vector3());

  useFrame((_, dt) => {
    if (mode === 'free') return;
    const cam = camera as THREE.PerspectiveCamera;
    let fov: number;
    if (mode === 'chase') {
      const rx = pose ? pose.current.x : rover.x;
      const rz = pose ? pose.current.z : rover.z;
      want.current.set(rx * 0.5, 6.4, rz - 9.8);
      const pitch = (27 * Math.PI) / 180;
      target.current.set(rx * 0.5, 6.4 - Math.sin(pitch) * 20, rz - 9.8 + Math.cos(pitch) * 20);
      fov = FOV(520);
    } else if (mode === 'course') {
      if (focus) {
        want.current.set(focus.x + 6, 70, focus.z + 58);
        target.current.set(focus.x, 0, focus.z);
        fov = 42;
      } else {
        want.current.set(...COURSE.pos);
        target.current.set(...COURSE.target);
        fov = COURSE.fov;
      }
    } else {
      const setup = setupCamera(tilt);
      want.current.set(...setup.pos);
      target.current.set(...setup.target);
      fov = setup.fov;
    }
    if (mode === 'setup') {
      const dx = size.width * 0.11;
      const dy = size.height * 0.035;
      cam.setViewOffset(size.width, size.height, -dx, dy, size.width, size.height);
    } else if (cam.view?.enabled) {
      cam.clearViewOffset();
    }
    const k = snap ? 1 : 1 - Math.exp(-Math.min(dt, 0.1) / CAM_TAU);
    cam.position.lerp(want.current, k);
    cam.lookAt(target.current);
    if (Math.abs(cam.fov - fov) > 0.01) {
      cam.fov += (fov - cam.fov) * k;
      cam.updateProjectionMatrix();
    }
  });

  return null;
}

function Lighting({ detail }: { detail: 'hi' | 'lo' }) {
  return (
    <>
      <ambientLight intensity={0.8} color="#9db4d6" />
      <hemisphereLight args={['#aebfd8', '#1a2230', 1.0]} />
      <directionalLight
        position={[40, 26, 60]}
        intensity={2.7}
        color="#ffc98a"
        castShadow={detail === 'hi'}
        shadow-mapSize={[1024, 1024]}
        shadow-camera-left={-40}
        shadow-camera-right={40}
        shadow-camera-top={40}
        shadow-camera-bottom={-40}
        shadow-camera-far={160}
      />
      <directionalLight position={[-30, 20, -20]} intensity={0.55} color="#9db4d6" />
      <directionalLight position={[0, 24, 40]} intensity={0.5} color="#cfe0ff" />
    </>
  );
}

function FrameRate({ fps }: { fps: number }) {
  const invalidate = useThree((state) => state.invalidate);
  useEffect(() => {
    const id = setInterval(() => invalidate(), Math.round(1000 / fps));
    return () => clearInterval(id);
  }, [fps, invalidate]);
  return null;
}

export type Pose = { x: number; z: number; yaw: number; travelled: number };

const TICK_MS = 150;
const LOOP_SECONDS = LOOP_MS / 1000;
const LOOKAHEAD = 3.2;
const CAM_TAU = 0.16;
const YAW_TAU = 0.22;

function DriveAnimator({
  track,
  pose,
  group,
  clock,
  onTick,
}: {
  track: Track;
  pose: React.RefObject<Pose>;
  group: React.RefObject<THREE.Group | null>;
  clock: React.RefObject<SimClock>;
  onTick?: (p: number) => void;
}) {
  const lastTick = useRef(0);
  const lastT = useRef(-1);
  const smoothYaw = useRef<number | null>(null);
  useFrame((_, dt) => {
    const c = clock.current;
    const step = Math.min(dt, 0.1);
    if (c.playing) c.t = (c.t + step) % LOOP_SECONDS;
    const p = (c.t % LOOP_SECONDS) / LOOP_SECONDS;
    const travelled = driveDist(p, track.len);
    const at = track.at(travelled);
    const ahead = track.at(travelled + LOOKAHEAD);
    const behind = track.at(travelled - LOOKAHEAD);
    const target = Math.atan2(ahead.x - behind.x, ahead.z - behind.z);

    if (smoothYaw.current == null) smoothYaw.current = target;
    else {
      let delta = target - smoothYaw.current;
      while (delta > Math.PI) delta -= 2 * Math.PI;
      while (delta < -Math.PI) delta += 2 * Math.PI;
      const k = 1 - Math.exp(-step / YAW_TAU);
      smoothYaw.current += delta * k;
    }
    const yaw = smoothYaw.current;

    pose.current = { x: at.x, z: at.z, yaw, travelled };
    const g = group.current;
    if (g) {
      g.position.set(at.x, 0, at.z);
      g.rotation.y = yaw;
    }
    const now = performance.now();
    const moved = Math.abs(c.t - lastT.current) > 1e-4;
    if (onTick && moved && (c.playing ? now - lastTick.current > TICK_MS : true)) {
      lastTick.current = now;
      lastT.current = c.t;
      onTick(p);
    }
  });
  return null;
}

function DrivenRig({
  target,
  pose,
  group,
}: {
  target: DrivenPose;
  pose: React.RefObject<Pose>;
  group: React.RefObject<THREE.Group | null>;
}) {
  const smooth = useRef({ x: target.x, z: target.z, yaw: target.yaw });
  useFrame((_, dt) => {
    const step = Math.min(dt, 0.1);
    const jump = Math.hypot(target.x - smooth.current.x, target.z - smooth.current.z);
    const k = jump > 6 ? 1 : 1 - Math.exp(-step / 0.09);
    smooth.current.x += (target.x - smooth.current.x) * k;
    smooth.current.z += (target.z - smooth.current.z) * k;
    let delta = target.yaw - smooth.current.yaw;
    while (delta > Math.PI) delta -= 2 * Math.PI;
    while (delta < -Math.PI) delta += 2 * Math.PI;
    smooth.current.yaw += delta * k;
    pose.current = {
      x: smooth.current.x,
      z: smooth.current.z,
      yaw: smooth.current.yaw,
      travelled: 0,
    };
    const g = group.current;
    if (g) {
      g.position.set(smooth.current.x, 0, smooth.current.z);
      g.rotation.y = smooth.current.yaw;
    }
  });
  return null;
}

const HOP_S = 1.2;

/** A few settling hops when the rover arrives; flat when motion is reduced. */
function CelebrationHop({
  active,
  lift,
  children,
}: {
  active: boolean;
  lift: number;
  children: React.ReactNode;
}) {
  const group = useRef<THREE.Group>(null);
  const start = useRef<number | null>(null);
  const still = useReducedMotion();
  useFrame(({ clock }) => {
    const g = group.current;
    if (!g) return;
    if (!active || still) {
      start.current = null;
      g.position.y = 0;
      return;
    }
    if (start.current == null) start.current = clock.elapsedTime;
    const t = clock.elapsedTime - start.current;
    g.position.y = t < HOP_S ? lift * Math.abs(Math.sin(t * Math.PI * 2.5)) * (1 - t / HOP_S) : 0;
  });
  return <group ref={group}>{children}</group>;
}

function DriveTrail({ track, pose }: { track: Track; pose: React.RefObject<Pose> }) {
  const [travelled, setTravelled] = useState(0);
  const last = useRef(0);
  useFrame(() => {
    const now = performance.now();
    if (now - last.current < 420) return;
    last.current = now;
    const t = pose.current.travelled;
    if (Math.abs(t - travelled) > 0.2) setTravelled(t);
  });
  const pts = useMemo(() => {
    const out: Pt[] = [];
    for (let q = Math.max(0, travelled - 26); q <= travelled; q += 1.1) out.push(track.at(q));
    out.push(track.at(travelled));
    return out;
  }, [track, travelled]);
  return <WheelTrail pts={pts} />;
}

function DriveAttention({
  track,
  pose,
  goal,
  selected,
  fork,
}: {
  track: Track;
  pose: React.RefObject<Pose>;
  goal: Pt;
  selected?: PathSpec;
  fork?: { x: number; z: number } | null;
}) {
  const [cell, setCell] = useState(0);
  const last = useRef(0);
  useFrame(() => {
    const now = performance.now();
    if (now - last.current < 500) return;
    last.current = now;
    const next = Math.round(pose.current.travelled / 2);
    if (next !== cell) setCell(next);
  });
  const at = track.at(cell * 2);
  return (
    <AttentionTiles rover={{ x: at.x, z: at.z }} goal={goal} selected={selected} fork={fork} />
  );
}

export type DrivenPose = { x: number; z: number; yaw: number };

export type WorldProps = {
  state: WorldStateName;
  cameraMode: CameraMode;
  attention?: boolean;
  detail?: 'hi' | 'lo';
  obstacles?: Obstacle[];
  selected?: number;
  route?: Pt[] | null;
  candidates?: Pt[][];
  fallback?: Pt[] | null;
  trail?: Pt[] | null;
  goalPoint?: Pt | null;
  driven?: DrivenPose | null;
  /** Live forklift poses from the bridge (placed patrols + hall crossing). */
  agents?: DrivenPose[];
  ring?: 'ok' | 'warn' | 'stop';
  /** The backend reported the goal reached: play the arrival animation. */
  celebrate?: boolean;
  facility?: FacilityGeometry | null;
  focus?: Pt | null;
  onFloorClick?: (p: Pt) => void;
  onPickObstacle?: (index: number) => void;
  onTick?: (p: number) => void;
  labels?: boolean;
  clock?: React.RefObject<SimClock>;
  runId?: number;
  quality?: QualityLevel;
  pixelRatio?: number;
  fps?: number;
  tilt?: number;
  className?: string;
};

function WorldCanvasImpl({
  state,
  cameraMode,
  clock,
  runId = 0,
  attention = false,
  detail = 'hi',
  obstacles,
  selected = -1,
  route = null,
  candidates = [],
  fallback = null,
  trail = null,
  goalPoint = null,
  driven = null,
  agents = [],
  ring = 'ok',
  celebrate = false,
  facility = null,
  focus = null,
  onFloorClick,
  onPickObstacle,
  onTick,
  labels = true,
  quality,
  pixelRatio,
  fps,
  tilt = 52,
  className,
}: WorldProps) {
  const render = quality
    ? RENDER[quality]
    : { detail, dpr: detail === 'hi' ? ([1, 2] as [number, number]) : 0.75 };
  const level = render.detail;
  const isSetup = cameraMode === 'setup';
  const base = SCENES[state] ?? SCENES.hero;

  const track = useMemo(
    () => (!isSetup && route && route.length > 1 ? buildTrack(route) : DRIVE_TRACK),
    [isSetup, route],
  );
  const courseCandidates = useMemo(
    () => (!isSetup && route && route.length > 1 ? candidates : []),
    [isSetup, route, candidates],
  );
  const liveDrive = driven != null;
  const driving = !!clock && state === 'hero' && !liveDrive;
  const pose = useRef<Pose>({
    x: track.pts[0].x,
    z: track.pts[0].z,
    yaw: 0,
    travelled: 0,
  });
  const roverGroup = useRef<THREE.Group | null>(null);
  const hasPlacedForklift = !!obstacles?.some((o) => o.t === 'forklift');
  const agentPose = useRef<AgentPose>({
    x: CROSSING_FORKLIFT.a.x,
    z: CROSSING_FORKLIFT.a.z,
    yaw: Math.PI / 2,
    moving: true,
    waiting: false,
  });

  const scene: WorldState = useMemo(() => {
    if (liveDrive && driven) {
      const paths: PathSpec[] = candidates.map((pts) => ({ k: 'cand', pts }));
      if (route && route.length > 1) paths.push({ k: 'sel', pts: route });
      if (fallback && fallback.length > 1) paths.push({ k: 'fb', pts: fallback });
      const beacon = ring === 'stop' ? '#E5484D' : ring === 'warn' ? '#F2A33A' : '#5B8DEF';
      return {
        rov: { x: driven.x, z: driven.z, yaw: driven.yaw },
        goal: goalPoint,
        ring,
        beacon,
        fork: null,
        speed: 0,
        paths,
      };
    }
    if (isSetup) {
      return {
        rov: { x: -1.5, z: 8, yaw: 0 },
        goal: { x: 3, z: 86 },
        ring: 'ok',
        beacon: '#5B8DEF',
        fork: null,
        speed: 0,
        paths: [
          ...candidates.map((pts): PathSpec => ({ k: 'cand', pts })),
          ...(route ? [{ k: 'sel', pts: route } as PathSpec] : []),
        ],
      };
    }
    if (driving) {
      const pts = track.pts;
      const goal = pts[pts.length - 1];
      const start = pts[0];
      const paths: PathSpec[] = courseCandidates.length
        ? courseCandidates.map((pts2): PathSpec => ({ k: 'cand', pts: pts2 }))
        : [
            { k: 'cand', off: -6.5 },
            { k: 'cand', off: -3.4 },
            { k: 'cand', off: 3.2 },
            { k: 'cand', off: 6.2 },
          ];
      paths.push({ k: 'sel', pts });
      return { ...base, rov: { x: start.x, z: start.z, yaw: 0 }, goal, paths };
    }
    return base;
  }, [
    isSetup,
    base,
    driving,
    track,
    courseCandidates,
    route,
    candidates,
    liveDrive,
    driven,
    fallback,
    goalPoint,
    ring,
  ]);

  const rover = scene.rov;
  const start: Pt = { x: rover.x, z: rover.z };
  const selectedPath = scene.paths.find((p) => p.k === 'sel');

  const staticTrail = useMemo(() => {
    const z0 = Math.max(8, rover.z - 24);
    const out: Pt[] = [];
    for (let z = z0; z <= rover.z + 0.001; z += 0.5) {
      const t = Math.min(1, Math.max(0, (z - 8) / Math.max(1, rover.z - 8)));
      out.push({ x: rover.x * t + 0.5 * Math.sin((z - rover.z) * 0.22) * (1 - t * 0.2), z });
    }
    return out;
  }, [rover.x, rover.z]);

  const handleFloor = (e: ThreeEvent<MouseEvent>) => {
    if (!onFloorClick) return;
    e.stopPropagation();
    onFloorClick({ x: e.point.x, z: e.point.z });
  };

  const worldLabels: WorldLabelSpec[] = [];
  if (labels) {
    scene.paths.forEach((p, i) => {
      if (!p.label || p.at == null) return;
      const pts = pathPoints(p, start, scene.goal);
      const q = pts[Math.floor(pts.length * p.at)];
      if (q) {
        worldLabels.push({
          id: 'path' + i,
          pos: [q.x, 0.4, q.z],
          text: p.label,
          color: '#E6E2D8',
          dashed: true,
        });
      }
    });
    if (isSetup && scene.goal) {
      worldLabels.push({
        id: 'start',
        pos: [rover.x, 0.6, rover.z - 3.4],
        text: 'Start',
        color: '#5B8DEF',
      });
      worldLabels.push({
        id: 'goal',
        pos: [scene.goal.x, 0.6, scene.goal.z + 4.6],
        text: 'Goal',
        color: '#4CAF7D',
      });
    }
    if (scene.cursor && scene.goal) {
      worldLabels.push({
        id: 'cursor',
        pos: [scene.goal.x, 0.6, scene.goal.z + 4],
        text: 'Goal · ' + Math.round(scene.goal.z - rover.z) + ' m',
        color: '#4CAF7D',
      });
    }
  }
  const labelNodes = useRef<Record<string, HTMLDivElement | null>>({});

  const ringRadius = isSetup ? 3.4 : cameraMode === 'course' ? 4.6 : 2.1;
  const goalRadius = isSetup ? 2.8 : cameraMode === 'course' ? 3.2 : 1.4;
  const roverScale = isSetup ? 2.6 : cameraMode === 'course' ? 3.2 : 1.5;
  const beacon = celebrate ? '#4CAF7D' : scene.beacon;

  return (
    <div className={className ?? 'absolute inset-0'}>
    <Canvas
      className="absolute inset-0"
      shadows={level === 'hi'}
      dpr={pixelRatio ?? render.dpr}
      frameloop={fps ? 'demand' : 'always'}
      gl={{ antialias: true, toneMapping: THREE.ACESFilmicToneMapping }}
      camera={{ position: [0, 6.4, 4], fov: FOV(520), near: 0.3, far: 400 }}
    >
      <color attach="background" args={['#0b0f15']} />
      {(cameraMode === 'chase' || cameraMode === 'free') && (
        <fog attach="fog" args={['#58697f', 34, 170]} />
      )}
      {fps ? <FrameRate fps={fps} /> : null}
      <Lighting detail={level} />
      <CameraRig
        mode={cameraMode}
        rover={rover}
        snap={false}
        tilt={tilt}
        pose={driving || liveDrive ? pose : null}
        focus={focus}
      />
      {driving && clock && (
        <DriveAnimator
          track={track}
          pose={pose}
          group={roverGroup}
          clock={clock}
          onTick={onTick}
        />
      )}
      {liveDrive && driven && (
        <DrivenRig target={driven} pose={pose} group={roverGroup} />
      )}
      {cameraMode === 'free' && (
        <OrbitControls
          target={[rover.x, 1, rover.z + 8]}
          maxPolarAngle={Math.PI / 2.05}
          enablePan
        />
      )}

      {facility ? (
        <FacilityHall facility={facility} onFloorClick={onFloorClick} />
      ) : (
        <Hall detail={level} />
      )}

      {onFloorClick && !facility && (
        <mesh
          position={[0, 0, HALL.ZE / 2]}
          rotation={[-Math.PI / 2, 0, 0]}
          onClick={handleFloor}
          visible={false}
        >
          <planeGeometry args={[HALL.X1 - HALL.X0, HALL.ZE]} />
          <meshBasicMaterial />
        </mesh>
      )}

      {obstacles && obstacles.length > 0 && (
        <Obstacles
          items={obstacles}
          selected={selected}
          onPick={onPickObstacle}
          clock={clock}
          rover={driving ? pose : null}
          animate={!isSetup && !liveDrive}
          runId={runId}
        />
      )}

      <PathRibbons paths={scene.paths} start={start} goal={scene.goal} />
      {liveDrive ? (
        <WheelTrail pts={trail && trail.length > 1 ? trail : []} />
      ) : driving ? (
        <DriveTrail track={track} pose={pose} />
      ) : (
        <WheelTrail pts={staticTrail} />
      )}

      {attention && scene.goal && driving && (
        <DriveAttention
          track={track}
          pose={pose}
          goal={scene.goal}
          selected={selectedPath}
          fork={scene.fork}
        />
      )}
      {attention && scene.goal && !driving && (
        <AttentionTiles
          rover={start}
          goal={scene.goal}
          selected={selectedPath}
          fork={scene.fork}
        />
      )}
      {scene.hazard && <HazardBand {...scene.hazard} />}

      {!driving && !liveDrive && (
        <SafetyRing position={[rover.x, 0.03, rover.z]} state={scene.ring} radius={ringRadius} />
      )}
      {scene.goal && (
        <GoalMarker
          position={[scene.goal.x, 0.03, scene.goal.z]}
          radius={goalRadius}
          cursor={scene.cursor}
        />
      )}
      {celebrate && scene.goal && (
        <GoalCelebration
          position={[scene.goal.x, 0.03, scene.goal.z]}
          radius={goalRadius}
        />
      )}
      {scene.contact && <ContactMark position={[scene.contact.x, 0.05, scene.contact.z]} />}

      <LabelTracker labels={worldLabels} nodes={labelNodes.current} />

      {driving || liveDrive ? (
        <group ref={roverGroup}>
          <CelebrationHop active={celebrate} lift={0.3 * roverScale}>
            <Rover x={0} z={0} yaw={0} beacon={beacon} scale={roverScale} />
          </CelebrationHop>
          <SafetyRing position={[0, 0.03, 0]} state={scene.ring} radius={ringRadius} />
        </group>
      ) : (
        <Rover x={rover.x} z={rover.z} yaw={rover.yaw} beacon={beacon} scale={roverScale} />
      )}
      {liveDrive &&
        agents.map((pose, i) => (
          <Forklift
            key={`live-fork-${i}`}
            x={pose.x}
            z={pose.z}
            yaw={pose.yaw}
            scale={cameraMode === 'chase' || cameraMode === 'free' ? 1.1 : 1.8}
          />
        ))}
      {liveDrive ? null : driving ? (
        hasPlacedForklift ? null : (
          <MovingForklift
            key={runId}
            spec={CROSSING_FORKLIFT}
            pose={agentPose}
            rover={pose}
            clock={clock}
            showLane
            scale={cameraMode === 'chase' || cameraMode === 'free' ? 1.1 : 1.8}
          />
        )
      ) : (
        scene.fork && (
          <Forklift
            x={scene.fork.x}
            z={scene.fork.z}
            yaw={scene.fork.yaw}
            scale={cameraMode === 'chase' || cameraMode === 'free' ? 1.1 : 1.8}
          />
        )
      )}
    </Canvas>
      <LabelOverlay labels={worldLabels} nodes={labelNodes.current} />
    </div>
  );
}

export const WorldCanvas = memo(WorldCanvasImpl);

export { pathPoints };
