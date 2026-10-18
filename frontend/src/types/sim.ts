export const SCHEMA_VERSION = 1;

export type Vec3 = [number, number, number];

export type RoverState = {
  x: number;
  y: number;
  z: number;
  heading: number;
  speed: number;
  steering: number;
};

export type ObstacleState = {
  id: string;
  position: Vec3;
  velocity: Vec3;
  shape: 'box' | 'cylinder';
  size: Vec3;
  kind?: string;
  yaw?: number;
};

export type PathPoint = { x: number; y: number; t?: number };

export type ControllerMode = 'tracking' | 'intervention' | 'estop' | 'replanning' | 'goal';

export type ControllerState = {
  mode: ControllerMode;
  nominal: { a: number; steer: number };
  actual: { a: number; steer: number };
  barrier_min: number | null;
  latency_ms: number | null;
  source: 'learned' | 'fallback';
};

export type RunMetrics = {
  collisions: number | null;
  min_clearance: number | null;
  replans: number | null;
  interventions: number | null;
  path_efficiency: number | null;
  goal_reached: boolean | null;
};

export type StateFrame = {
  type: 'state';
  version: number;
  tick: number;
  sim_time: number;
  seed: number;
  rover: RoverState;
  goal: Vec3 | null;
  obstacles: ObstacleState[];
  candidate_paths: PathPoint[][];
  selected_path: PathPoint[];
  fallback_paths: PathPoint[];
  executed_trail: PathPoint[];
  controller: ControllerState;
  metrics: RunMetrics;
  measured: boolean;
};

export type Intent =
  | { type: 'reset'; seed: number }
  | { type: 'set_goal'; x: number; y: number; z: number }
  | { type: 'spawn_obstacle'; kind: string; x: number; y: number; yaw: number }
  | { type: 'toggle_safety'; enabled: boolean }
  | { type: 'start' }
  | { type: 'pause' }
  | { type: 'resume' }
  | { type: 'step' }
  | { type: 'start_replay'; seed: number; policy_mode: 'safety_on' | 'safety_off' };

export interface SimSource {
  readonly live: boolean;
  subscribe(onFrame: (frame: StateFrame) => void): () => void;
  send(intent: Intent): void;
}
