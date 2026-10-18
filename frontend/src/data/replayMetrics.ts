export type MetricRow = {
  label: string;
  goal: string;
  collisions: string;
  clearance: string;
  efficiency: string;
  latency: string;
  interventions: string;
  goalTone?: 'good' | 'bad' | 'muted';
  collisionTone?: 'bad' | 'normal';
  clearanceTone?: 'bad' | 'normal';
};

export const METRIC_COLUMNS = [
  'Goal success',
  'Collisions',
  'Min clearance',
  'Path efficiency',
  'Latency',
  'Interventions',
];

// Measured: 12 held-out v2 scenarios, closed loop, flow-matching planner ->
// ranking -> discrete-time CBF filter -> bicycle plant, scored against ground
// truth the same way northstar.control.simulate scores. Checkpoint
// v2_static_control.pt. Source data: backend/artifacts/frames/summary.json.
// Goal arrival is identical in both arms (4/12), so the filter removes every
// collision without costing a single arrival. Efficiency is averaged over
// arrivals only: start-gap over distance travelled is meaningless for a run
// that never arrives, where it reads above 1.
export const REPLAY_METRICS: MetricRow[] = [
  {
    label: 'Safety off',
    goal: '4 / 12',
    collisions: '9 / 12',
    clearance: '-0.80 m',
    efficiency: '92%',
    latency: '45 ms plan',
    interventions: '0',
    goalTone: 'muted',
    collisionTone: 'bad',
    clearanceTone: 'bad',
  },
  {
    label: 'Safety on',
    goal: '4 / 12',
    collisions: '0 / 12',
    clearance: '+0.16 m',
    efficiency: '98%',
    latency: '82 ms plan + 280 ms filter',
    interventions: '213 avg',
    goalTone: 'good',
  },
];

export const DESIGN_SAMPLE_METRICS: MetricRow[] = [
  {
    label: 'Safety off',
    goal: 'Missed',
    collisions: '1',
    clearance: '0.00 m',
    efficiency: '71%',
    latency: '—',
    interventions: '0',
    goalTone: 'muted',
    collisionTone: 'bad',
    clearanceTone: 'bad',
  },
  {
    label: 'Safety on',
    goal: 'Reached',
    collisions: '0',
    clearance: '0.46 m',
    efficiency: '93%',
    latency: '4.2 ms',
    interventions: '6',
    goalTone: 'good',
  },
];

export const REPLAY_SEED = '0427';
