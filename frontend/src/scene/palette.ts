
export const KIT = {
  charcoal: '#14171C',
  steel: '#1F252D',
  amber: '#F2A33A',
  dusk: '#5B8DEF',
  bone: '#E8E6E1',
  alert: '#E5484D',
  go: '#4CAF7D',
  fallback: '#DCD8CE',
} as const;

export const MAT = {
  conc: '#8f9aa8',
  post: '#2f5aa8',
  beam: '#e07a2c',
  steel: '#3b4553',
  crate: '#cfa676',
  tote: '#8aa0b8',
  wood: '#bb9568',
  guard: '#eab932',
  fork: '#dca83c',
  dark: '#1d2126',
  rover: '#46505d',
  roverTrim: '#8e7562',
  slab: '#2d3643',
  floor: '#39465a',
  wall: '#34445c',
  wallUpper: '#42567a',
} as const;

export const HALL = { X0: -16, X1: 16, ZE: 96, H: 11 } as const;

export const PATH_STYLE: Record<string, { color: string; opacity: number; width: number }> = {
  cand: { color: '#9ec3ff', opacity: 0.75, width: 0.26 },
  sel: { color: '#F2A33A', opacity: 0.98, width: 0.44 },
  dim: { color: '#8f9aaa', opacity: 0.4, width: 0.28 },
  fb: { color: '#F2EFE6', opacity: 0.95, width: 0.34 },
};

export const OBSTACLE_R: Record<string, number> = {
  pallet: 1.7,
  crates: 2.4,
  barrier: 2.7,
  forklift: 2.8,
};
