import { useCallback, useState } from 'react';
import type { CameraMode, QualityLevel } from '../scene/WorldCanvas';
import type { DockVariant } from '../overlays/Dock';
import type { ChipKind } from '../overlays/StatusChip';
import { liveAt, type WorldStateName } from '../scene/scenes';
import { useSimClock } from './useSimClock';

export type Scene =
  | 'hero'
  | 'goal'
  | 'intervention'
  | 'estop'
  | 'replan'
  | 'panel'
  | 'network'
  | 'networkAvoid';

type SceneDef = {
  world: WorldStateName;
  chip: ChipKind;
  speed: string;
  toast?: 'int' | 'stop' | 'replan' | 'goal';
  panel?: boolean;
  network?: 'track' | 'avoid';
  live?: boolean;
};

export const SCENE_DEFS: Record<Scene, SceneDef> = {
  hero: { world: 'hero', chip: 'tracking', speed: '1.4', live: true },
  goal: { world: 'goal', chip: 'replanning', speed: '1.1', toast: 'goal' },
  intervention: { world: 'intervention', chip: 'intervention', speed: '0.6', toast: 'int' },
  estop: { world: 'estop', chip: 'estop', speed: '0.0', toast: 'stop' },
  replan: { world: 'replan', chip: 'replanning', speed: '0.0', toast: 'replan' },
  panel: { world: 'intervention', chip: 'intervention', speed: '0.6', panel: true },
  network: { world: 'hero', chip: 'tracking', speed: '1.4', network: 'track', live: true },
  networkAvoid: {
    world: 'intervention',
    chip: 'intervention',
    speed: '0.6',
    network: 'avoid',
  },
};

const QUALITIES: QualityLevel[] = ['High', 'Balanced', 'Fast'];

export function useRunState() {
  const [scene, setScene] = useState<Scene>('hero');
  const [variant, setVariant] = useState<DockVariant>('floating');
  const [camera, setCamera] = useState<CameraMode>('chase');
  const [swapped, setSwapped] = useState(false);
  const clock = useSimClock(true);
  const [progress, setProgress] = useState(0);
  const [safety, setSafety] = useState(true);
  const [panelOpen, setPanelOpen] = useState(true);
  const [quality, setQuality] = useState<QualityLevel>(QUALITIES[0]);
  const [attentionOnMap, setAttentionOnMap] = useState(true);

  const def = SCENE_DEFS[scene];
  const live = !!def.live;
  const liveFrame = liveAt(progress * 44000);

  const toggleNetwork = useCallback(() => {
    setScene((s) => {
      if (s === 'network') return 'hero';
      if (s === 'networkAvoid') return 'intervention';
      if (s === 'intervention' || s === 'panel') return 'networkAvoid';
      return 'network';
    });
  }, []);

  const cycleQuality = useCallback(() => {
    setQuality((q) => QUALITIES[(QUALITIES.indexOf(q) + 1) % QUALITIES.length]);
  }, []);

  return {
    scene,
    setScene,
    def,
    variant,
    setVariant,
    camera,
    setCamera,
    swapped,
    toggleSwap: () => setSwapped((s) => !s),
    playing: clock.playing,
    togglePlay: clock.toggle,
    step: clock.step,
    restart: clock.restart,
    clock: clock.ref,
    runId: clock.runId,
    progress,
    setProgress,
    safety,
    toggleSafety: () => setSafety((s) => !s),
    panelOpen,
    togglePanel: () => setPanelOpen((p) => !p),
    quality,
    cycleQuality,
    attentionOnMap,
    toggleAttentionOnMap: () => setAttentionOnMap((a) => !a),
    toggleNetwork,
    chip: live ? liveFrame.chip : def.chip,
    speed: live ? liveFrame.speed : def.speed,
  };
}
