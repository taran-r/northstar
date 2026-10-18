import type { DockVariant } from './Dock';
import type { Scene } from '../hooks/useRunState';

const SCENES: [Scene, string][] = [
  ['hero', 'Tracking'],
  ['goal', 'Goal selection'],
  ['intervention', 'Intervention'],
  ['estop', 'Emergency stop'],
  ['replan', 'Replanning'],
  ['panel', 'Detail panel'],
  ['network', 'Network view'],
  ['networkAvoid', 'Network · slowing'],
];

const VARIANTS: [DockVariant, string][] = [
  ['floating', 'Dock'],
  ['rail', 'Rail'],
  ['split', 'Split'],
];

export function SceneSwitcher({
  scene,
  onScene,
  variant,
  onVariant,
  onSetup,
  onCompare,
}: {
  scene: Scene;
  onScene: (s: Scene) => void;
  variant: DockVariant;
  onVariant: (v: DockVariant) => void;
  onSetup: () => void;
  onCompare: () => void;
}) {
  return (
    <div className="glass absolute left-7 top-[120px] flex w-[188px] flex-col gap-2 rounded-2xl p-2.5">
      <span className="lbl px-1">States</span>
      <div className="flex flex-col gap-0.5">
        {SCENES.map(([id, label]) => (
          <button
            key={id}
            onClick={() => onScene(id)}
            className={`h-7 rounded-lg border-0 px-2 text-left text-[12px] ${
              scene === id
                ? 'bg-[rgba(242,163,58,0.16)] text-amber'
                : 'bg-transparent text-[rgba(232,230,225,0.72)] hover:bg-[rgba(232,230,225,0.08)]'
            }`}
          >
            {label}
          </button>
        ))}
      </div>
      <button
        onClick={onSetup}
        className="h-7 rounded-lg border-0 bg-transparent px-2 text-left text-[12px] text-[rgba(232,230,225,0.72)] hover:bg-[rgba(232,230,225,0.08)]"
      >
        Course setup
      </button>
      <button
        onClick={onCompare}
        className="h-7 rounded-lg border-0 bg-transparent px-2 text-left text-[12px] text-[rgba(232,230,225,0.72)] hover:bg-[rgba(232,230,225,0.08)]"
      >
        Replay comparison
      </button>
      <div className="h-px bg-[rgba(232,230,225,0.08)]" />
      <div className="seg">
        {VARIANTS.map(([id, label]) => (
          <button
            key={id}
            className={variant === id ? 'on' : ''}
            style={{ flex: '1 1 0' }}
            onClick={() => onVariant(id)}
          >
            {label}
          </button>
        ))}
      </div>
    </div>
  );
}
