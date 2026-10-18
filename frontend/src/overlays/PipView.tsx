import { WorldCanvas, type CameraMode, type DrivenPose } from '../scene/WorldCanvas';
import type { Obstacle } from '../scene/Obstacles';
import type { Pt } from '../scene/math';
import type { WorldStateName } from '../scene/scenes';
import type { SimClock } from '../hooks/useSimClock';
import { SwapIcon } from './Icons';

export function PipView({
  state,
  cameraMode,
  label,
  clock,
  runId,
  obstacles,
  route,
  candidates,
  fallback,
  trail,
  goalPoint,
  driven,
  agents,
  ring,
  onSwap,
}: {
  state: WorldStateName;
  cameraMode: CameraMode;
  label: string;
  clock?: React.RefObject<SimClock>;
  runId?: number;
  obstacles?: Obstacle[];
  route?: Pt[] | null;
  candidates?: Pt[][];
  fallback?: Pt[] | null;
  trail?: Pt[] | null;
  goalPoint?: Pt | null;
  driven?: DrivenPose | null;
  agents?: DrivenPose[];
  ring?: 'ok' | 'warn' | 'stop';
  onSwap: () => void;
}) {
  return (
    <div
      className="absolute right-7 top-[26px] h-[234px] w-[416px] overflow-hidden rounded-2xl bg-[#0b0c0f]"
      style={{ boxShadow: '0 0 0 1px rgba(242,163,58,0.72), 0 14px 36px rgba(0,0,0,0.45)' }}
    >
      <WorldCanvas
        className="absolute inset-0"
        state={state}
        cameraMode={cameraMode}
        clock={clock}
        runId={runId}
        obstacles={obstacles}
        route={route}
        candidates={candidates}
        fallback={fallback}
        trail={trail}
        goalPoint={goalPoint}
        driven={driven}
        agents={agents}
        ring={ring}
        detail="hi"
        labels={false}
      />
      <div className="glass absolute left-2.5 top-2.5 flex h-5 items-center rounded-[10px] px-2.5">
        <span className="lbl text-[9.5px] text-bone">{label}</span>
      </div>
      <button
        aria-label="Swap main and corner views"
        onClick={onSwap}
        className="glass absolute right-2.5 top-2.5 flex h-7 w-7 items-center justify-center rounded-[9px] border-0 p-0 text-bone"
      >
        <SwapIcon />
      </button>
    </div>
  );
}
