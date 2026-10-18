import { WorldCanvas } from '../scene/WorldCanvas';
import { Wordmark } from '../overlays/Wordmark';
import { ObstacleTile } from '../overlays/ObstacleTile';
import { ArrowRightIcon, DiceIcon, RotateIcon, RouteIcon, TrashIcon } from '../overlays/Icons';
import { TOOL_NAMES, type CourseSetup, type Density, type ToolId } from '../hooks/useCourseSetup';

const TOOLS: ToolId[] = ['pallet', 'crates', 'barrier', 'forklift'];
const DENSITIES: Density[] = ['sparse', 'medium', 'dense'];

export function SetupScreen({ setup, onStart }: { setup: CourseSetup; onStart: () => void }) {

  return (
    <div className="relative h-full w-full overflow-hidden bg-[#0b0c0f]">
      <WorldCanvas
        className="absolute inset-0"
        state="hero"
        cameraMode="setup"
        tilt={setup.tilt}
        obstacles={setup.obstacles}
        selected={setup.selected}
        route={setup.route}
        candidates={setup.candidates}
        onFloorClick={setup.placeAt}
        onPickObstacle={setup.setSelected}
        detail="hi"
      />

      <div
        className="pointer-events-none absolute inset-x-0 top-0 h-[150px]"
        style={{ background: 'linear-gradient(180deg, rgba(8,9,12,0.66), rgba(8,9,12,0))' }}
      />
      <div
        className="pointer-events-none absolute inset-x-0 bottom-0 h-[180px]"
        style={{ background: 'linear-gradient(0deg, rgba(8,9,12,0.7), rgba(8,9,12,0))' }}
      />

      <Wordmark />

      <div className="glass absolute left-1/2 top-[26px] flex h-[34px] -translate-x-1/2 items-center gap-2.5 rounded-[17px] pl-3.5 pr-4">
        <span className="block h-2.5 w-2.5 rounded-full border-2 border-dusk box-border" />
        <span className="text-[12px] font-medium uppercase leading-4 tracking-[0.16em]">
          Course setup
        </span>
      </div>

      <div className="glass absolute right-7 top-[22px] flex h-11 items-center gap-3 rounded-[14px] pl-4 pr-1.5">
        <span className="lbl">Seed</span>
        <span className="min-w-10 text-[17px] font-medium tracking-[0.04em]">{setup.seedText}</span>
        <span className="hairline" />
        <button className="btn btn-amber" onClick={setup.randomize}>
          <DiceIcon />
          <span>Randomize</span>
        </button>
      </div>

      <div className="glass absolute left-7 top-[84px] flex w-[292px] flex-col gap-[18px] rounded-[18px] p-[18px]">
        <div className="flex items-baseline justify-between">
          <span className="lbl text-bone">Add obstacle</span>
          <span className="text-[11px] text-[rgba(232,230,225,0.5)]">Scaled up for the map</span>
        </div>
        <div className="-mt-1.5 grid grid-cols-2 gap-2">
          {TOOLS.map((id) => (
            <ObstacleTile
              key={id}
              id={id}
              label={TOOL_NAMES[id]}
              active={setup.tool === id}
              onPick={() => setup.setTool(id)}
            />
          ))}
        </div>

        <div className="h-px bg-[rgba(232,230,225,0.08)]" />

        <div className="flex flex-col gap-3">
          <div className="flex items-baseline justify-between">
            <span className="lbl text-bone">Layout</span>
            <span className="text-[12px] text-[rgba(232,230,225,0.66)]">
              {setup.obstacles.length} {setup.obstacles.length === 1 ? 'obstacle' : 'obstacles'}
            </span>
          </div>
          <div className="seg w-full">
            {DENSITIES.map((d) => (
              <button
                key={d}
                style={{ flex: '1 1 0' }}
                className={setup.density === d ? 'on' : ''}
                onClick={() => setup.pickDensity(d)}
              >
                {d.charAt(0).toUpperCase() + d.slice(1)}
              </button>
            ))}
          </div>
          <div className="flex gap-2">
            <button className="btn flex-1" onClick={setup.randomize}>
              Randomize
            </button>
            <button className="btn flex-1" onClick={setup.clear}>
              Clear
            </button>
          </div>
        </div>

        {setup.selectedObstacle && (
          <>
            <div className="h-px bg-[rgba(232,230,225,0.08)]" />
            <div className="flex flex-col gap-3">
              <div className="flex items-center gap-2.5">
                <span className="h-[7px] w-[7px] rounded-full bg-amber" />
                <span className="lbl text-bone">Selected</span>
                <span className="text-[12px] text-[rgba(232,230,225,0.8)]">
                  {TOOL_NAMES[setup.selectedObstacle.t as ToolId]}
                </span>
              </div>
              <div className="flex items-baseline justify-between">
                <span className="text-[12px] text-[rgba(232,230,225,0.66)]">Position</span>
                <span className="text-[13px]">
                  x {setup.selectedObstacle.x.toFixed(1)} · z {setup.selectedObstacle.z.toFixed(1)} m
                </span>
              </div>
              <div className="flex gap-2">
                <button className="btn flex-1" onClick={setup.rotate}>
                  <RotateIcon />
                  Rotate
                </button>
                <button className="btn flex-1" onClick={setup.remove}>
                  <TrashIcon />
                  Delete
                </button>
              </div>
            </div>
          </>
        )}

        <div className="h-px bg-[rgba(232,230,225,0.08)]" />
        <span className="text-[12px] leading-[18px] text-[rgba(232,230,225,0.6)]">
          Click the floor to place. Click an obstacle to select it. The amber route replans as you
          edit.
        </span>
      </div>

      <div
        className="absolute bottom-[30px] flex flex-col items-end gap-0.5"
        style={{ right: 'calc(50% + 370px)' }}
      >
        <span className="lbl">Planned route</span>
        <div className="flex items-baseline gap-1.5">
          <span className="text-[34px] font-light leading-[38px] tracking-[-0.01em]">
            {setup.length}
          </span>
          <span className="text-[12px] tracking-[0.04em] text-[rgba(232,230,225,0.66)]">m</span>
        </div>
        <span className="text-[11px] text-[rgba(232,230,225,0.6)]">
          Closest pass {setup.clearance} m
        </span>
      </div>

      <div className="glass absolute bottom-7 left-1/2 flex h-14 w-[700px] max-w-[calc(100%-56px)] -translate-x-1/2 items-center justify-between rounded-[18px] pl-3 pr-2">
        <button className="tgl" aria-pressed={setup.routePreview} onClick={setup.toggleRoutePreview}>
          <RouteIcon />
          <span>Route preview</span>
          <span className={`sw ${setup.routePreview ? 'on' : ''}`}>
            <i />
          </span>
        </button>
        <span className="hairline" />
        <div className="seg">
          <button className={setup.tilt === 76 ? 'on' : ''} onClick={() => setup.setTilt(76)}>
            Overhead
          </button>
          <button className={setup.tilt === 52 ? 'on' : ''} onClick={() => setup.setTilt(52)}>
            Angled
          </button>
        </div>
        <span className="hairline" />
        <span className="px-1.5 text-[12px] text-[rgba(232,230,225,0.66)]">Hall 32 × 96 m</span>
        <button className="go" onClick={onStart}>
          <span>Start run</span>
          <ArrowRightIcon size={16} />
        </button>
      </div>
    </div>
  );
}
