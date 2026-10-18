import { ChevronDownIcon, ChevronRightIcon } from './Icons';
import { formatSigned, formatUnit } from '../sim/frames';
import type { ControllerState } from '../types/sim';

function Row({ label }: { label: string }) {
  return (
    <button className="flex h-11 w-full items-center justify-between border-0 border-t border-[rgba(232,230,225,0.08)] bg-transparent px-[18px] text-[13px] text-[rgba(232,230,225,0.78)]">
      <span>{label}</span>
      <span className="text-[rgba(232,230,225,0.6)]">
        <ChevronRightIcon />
      </span>
    </button>
  );
}

export function DetailPanel({
  open,
  onToggle,
  controller,
}: {
  open: boolean;
  onToggle: () => void;
  controller: ControllerState;
}) {
  const pct = (v: number) => `${Math.round(Math.max(0, Math.min(1, v)) * 100)}%`;
  const steerPct = (v: number) => `${Math.round((Math.max(-1, Math.min(1, v)) * 0.5 + 0.5) * 100)}%`;
  return (
    <div className="glass absolute right-7 top-[278px] w-[416px] overflow-hidden rounded-[18px]">
      <button
        aria-expanded={open}
        onClick={onToggle}
        className="flex h-[50px] w-full items-center justify-between border-0 bg-transparent px-[18px] text-[13px] font-medium text-bone"
      >
        <span className="flex items-center gap-2.5">
          <span className="h-[7px] w-[7px] rounded-full bg-amber" />
          Safety filter
        </span>
        <span
          className="text-[rgba(232,230,225,0.7)] transition-transform"
          style={{ transform: open ? 'none' : 'rotate(-90deg)' }}
        >
          <ChevronDownIcon />
        </span>
      </button>

      {open && (
        <div className="flex flex-col gap-4 px-[18px] pb-[18px] pt-0.5">
          <div className="flex flex-col gap-3">
            <span className="lbl">Nominal vs corrected</span>

            <div className="flex flex-col gap-1.5">
              <div className="flex items-baseline justify-between">
                <span className="text-[12px] text-[rgba(232,230,225,0.78)]">Throttle</span>
                <span className="text-[12px]">
                  <span className="text-[rgba(232,230,225,0.5)]">
                    {formatUnit(controller.nominal.a)}
                  </span>
                  <span className="text-[rgba(232,230,225,0.5)]"> → </span>
                  <span className="font-medium">{formatUnit(controller.actual.a)}</span>
                </span>
              </div>
              <div className="relative h-1.5 rounded-[3px] bg-[rgba(232,230,225,0.1)]">
                <div
                  className="absolute left-0 top-0 h-1.5 rounded-[3px] bg-amber"
                  style={{ width: pct(controller.actual.a) }}
                />
                <div
                  className="absolute -top-1 h-3.5 w-0.5 rounded-sm bg-[rgba(232,230,225,0.8)]"
                  style={{ left: pct(controller.nominal.a) }}
                />
              </div>
            </div>

            <div className="flex flex-col gap-1.5">
              <div className="flex items-baseline justify-between">
                <span className="text-[12px] text-[rgba(232,230,225,0.78)]">Steering</span>
                <span className="text-[12px]">
                  <span className="text-[rgba(232,230,225,0.5)]">
                    {formatSigned(controller.nominal.steer)}
                  </span>
                  <span className="text-[rgba(232,230,225,0.5)]"> → </span>
                  <span className="font-medium">{formatSigned(controller.actual.steer)}</span>
                </span>
              </div>
              <div className="relative h-1.5 rounded-[3px] bg-[rgba(232,230,225,0.1)]">
                <div
                  className="absolute top-0 h-1.5 bg-amber"
                  style={{
                    left: steerPct(Math.min(controller.actual.steer, 0)),
                    width: `${Math.round(Math.abs(controller.actual.steer) * 50)}%`,
                  }}
                />
                <div
                  className="absolute -top-0.5 h-2.5 w-px bg-[rgba(232,230,225,0.35)]"
                  style={{ left: '50%' }}
                />
                <div
                  className="absolute -top-1 h-3.5 w-0.5 rounded-sm bg-[rgba(232,230,225,0.8)]"
                  style={{ left: steerPct(controller.nominal.steer) }}
                />
              </div>
            </div>

            <div className="flex items-center gap-4">
              <span className="flex items-center gap-1.5 text-[11px] text-[rgba(232,230,225,0.66)]">
                <span className="h-[11px] w-0.5 bg-[rgba(232,230,225,0.8)]" />
                Nominal
              </span>
              <span className="flex items-center gap-1.5 text-[11px] text-[rgba(232,230,225,0.66)]">
                <span className="h-1.5 w-2.5 rounded-sm bg-amber" />
                Corrected
              </span>
            </div>
          </div>

          <div className="h-px bg-[rgba(232,230,225,0.08)]" />

          <div className="flex flex-col gap-2.5">
            <div className="flex items-end justify-between">
              <span className="lbl">Barrier margin</span>
              <div className="flex items-baseline gap-[5px]">
                <span className="text-[26px] font-light leading-7">
                  {formatUnit(controller.barrier_min)}
                </span>
                <span className="text-[12px] text-[rgba(232,230,225,0.66)]">m</span>
              </div>
            </div>
            <svg
              width="380"
              height="64"
              viewBox="0 0 380 64"
              role="img"
              aria-label="Barrier margin over the last six seconds"
              className="block w-full"
            >
              <line
                x1="0"
                y1="52"
                x2="380"
                y2="52"
                stroke="rgba(232,230,225,0.3)"
                strokeWidth="1"
                strokeDasharray="3 4"
              />
              <polyline
                points="0,10 54,12 108,18 162,27 216,38 270,45 324,47 358,43 372,36"
                fill="none"
                stroke="#5B8DEF"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
              <circle cx="372" cy="36" r="4" fill="#F2A33A" />
              <text
                x="0"
                y="63"
                fontSize="9.5"
                fill="rgba(232,230,225,0.66)"
                fontFamily="IBM Plex Sans, system-ui, sans-serif"
                letterSpacing="1.2"
              >
                0 = CONTACT
              </text>
            </svg>
          </div>
        </div>
      )}

      <Row label="Planner" />
      <Row label="Telemetry" />
    </div>
  );
}
