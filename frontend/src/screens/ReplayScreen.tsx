import { WorldCanvas } from '../scene/WorldCanvas';
import { Wordmark } from '../overlays/Wordmark';
import { CloseIcon } from '../overlays/Icons';
import {
  METRIC_COLUMNS,
  REPLAY_METRICS,
  REPLAY_SEED,
  type MetricRow,
} from '../data/replayMetrics';

const GRID = '104px repeat(6, minmax(0, 1fr))';

function toneClass(tone?: string) {
  if (tone === 'bad') return 'text-alert font-medium';
  if (tone === 'good') return 'text-go font-medium';
  if (tone === 'muted') return 'text-[rgba(232,230,225,0.7)]';
  return '';
}

function Row({ row }: { row: MetricRow }) {
  const cells: [string, string | undefined][] = [
    [row.goal, row.goalTone],
    [row.collisions, row.collisionTone],
    [row.clearance, row.clearanceTone],
    [row.efficiency, undefined],
    [row.latency, undefined],
    [row.interventions, undefined],
  ];
  return (
    <div
      className="grid h-8 items-center border-t border-[rgba(232,230,225,0.08)]"
      style={{ gridTemplateColumns: GRID }}
    >
      <span className="text-[12px] text-[rgba(232,230,225,0.78)]">{row.label}</span>
      {cells.map(([value, tone], i) => (
        <span key={i} className={`text-[15px] ${toneClass(tone)}`}>
          {value}
        </span>
      ))}
    </div>
  );
}

function SideChip({ side, label, color }: { side: 'left' | 'right'; label: string; color: string }) {
  return (
    <div
      className="glass absolute top-[84px] flex h-[30px] -translate-x-1/2 items-center gap-2.5 rounded-[15px] px-3.5"
      style={{ left: side === 'left' ? '25%' : '75%' }}
    >
      <span className="h-[7px] w-[7px] rounded-full" style={{ background: color }} />
      <span className="lbl text-bone">{label}</span>
    </div>
  );
}

export function ReplayScreen({ onExit }: { onExit: () => void }) {
  return (
    <div className="relative h-full w-full overflow-hidden bg-[#0b0c0f]">
      <div className="absolute inset-y-0 left-0 w-1/2 overflow-hidden">
        <WorldCanvas className="absolute inset-0" state="replayOff" cameraMode="chase" detail="lo" labels={false} />
      </div>
      <div className="absolute inset-y-0 right-0 w-1/2 overflow-hidden">
        <WorldCanvas className="absolute inset-0" state="replayOn" cameraMode="chase" detail="lo" labels={false} />
      </div>

      <div className="pointer-events-none absolute inset-y-0 left-1/2 w-0.5 -translate-x-1/2 bg-[rgba(232,230,225,0.55)]" />
      <div className="glass pointer-events-none absolute left-1/2 top-1/2 flex h-9 w-9 -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-full">
        <svg
          width="18"
          height="18"
          viewBox="0 0 24 24"
          fill="none"
          stroke="#E8E6E1"
          strokeWidth="1.7"
          strokeLinecap="round"
          strokeLinejoin="round"
          aria-hidden="true"
        >
          <path d="M8 7l-5 5 5 5" />
          <path d="M16 7l5 5-5 5" />
        </svg>
      </div>

      <div
        className="pointer-events-none absolute inset-x-0 top-0 h-[170px]"
        style={{ background: 'linear-gradient(180deg, rgba(8,9,12,0.66), rgba(8,9,12,0))' }}
      />
      <div
        className="pointer-events-none absolute inset-x-0 bottom-0 h-[190px]"
        style={{ background: 'linear-gradient(0deg, rgba(8,9,12,0.7), rgba(8,9,12,0))' }}
      />

      <Wordmark />

      <div className="glass absolute left-1/2 top-[26px] flex h-9 -translate-x-1/2 items-center gap-4 rounded-[18px] px-[18px]">
        <span className="lbl text-bone">Replay</span>
        <span className="text-[12px] text-[rgba(232,230,225,0.66)]">Seed {REPLAY_SEED}</span>
        <div className="relative h-[3px] w-[150px] rounded-sm bg-[rgba(232,230,225,0.14)]">
          <div className="h-[3px] w-[44%] rounded-sm bg-bone" />
          <div className="absolute -top-1 -ml-[5px] h-[11px] w-[11px] rounded-full bg-bone" style={{ left: '44%' }} />
        </div>
        <span className="text-[12px]">
          00:14<span className="text-[rgba(232,230,225,0.5)]"> / 00:32</span>
        </span>
        <button
          aria-label="Close replay"
          onClick={onExit}
          className="flex h-7 w-7 items-center justify-center rounded-[9px] border-0 bg-transparent p-0 text-bone"
        >
          <CloseIcon />
        </button>
      </div>

      <SideChip side="left" label="Safety off" color="#E5484D" />
      <SideChip side="right" label="Safety on" color="#F2A33A" />

      <div className="glass absolute bottom-[104px] left-1/2 w-[920px] max-w-[calc(100%-56px)] -translate-x-1/2 rounded-2xl px-[22px] pb-3.5 pt-3">
        <div className="grid h-[26px] items-center" style={{ gridTemplateColumns: GRID }}>
          <span />
          {METRIC_COLUMNS.map((c) => (
            <span key={c} className="lbl">
              {c}
            </span>
          ))}
        </div>
        {REPLAY_METRICS.map((row) => (
          <Row key={row.label} row={row} />
        ))}
      </div>
    </div>
  );
}
