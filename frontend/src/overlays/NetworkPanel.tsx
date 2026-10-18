import { CloseIcon } from './Icons';
import type { Conditioning, FlowCurve } from '../types/sim';
import './network.css';

const SPEC = [
  ['Encoder', '2 × 40 × 40 → 128'],
  ['Condition', 'goal, yaw, v, δ → 256'],
  ['Time', 't → 64'],
  ['Field', 'vθ · 3 × 256'],
  ['Integrate', 'Heun on vθ(zₜ, t | c)'],
] as const;

function Heatmap({
  cells,
  size,
  title,
}: {
  cells: number[];
  size: number;
  title: string;
}) {
  const gap = 1;
  const cell = 3;
  const span = size * cell + Math.max(0, size - 1) * gap;
  const ready = cells.length === size * size;
  return (
    <div className="min-w-0">
      <div className="lbl mb-1 h-3.5 truncate">{title}</div>
      <svg width={span} height={span} viewBox={`0 0 ${span} ${span}`} role="img" aria-label={title}>
        {Array.from({ length: size * size }, (_, index) => {
          const column = index % size;
          const row = Math.floor(index / size);
          const occupied = ready && cells[index] === 1;
          return (
            <rect
              key={index}
              x={column * (cell + gap)}
              y={row * (cell + gap)}
              width={cell}
              height={cell}
              rx="0.5"
              fill={occupied ? '#F2A33A' : 'rgba(91,141,239,0.18)'}
            />
          );
        })}
        <circle cx={span / 2} cy={span / 2} r="2.4" fill="#E8E6E1" />
      </svg>
    </div>
  );
}

function Stage({
  curves,
  title,
  range,
}: {
  curves: FlowCurve[];
  title: string;
  range: { minX: number; maxX: number; minY: number; maxY: number };
}) {
  const width = 320;
  const height = 168;
  const pad = 16;
  const spanX = Math.max(1, range.maxX - range.minX);
  const spanY = Math.max(1, range.maxY - range.minY);
  const mapX = (value: number) => pad + ((value - range.minX) / spanX) * (width - pad * 2);
  const mapY = (value: number) => height - pad - ((value - range.minY) / spanY) * (height - pad * 2);
  return (
    <div className="flex min-h-0 min-w-0 flex-1 flex-col">
      <div className="lbl h-4 shrink-0 truncate">{title}</div>
      <svg viewBox={`0 0 ${width} ${height}`} className="mt-1 min-h-0 w-full flex-1" role="img" aria-label={title}>
        {[0.25, 0.5, 0.75].map((mark) => (
          <line
            key={mark}
            x1={pad}
            x2={width - pad}
            y1={pad + mark * (height - pad * 2)}
            y2={pad + mark * (height - pad * 2)}
            stroke="rgba(232,230,225,0.08)"
          />
        ))}
        <line x1={pad} y1={height - pad} x2={width - pad} y2={height - pad} stroke="rgba(232,230,225,0.22)" />
        <line x1={pad} y1={pad} x2={pad} y2={height - pad} stroke="rgba(232,230,225,0.22)" />
        {curves.map((curve) => {
          const d = curve.points
            .map((point, index) => `${index === 0 ? 'M' : 'L'}${mapX(point.x).toFixed(1)} ${mapY(point.y).toFixed(1)}`)
            .join(' ');
          return (
            <path
              key={curve.index}
              d={d}
              fill="none"
              stroke={curve.selected ? '#F2A33A' : '#8eb6ff'}
              strokeWidth={curve.selected ? 2.4 : 1.15}
              strokeOpacity={curve.selected ? 1 : 0.8}
            />
          );
        })}
      </svg>
    </div>
  );
}

function rangeOf(curves: FlowCurve[]) {
  let minX = 0;
  let maxX = 6;
  let minY = -2.5;
  let maxY = 2.5;
  for (const curve of curves) {
    for (const point of curve.points) {
      minX = Math.min(minX, point.x);
      maxX = Math.max(maxX, point.x);
      minY = Math.min(minY, point.y);
      maxY = Math.max(maxY, point.y);
    }
  }
  return { minX, maxX, minY, maxY };
}

export function NetworkPanel({
  curves,
  conditioning,
  source,
  onClose,
}: {
  curves: FlowCurve[];
  conditioning: Conditioning | null;
  source: 'learned' | 'fallback' | null;
  latencyMs: number | null;
  checkpoint: string | null;
  onClose: () => void;
}) {
  const times = Array.from(new Set(curves.map((curve) => curve.t))).sort((a, b) => a - b);
  const at = (t: number) => curves.filter((curve) => Math.abs(curve.t - t) < 0.02);
  const early = times[0] ?? 0;
  const mid = times[Math.floor(times.length / 2)] ?? 0.5;
  const late = times[times.length - 1] ?? 1;
  const range = rangeOf(curves);
  const size = conditioning?.size ?? 20;
  const sourceLabel =
    source === 'learned' ? 'Kept sample is driving' : source === 'fallback' ? 'Hybrid A* detour' : 'Waiting for a plan';

  return (
    <div className="nnr glass absolute inset-x-7 bottom-[96px] flex h-[318px] flex-col overflow-hidden rounded-[18px]">
      <div className="flex h-10 shrink-0 items-center justify-between border-b border-[rgba(232,230,225,0.08)] pl-5 pr-2">
        <div className="flex min-w-0 items-center gap-3">
          <span className="live block h-[7px] w-[7px] shrink-0 rounded-full bg-amber" />
          <span className="shrink-0 text-[13px] font-medium tracking-[0.02em]">Conditional flow matching</span>
          <span
            className="truncate text-[14px] text-[rgba(232,230,225,0.78)]"
            style={{ fontFamily: 'Georgia, "Iowan Old Style", Palatino, serif' }}
          >
            vθ(zₜ, t | c)
          </span>
        </div>
        <button
          aria-label="Close network view"
          onClick={onClose}
          className="flex h-8 w-8 shrink-0 items-center justify-center rounded-[10px] border-0 bg-transparent p-0 text-bone"
        >
          <CloseIcon />
        </button>
      </div>

      <div className="grid min-h-0 flex-1 grid-cols-[248px_44px_minmax(0,1fr)]">
        <div className="flex min-h-0 flex-col justify-center gap-2 overflow-hidden border-r border-[rgba(232,230,225,0.08)] px-3 py-2">
          <div className="flex gap-3">
            <Heatmap cells={conditioning?.facility ?? []} size={size} title="Facility" />
            <Heatmap cells={conditioning?.static ?? []} size={size} title="Static" />
          </div>
          <div className="flex h-4 shrink-0 items-center gap-3 overflow-hidden whitespace-nowrap text-[10px] tracking-[0.12em] text-[rgba(232,230,225,0.55)] uppercase">
            <span className="inline-flex items-center gap-1">
              <span className="inline-block h-2 w-2 bg-amber" /> occupied
            </span>
            <span className="inline-flex items-center gap-1">
              <span className="inline-block h-2 w-2 bg-[rgba(91,141,239,0.7)]" /> free
            </span>
            <span className="inline-flex items-center gap-1">
              <span className="inline-block h-2 w-2 rounded-full bg-bone" /> rover
            </span>
          </div>
          <div className="grid grid-cols-[72px_minmax(0,1fr)] gap-x-2 gap-y-0.5">
            {SPEC.map(([name, detail]) => (
              <div key={name} className="contents">
                <span className="lbl truncate">{name}</span>
                <span className="truncate text-[11px] leading-4 text-[rgba(232,230,225,0.78)]">{detail}</span>
              </div>
            ))}
          </div>
        </div>

        <svg viewBox="0 0 44 220" className="h-full w-full" aria-hidden="true">
          {(
            [
              [28, 36, '#8eb6ff'],
              [48, 70, '#F2A33A'],
              [78, 110, '#8eb6ff'],
              [108, 150, '#c4d8ff'],
              [138, 184, '#8eb6ff'],
            ] as [number, number, string][]
          ).map(([y0, y1, stroke]) => (
            <path
              key={`${y0}-${y1}`}
              d={`M 0 ${y0} C 18 ${y0}, 26 ${y1}, 44 ${y1}`}
              fill="none"
              stroke={stroke}
              strokeWidth="1.2"
              className="fl"
            />
          ))}
        </svg>
        <div className="flex min-h-0 min-w-0 gap-2 overflow-hidden px-3 py-2">
          <Stage curves={at(early)} title={`t = ${early.toFixed(2)} · noise`} range={range} />
          <Stage curves={at(mid)} title={`t = ${mid.toFixed(2)} · midpoint`} range={range} />
          <Stage curves={at(late)} title={`t = ${late.toFixed(2)} · trajectory`} range={range} />
        </div>
      </div>

      <div className="flex h-8 shrink-0 items-center gap-3 overflow-hidden border-t border-[rgba(232,230,225,0.08)] px-5">
        <span className="lbl shrink-0 text-amber">{sourceLabel}</span>
        <span className="truncate text-[12px] text-[rgba(232,230,225,0.72)]">
          Amber is the sample that cleared collision and progress. A detour hands back to the network.
        </span>
      </div>
    </div>
  );
}
