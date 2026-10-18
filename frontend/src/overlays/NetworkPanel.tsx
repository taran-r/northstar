import { useMemo } from 'react';
import { CloseIcon } from './Icons';
import { buildNetworkGraph, type NetworkMode } from './networkGraph';
import './network.css';

export function NetworkPanel({
  mode,
  attentionOnMap,
  onToggleAttention,
  onClose,
}: {
  mode: NetworkMode;
  attentionOnMap: boolean;
  onToggleAttention: () => void;
  onClose: () => void;
}) {
  const graph = useMemo(() => buildNetworkGraph(mode), [mode]);

  return (
    <div className="nnr glass absolute inset-x-7 bottom-[96px] h-[302px] overflow-hidden rounded-[18px]">
      <div className="flex h-[46px] items-center justify-between border-b border-[rgba(232,230,225,0.08)] pl-[22px] pr-3">
        <div className="flex items-center gap-3">
          <span className="live block h-[7px] w-[7px] rounded-full bg-amber" />
          <span className="text-[13px] font-medium tracking-[0.02em]">Policy network</span>
          <span className="text-[12px] text-[rgba(232,230,225,0.66)]">
            Rover policy · 2.1M parameters · 7 layers
          </span>
        </div>
        <div className="flex items-center gap-2">
          <button className="tgl h-8 px-2.5" aria-pressed={attentionOnMap} onClick={onToggleAttention}>
            <span>Attention on map</span>
            <span className={`sw ${attentionOnMap ? 'on' : ''}`}>
              <i />
            </span>
          </button>
          <span className="hairline h-5" />
          <button
            aria-label="Close network view"
            onClick={onClose}
            className="flex h-8 w-8 items-center justify-center rounded-[10px] border-0 bg-transparent p-0 text-bone"
          >
            <CloseIcon />
          </button>
        </div>
      </div>

      <div className="relative h-[216px] w-full">
        <svg
          viewBox="0 0 1544 216"
          preserveAspectRatio="xMidYMid meet"
          className="absolute inset-0 block h-full w-full"
          aria-label="How the policy network turns the local view into one of seven actions"
          role="img"
        >
          <defs>
            <linearGradient id="nn-sweep" x1="0" x2="1" y1="0" y2="0">
              <stop offset="0" stopColor="#F2A33A" stopOpacity="0" />
              <stop offset="0.7" stopColor="#F2A33A" stopOpacity="0.12" />
              <stop offset="1" stopColor="#FFE2B0" stopOpacity="0.26" />
            </linearGradient>
          </defs>
          {graph.shapes.map((s, i) => (
            <path
              key={i}
              className={s.cls}
              style={s.style}
              d={s.d}
              fill={s.fill}
              fillOpacity={s.fo}
              stroke={s.stroke}
              strokeWidth={s.sw}
              strokeOpacity={s.so}
              strokeLinecap="round"
            />
          ))}
          <rect
            className="sweep"
            x="0"
            y="8"
            width="220"
            height="200"
            fill="url(#nn-sweep)"
            style={{ mixBlendMode: 'screen' }}
          />
          {graph.heads.map((h, i) => (
            <text key={`h${i}`} x={h.x} y={h.y + 10} className="nn-head">
              {h.text}
            </text>
          ))}
          {graph.subs.map((h, i) => (
            <text key={`s${i}`} x={h.x} y={h.y + 9} className="nn-sub">
              {h.text}
            </text>
          ))}
          {graph.outputs.map((o, i) => (
            <g key={`o${i}`} transform={`translate(1200 ${o.top})`}>
              <text
                x="0"
                y="13"
                className="nn-out"
                fill={o.color}
                style={{ fontWeight: o.weight }}
              >
                {o.name}
              </text>
              <rect x="128" y="7" width="130" height="4" rx="2" fill="rgba(232,230,225,0.08)" />
              <rect className="bar" x="128" y="7" width={o.w} height="4" rx="2" fill={o.bar} />
              <text
                x="306"
                y="13"
                textAnchor="end"
                className="nn-out"
                fill={o.color}
                style={{ fontWeight: o.weight }}
              >
                {o.pct}%
              </text>
            </g>
          ))}
        </svg>
      </div>

      <div className="flex h-10 items-center justify-between border-t border-[rgba(232,230,225,0.08)] px-[22px]">
        <span className="text-[12px] text-[rgba(232,230,225,0.8)]">
          <span className="lbl mr-3 inline text-amber">Why</span>
          {graph.why}
        </span>
        <span className="text-[12px] text-[rgba(232,230,225,0.66)]">{graph.foot}</span>
      </div>
    </div>
  );
}
