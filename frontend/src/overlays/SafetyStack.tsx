import { useState } from 'react';
import type { ControllerState } from '../types/sim';
import { formatSigned, formatUnit } from '../sim/frames';
import { ArrowRightIcon } from './Icons';

const EQUATION = {
  fontFamily: 'Georgia, "Iowan Old Style", Palatino, serif',
} as const;

function signed(value: number): string {
  const rounded = Number(value.toFixed(2));
  const text = Math.abs(rounded).toFixed(2);
  return rounded < 0 ? `−${text}` : text;
}

function CorrectionRow({ label, from, to }: { label: string; from: string; to: string }) {
  return (
    <div
      className="grid h-[30px] items-center border-t border-[rgba(232,230,225,0.08)]"
      style={{ gridTemplateColumns: '72px 1fr 16px 1fr' }}
    >
      <span className="text-[11px] text-[rgba(232,230,225,0.78)]">{label}</span>
      <span className="text-[15px] text-[rgba(232,230,225,0.5)] line-through decoration-1">{from}</span>
      <ArrowRightIcon className="text-[rgba(232,230,225,0.5)]" />
      <span className="text-[15px] font-medium text-bone">{to}</span>
    </div>
  );
}

/**
 * Discrete-time CBF card and the corrected-command card, stacked on the left.
 * A single control collapses both so they stop covering the aisle.
 */
export function SafetyStack({ controller }: { controller: ControllerState }) {
  const [open, setOpen] = useState(true);
  const correcting = controller.mode === 'intervention';
  const clearance = controller.barrier_min;

  if (!open) {
    return (
      <button
        type="button"
        className="glass absolute left-7 top-[72px] flex items-center gap-2 rounded-[12px] px-3 py-2 text-left"
        onClick={() => setOpen(true)}
        title="Show safety filter readout"
      >
        <span className="h-[7px] w-[7px] rounded-full bg-amber" />
        <span className="lbl text-amber">CBF</span>
        <span className="text-[12px] tabular-nums text-bone">
          {clearance == null ? '—' : `${signed(clearance)} m`}
        </span>
      </button>
    );
  }

  return (
    <div className="absolute left-7 top-[72px] z-10 flex w-[280px] flex-col gap-2">
      <div className="glass rounded-[14px] px-3.5 py-3">
        <div className="relative flex h-4 items-center pr-6">
          <div className="lbl text-amber">Discrete-time CBF</div>
          <button
            type="button"
            className="absolute -right-1 -top-1 flex h-6 w-6 items-center justify-center rounded-[8px] border-0 bg-transparent text-[rgba(232,230,225,0.55)] hover:bg-[rgba(232,230,225,0.08)] hover:text-bone"
            onClick={() => setOpen(false)}
            title="Minimize"
            aria-label="Minimize"
          >
            <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true">
              <path
                d="M3 7h8"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.7"
                strokeLinecap="round"
              />
            </svg>
          </button>
        </div>
        <div className="mt-2 whitespace-nowrap text-[15px] leading-6 text-bone" style={EQUATION}>
          <div>
            h<sub style={{ fontSize: 11 }}>k+1</sub>
            {' ≥ (1 − γ) h'}
            <sub style={{ fontSize: 11 }}>k</sub>
          </div>
          <div>
            u* = arg min ||u − u<sub style={{ fontSize: 11 }}>nom</sub>||²
          </div>
        </div>
        <div className="mt-1 text-[11px] tracking-wide text-[rgba(232,230,225,0.5)]" style={EQUATION}>
          γ = 0.40 · m = 0.12 m
        </div>
        <dl className="mt-2 grid grid-cols-[52px_1fr] gap-x-2 gap-y-0.5 text-[12px] leading-5 tabular-nums text-bone">
          <dt className="text-[rgba(232,230,225,0.5)]" style={EQUATION}>
            h
          </dt>
          <dd className="m-0">{clearance == null ? '—' : `${signed(clearance)} m`}</dd>
          <dt className="text-[rgba(232,230,225,0.5)]" style={EQUATION}>
            u<span className="text-[10px] align-sub not-italic">nom</span>
          </dt>
          <dd className="m-0">
            a {signed(controller.nominal.a)} · δ {signed(controller.nominal.steer)}
          </dd>
          <dt className="text-[rgba(232,230,225,0.5)]" style={EQUATION}>
            u*
          </dt>
          <dd className="m-0">
            a {signed(controller.actual.a)} · δ {signed(controller.actual.steer)}
          </dd>
        </dl>
      </div>

      {correcting && (
        <div className="glass rounded-[14px] px-3.5 py-3">
          <div className="flex h-4 items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="h-[7px] w-[7px] rounded-full bg-amber" />
              <span className="lbl text-bone">Safety filter · corrected</span>
            </div>
            <span className="text-[11px] text-[rgba(232,230,225,0.66)]">
              {controller.latency_ms == null ? '—' : `${Math.round(controller.latency_ms)} ms`}
            </span>
          </div>
          <div
            className="mt-2 grid h-[16px] items-center"
            style={{ gridTemplateColumns: '72px 1fr 16px 1fr' }}
          >
            <span />
            <span className="lbl">Proposed</span>
            <span />
            <span className="lbl">Corrected</span>
          </div>
          <CorrectionRow
            label="Throttle"
            from={formatUnit(controller.nominal.a)}
            to={formatUnit(controller.actual.a)}
          />
          <CorrectionRow
            label="Steering"
            from={formatSigned(controller.nominal.steer)}
            to={formatSigned(controller.actual.steer)}
          />
        </div>
      )}
    </div>
  );
}
