import type { ReactNode } from 'react';
import { ArrowRightIcon } from './Icons';
import { formatSigned, formatUnit } from '../sim/frames';
import type { ControllerState } from '../types/sim';

function Toast({
  top,
  ring,
  children,
  className = '',
}: {
  top: number;
  ring?: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <div
      className={`glass absolute left-1/2 w-[380px] max-w-[calc(100%-56px)] -translate-x-1/2 rounded-2xl px-[18px] py-3.5 ${className}`}
      style={{
        top,
        boxShadow: ring
          ? `0 0 0 1px ${ring} inset, 0 12px 32px rgba(0,0,0,0.3)`
          : undefined,
      }}
    >
      {children}
    </div>
  );
}

function CorrectionRow({ label, from, to }: { label: string; from: string; to: string }) {
  return (
    <div
      className="grid h-[34px] items-center border-t border-[rgba(232,230,225,0.08)]"
      style={{ gridTemplateColumns: '84px 1fr 20px 1fr' }}
    >
      <span className="text-[12px] text-[rgba(232,230,225,0.78)]">{label}</span>
      <span className="text-[18px] text-[rgba(232,230,225,0.5)] line-through decoration-1">{from}</span>
      <ArrowRightIcon className="text-[rgba(232,230,225,0.5)]" />
      <span className="text-[18px] font-medium text-bone">{to}</span>
    </div>
  );
}

export function InterventionToast({
  top,
  controller,
}: {
  top: number;
  controller: ControllerState;
}) {
  return (
    <Toast top={top} ring="rgba(242,163,58,0.38)">
      <div className="flex h-4 items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="h-[7px] w-[7px] rounded-full bg-amber" />
          <span className="lbl text-bone">Safety filter · corrected</span>
        </div>
        <span className="text-[11px] text-[rgba(232,230,225,0.66)]">
          {controller.latency_ms == null ? '—' : `${controller.latency_ms} ms`}
        </span>
      </div>
      <div
        className="mt-3 grid h-[18px] items-center"
        style={{ gridTemplateColumns: '84px 1fr 20px 1fr' }}
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
    </Toast>
  );
}

export function StopToast({ top, clearance }: { top: number; clearance: number | null }) {
  return (
    <Toast top={top} ring="rgba(229,72,77,0.45)" className="flex items-center gap-4">
      <div className="flex flex-none flex-col items-start">
        <span className="lbl">Clearance</span>
        <div className="flex items-baseline gap-[5px]">
          <span className="text-[30px] font-light leading-[34px] text-alert">
            {formatUnit(clearance, 1)}
          </span>
          <span className="text-[12px] text-[rgba(232,230,225,0.66)]">m</span>
        </div>
      </div>
      <div className="h-[38px] w-px bg-[rgba(232,230,225,0.12)]" />
      <div className="flex flex-col gap-0.5">
        <span className="text-[14px] font-medium text-bone">Stopped for crossing forklift</span>
        <span className="text-[12px] text-[rgba(232,230,225,0.66)]">
          Holding position while the planner finds a route
        </span>
      </div>
    </Toast>
  );
}

export function ReplanToast({ top }: { top: number }) {
  return (
    <Toast top={top} className="flex flex-col gap-2.5">
      <div className="flex items-center justify-between">
        <span className="text-[14px] font-medium text-bone">New route around the obstruction</span>
        <span className="text-[11px] text-[rgba(232,230,225,0.66)]">38 ms</span>
      </div>
      <div className="flex h-[18px] items-center gap-[18px]">
        <div className="flex items-center gap-2">
          <span className="block h-0 w-5 rounded-sm border-t-[3px] border-amber" />
          <span className="text-[12px] text-[rgba(232,230,225,0.85)]">Selected</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="block h-0 w-5 rounded-sm border-t-2 border-dusk" />
          <span className="text-[12px] text-[rgba(232,230,225,0.85)]">Learned</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="block h-0 w-5 border-t-2 border-dashed border-[#DCD8CE]" />
          <span className="text-[12px] text-[rgba(232,230,225,0.85)]">Fallback</span>
        </div>
      </div>
    </Toast>
  );
}

export function GoalToast({ top }: { top: number }) {
  return (
    <Toast top={top} className="flex items-center justify-between gap-4">
      <div className="flex flex-col gap-0.5">
        <span className="text-[14px] font-medium text-bone">Click the floor to place a goal</span>
        <span className="text-[12px] text-[rgba(232,230,225,0.66)]">5 candidate paths · 20 m</span>
      </div>
      <span className="rounded-[7px] bg-[rgba(232,230,225,0.1)] px-2 py-1 text-[11px] tracking-[0.08em] text-[rgba(232,230,225,0.8)]">
        ESC
      </span>
    </Toast>
  );
}
