import type { ControllerState } from '../types/sim';

const EQUATION = {
  fontFamily: 'Georgia, "Iowan Old Style", Palatino, serif',
} as const;

function signed(value: number): string {
  const rounded = Number(value.toFixed(2));
  const text = Math.abs(rounded).toFixed(2);
  return rounded < 0 ? `−${text}` : text;
}

/**
 * The discrete-time barrier the filter actually enforces.
 * The lines stay put; only the measured numbers change.
 */
export function BarrierReadout({ controller }: { controller: ControllerState }) {
  const clearance = controller.barrier_min;
  const nominal = controller.nominal;
  const applied = controller.actual;
  return (
    <div className="glass absolute left-7 top-[72px] w-[268px] rounded-[14px] px-3.5 py-3">
      <div className="lbl text-amber">Discrete-time CBF</div>
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
          a {signed(nominal.a)} · δ {signed(nominal.steer)}
        </dd>
        <dt className="text-[rgba(232,230,225,0.5)]" style={EQUATION}>
          u*
        </dt>
        <dd className="m-0">
          a {signed(applied.a)} · δ {signed(applied.steer)}
        </dd>
      </dl>
    </div>
  );
}
