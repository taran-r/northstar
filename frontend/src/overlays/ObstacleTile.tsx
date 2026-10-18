import type { ReactElement } from 'react';
import type { ToolId } from '../hooks/useCourseSetup';

function PalletArt() {
  return (
    <svg width="44" height="30" viewBox="0 0 44 30" aria-hidden="true">
      <path d="M4 22l18-8 18 8-18 8z" fill="#6c5337" />
      <path d="M4 22v-4l18 8v4z" fill="#a98258" />
      <path d="M40 22v-4l-18 8v4z" fill="#8a6a46" />
      <path d="M6 15l16-7 16 7-16 7z" fill="#d4aa76" />
      <path d="M10 9l12-5 12 5-12 5z" fill="#cfa676" />
    </svg>
  );
}

function CratesArt() {
  return (
    <svg width="44" height="30" viewBox="0 0 44 30" aria-hidden="true">
      <path d="M3 19l11-5 11 5-11 5z" fill="#cfa676" />
      <path d="M3 19v5l11 5v-5z" fill="#b88e62" />
      <path d="M25 19v5l-11 5v-5z" fill="#8a6a46" />
      <path d="M19 11l11-5 11 5-11 5z" fill="#a6b8cc" />
      <path d="M19 11v5l11 5v-5z" fill="#6f8196" />
      <path d="M41 11v5l-11 5v-5z" fill="#566a80" />
    </svg>
  );
}

function BarrierArt() {
  return (
    <svg width="44" height="30" viewBox="0 0 44 30" aria-hidden="true">
      <path d="M3 17l26-9 12 5-26 9z" fill="#eab932" />
      <path d="M3 17v4l12 5v-4z" fill="#d9aa26" />
      <path d="M41 13v4l-26 9v-4z" fill="#8a6c14" />
      <path d="M12 14l6-2 0 7-6 2z" fill="#16191d" />
      <path d="M24 10l6-2 0 7-6 2z" fill="#16191d" />
    </svg>
  );
}

function ForkliftArt() {
  return (
    <svg width="44" height="30" viewBox="0 0 44 30" aria-hidden="true">
      <path d="M6 20l14-6 14 6-14 7z" fill="#eab048" />
      <path d="M6 20v3l14 7v-3z" fill="#b8812a" />
      <path d="M34 20v3l-14 7v-3z" fill="#8a6218" />
      <path d="M14 11l8-3 8 3-8 3z" fill="#16191d" opacity="0.8" />
      <path d="M36 6v14" stroke="#16191d" strokeWidth="2" />
      <path d="M36 19l6 3" stroke="#16191d" strokeWidth="2" />
    </svg>
  );
}

const ART: Record<ToolId, () => ReactElement> = {
  pallet: PalletArt,
  crates: CratesArt,
  barrier: BarrierArt,
  forklift: ForkliftArt,
};

export function ObstacleTile({
  id,
  label,
  active,
  onPick,
}: {
  id: ToolId;
  label: string;
  active: boolean;
  onPick: () => void;
}) {
  const Art = ART[id];
  return (
    <button
      onClick={onPick}
      className="flex cursor-pointer flex-col items-start gap-2 rounded-xl border-0 px-3 py-2.5 text-left text-[12px] font-medium tracking-[0.01em] text-bone"
      style={
        active
          ? {
              background: 'rgba(242,163,58,0.14)',
              boxShadow: '0 0 0 1px rgba(242,163,58,0.6) inset',
            }
          : { background: 'rgba(232,230,225,0.06)' }
      }
    >
      <Art />
      <span>{label}</span>
    </button>
  );
}
