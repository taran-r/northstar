export type ChipKind = 'tracking' | 'intervention' | 'estop' | 'replanning' | 'goal';

type ChipDef = { label: string; color: string; rgb: string; ring?: boolean };

export const CHIPS: Record<ChipKind, ChipDef> = {
  tracking: { label: 'Tracking', color: '#5B8DEF', rgb: '91,141,239' },
  intervention: { label: 'Intervention', color: '#F2A33A', rgb: '242,163,58' },
  estop: { label: 'Emergency stop', color: '#E5484D', rgb: '229,72,77' },
  replanning: { label: 'Replanning', color: '#5B8DEF', rgb: '91,141,239', ring: true },
  goal: { label: 'Goal reached', color: '#4CAF7D', rgb: '76,175,125' },
};

export function StatusChip({ kind, placement }: { kind: ChipKind; placement: 'center' | 'left' }) {
  const chip = CHIPS[kind];
  const isStop = kind === 'estop';
  const position =
    placement === 'left'
      ? 'left-7 top-[62px]'
      : 'left-1/2 top-[26px] -translate-x-1/2';

  return (
    <div
      className={`glass absolute ${position} flex h-[34px] items-center gap-2.5 rounded-[17px] pl-3.5 pr-4`}
      style={{
        background: isStop ? 'rgba(60,20,24,0.66)' : 'rgba(20,23,28,0.62)',
        boxShadow: `0 0 0 1px rgba(${chip.rgb},${isStop ? 0.5 : 0.32}) inset, 0 12px 32px rgba(0,0,0,0.28)`,
      }}
    >
      {chip.ring ? (
        <span
          className="pulse-dot block h-2.5 w-2.5 rounded-full border-2 box-border"
          style={{ borderColor: chip.color }}
        />
      ) : (
        <span className="block h-2 w-2 rounded-full" style={{ background: chip.color }} />
      )}
      <span className="text-[12px] font-medium uppercase leading-4 tracking-[0.16em] text-bone">
        {chip.label}
      </span>
    </div>
  );
}
