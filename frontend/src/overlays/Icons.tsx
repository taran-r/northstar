type IconProps = { size?: number; className?: string };

const stroke = {
  fill: 'none' as const,
  stroke: 'currentColor',
  strokeWidth: 1.7,
  strokeLinecap: 'round' as const,
  strokeLinejoin: 'round' as const,
};

export function PauseIcon({ size = 20 }: IconProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
      <rect x="6.5" y="5" width="3.6" height="14" rx="1.2" />
      <rect x="13.9" y="5" width="3.6" height="14" rx="1.2" />
    </svg>
  );
}

export function PlayIcon({ size = 20 }: IconProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
      <path d="M7 5.2v13.6L18 12z" />
    </svg>
  );
}

export function StepIcon({ size = 20 }: IconProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
      <path d="M6 5.5v13l9-6.5z" />
      <rect x="17" y="5.5" width="2.6" height="13" rx="1" />
    </svg>
  );
}

export function ReplayIcon({ size = 20 }: IconProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" {...stroke} aria-hidden="true">
      <path d="M4 12a8 8 0 1 0 2.6-5.9" />
      <path d="M4 4.5v4.2h4.2" />
    </svg>
  );
}

export function NetworkIcon({ size = 20 }: IconProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" {...stroke} aria-hidden="true">
      <circle cx="5" cy="6" r="2" />
      <circle cx="5" cy="18" r="2" />
      <circle cx="19" cy="12" r="2" />
      <path d="M7 6.6l10 4.8" />
      <path d="M7 17.4l10-4.8" />
    </svg>
  );
}

export function ShieldIcon({ size = 20, color = '#F2A33A' }: IconProps & { color?: string }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke={color}
      strokeWidth={1.7}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M12 3.2l7 2.6v5.4c0 4.4-2.9 7.6-7 9.6-4.1-2-7-5.2-7-9.6V5.8z" />
      <path d="M9 12l2.2 2.2L15.2 10" />
    </svg>
  );
}

export function QualityIcon({ size = 18 }: IconProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.7}
      strokeLinecap="round"
      aria-hidden="true"
    >
      <path d="M4 7h9" />
      <path d="M17 7h3" />
      <path d="M4 17h3" />
      <path d="M11 17h9" />
      <circle cx="15" cy="7" r="2" />
      <circle cx="9" cy="17" r="2" />
    </svg>
  );
}

export function SwapIcon({ size = 15 }: IconProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      {...stroke}
      strokeWidth={1.8}
      aria-hidden="true"
    >
      <path d="M7 7h11l-3-3" />
      <path d="M17 17H6l3 3" />
    </svg>
  );
}

export function ArrowRightIcon({ size = 14, className }: IconProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      className={className}
      fill="none"
      stroke="currentColor"
      strokeWidth={2}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M5 12h14" />
      <path d="M13 6l6 6-6 6" />
    </svg>
  );
}

export function ChevronDownIcon({ size = 16 }: IconProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      {...stroke}
      strokeWidth={1.8}
      aria-hidden="true"
    >
      <path d="M6 9l6 6 6-6" />
    </svg>
  );
}

export function ChevronRightIcon({ size = 16 }: IconProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      {...stroke}
      strokeWidth={1.8}
      aria-hidden="true"
    >
      <path d="M9 6l6 6-6 6" />
    </svg>
  );
}

export function CloseIcon({ size = 16 }: IconProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      {...stroke}
      strokeWidth={1.8}
      aria-hidden="true"
    >
      <path d="M6 6l12 12" />
      <path d="M18 6L6 18" />
    </svg>
  );
}

export function StarIcon({ size = 16 }: IconProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden="true">
      <path
        d="M12 1.5c.7 5.4 5.1 9.8 10.5 10.5-5.4.7-9.8 5.1-10.5 10.5-.7-5.4-5.1-9.8-10.5-10.5C6.9 11.3 11.3 6.9 12 1.5z"
        fill="#F2A33A"
      />
    </svg>
  );
}

export function CompareIcon({ size = 20 }: IconProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" {...stroke} aria-hidden="true">
      <rect x="3" y="5" width="18" height="14" rx="2.5" />
      <path d="M12 5v14" />
    </svg>
  );
}

export function RouteIcon({ size = 20 }: IconProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="#F2A33A"
      strokeWidth={1.7}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M5 19c0-6 6-4 7-8s6-2 7-6" />
      <circle cx="5" cy="19" r="1.6" />
      <circle cx="19" cy="5" r="1.6" />
    </svg>
  );
}

export function DiceIcon({ size = 16 }: IconProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      {...stroke}
      strokeWidth={1.8}
      aria-hidden="true"
    >
      <rect x="4" y="4" width="16" height="16" rx="3.5" />
      <circle cx="9" cy="9" r="1" fill="currentColor" />
      <circle cx="15" cy="15" r="1" fill="currentColor" />
      <circle cx="15" cy="9" r="1" fill="currentColor" />
      <circle cx="9" cy="15" r="1" fill="currentColor" />
    </svg>
  );
}

export function RotateIcon({ size = 15 }: IconProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      {...stroke}
      strokeWidth={1.8}
      aria-hidden="true"
    >
      <path d="M20 12a8 8 0 1 1-2.6-5.9" />
      <path d="M20 4.5v4.2h-4.2" />
    </svg>
  );
}

export function TrashIcon({ size = 15 }: IconProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      {...stroke}
      strokeWidth={1.8}
      aria-hidden="true"
    >
      <path d="M5 7h14" />
      <path d="M9 7V5h6v2" />
      <path d="M7 7l1 12h8l1-12" />
    </svg>
  );
}
