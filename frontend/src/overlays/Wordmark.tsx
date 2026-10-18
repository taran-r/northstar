import { APP_NAME, SHOW_WORDMARK } from '../branding';

export function NorthstarMark({ size = 22 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden="true">
      <circle cx="16" cy="16" r="13.4" fill="none" stroke="#E8E6E1" strokeOpacity="0.28" strokeWidth="1" />
      <path d="M16 2.2 19.1 16 16 12.6 12.9 16Z" fill="#F2A33A" />
      <path d="M16 29.8 12.9 16 16 19.4 19.1 16Z" fill="#E8E6E1" fillOpacity="0.78" />
      <path d="M2.2 16 16 12.9 12.6 16 16 19.1Z" fill="#E8E6E1" fillOpacity="0.55" />
      <path d="M29.8 16 16 19.1 19.4 16 16 12.9Z" fill="#E8E6E1" fillOpacity="0.55" />
      <path
        d="M8.2 8.2 16 16 23.8 8.2M8.2 23.8 16 16 23.8 23.8"
        fill="none"
        stroke="#F2A33A"
        strokeOpacity="0.45"
        strokeWidth="0.9"
      />
      <circle cx="16" cy="16" r="1.7" fill="#14171C" stroke="#F2A33A" strokeWidth="1.1" />
    </svg>
  );
}

export function Wordmark({ onClick }: { onClick?: () => void }) {
  if (!SHOW_WORDMARK) return null;
  const mark = (
    <>
      <NorthstarMark />
      <span className="text-[13px] font-medium uppercase leading-5 tracking-[0.28em] text-bone">
        {APP_NAME}
      </span>
    </>
  );
  if (!onClick) {
    return (
      <div className="pointer-events-none absolute left-7 top-[22px] flex h-8 items-center gap-2.5">
        {mark}
      </div>
    );
  }
  return (
    <button
      type="button"
      className="absolute left-7 top-[22px] flex h-8 items-center gap-2.5 border-0 bg-transparent p-0"
      onClick={onClick}
      title="Back to course setup"
      aria-label="Back to course setup"
    >
      {mark}
    </button>
  );
}
