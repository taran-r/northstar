import { useId } from 'react';
import { APP_NAME, SHOW_WORDMARK } from '../branding';

const CX = 16;
const CY = 16;
const RING = 13.4;
// Point lengths clockwise from north. North runs longest so the mark reads as
// a bearing rather than a symmetric sparkle; east and west stop at the bezel
// while north and south cross it.
const TIP = [15.3, 5.7, 13.2, 5.7, 13.6, 5.7, 13.2, 5.7];

function polar(deg: number, r: number) {
  const a = ((deg - 90) * Math.PI) / 180;
  return `${(CX + Math.cos(a) * r).toFixed(2)} ${(CY + Math.sin(a) * r).toFixed(2)}`;
}

/**
 * Each point is split along its axis into two facets. Shading them
 * consistently by rotation -- counter-clockwise half lit, clockwise half in
 * shadow -- is how a nautical compass rose gets its folded-paper depth.
 */
function buildFacets(waist: number) {
  return TIP.map((r, i) => {
    const deg = i * 45;
    const tip = polar(deg, r);
    return {
      lit: `M${CX} ${CY}L${polar(deg - 22.5, waist)}L${tip}Z`,
      shade: `M${CX} ${CY}L${tip}L${polar(deg + 22.5, waist)}Z`,
    };
  });
}

// Needle-thin at display sizes; a touch broader when the points would
// otherwise fall below a pixel.
const FACETS_FINE = buildFacets(2.7);
const FACETS_BOLD = buildFacets(3.3);

const MAJOR = [2, 4, 6];
const MINOR = [1, 3, 5, 7];
const TICKS = [45, 135, 225, 315];

export function NorthstarMark({ size = 22 }: { size?: number }) {
  // useId emits delimiter characters that are awkward inside url(#...) refs.
  const uid = useId().replace(/[^a-zA-Z0-9]/g, '');
  const amberLit = `${uid}-al`;
  const amberShade = `${uid}-as`;
  const boneLit = `${uid}-bl`;
  const glow = `${uid}-gl`;

  // Hold stroke weights near-constant in rendered pixels, easing up a little
  // at display sizes so the hero mark does not look spindly.
  const rendered = (target: number) => ((target * 32) / size).toFixed(2);
  const ringW = rendered(0.8 + (size / 128) * 0.8);
  const hubW = rendered(0.9 + (size / 128) * 0.7);
  const tickW = rendered(0.7 + (size / 128) * 0.6);
  const detail = size >= 36;
  const F = detail ? FACETS_FINE : FACETS_BOLD;

  return (
    <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden="true">
      <defs>
        <linearGradient id={amberLit} x1="16" y1="1" x2="16" y2="16" gradientUnits="userSpaceOnUse">
          <stop offset="0" stopColor="#FFE0B0" />
          <stop offset="0.55" stopColor="#FFC570" />
          <stop offset="1" stopColor="#F2A33A" />
        </linearGradient>
        <linearGradient id={amberShade} x1="16" y1="1" x2="16" y2="16" gradientUnits="userSpaceOnUse">
          <stop offset="0" stopColor="#F0AC46" />
          <stop offset="1" stopColor="#C87D18" />
        </linearGradient>
        <linearGradient id={boneLit} x1="6" y1="6" x2="26" y2="26" gradientUnits="userSpaceOnUse">
          <stop offset="0" stopColor="#FFFFFF" />
          <stop offset="1" stopColor="#DCD9D2" />
        </linearGradient>
        <radialGradient id={glow} cx="16" cy="16" r="15" gradientUnits="userSpaceOnUse">
          <stop offset="0" stopColor="#F2A33A" stopOpacity="0.3" />
          <stop offset="0.45" stopColor="#F2A33A" stopOpacity="0.09" />
          <stop offset="1" stopColor="#F2A33A" stopOpacity="0" />
        </radialGradient>
      </defs>

      <circle cx="16" cy="16" r="15" fill={`url(#${glow})`} />

      <circle
        cx="16"
        cy="16"
        r={RING}
        fill="none"
        stroke="#E8E6E1"
        strokeOpacity="0.2"
        strokeWidth={ringW}
      />

      {detail &&
        TICKS.map((deg) => {
          const a = ((deg - 90) * Math.PI) / 180;
          return (
            <line
              key={deg}
              x1={CX + Math.cos(a) * (RING + 0.8)}
              y1={CY + Math.sin(a) * (RING + 0.8)}
              x2={CX + Math.cos(a) * (RING + 1.7)}
              y2={CY + Math.sin(a) * (RING + 1.7)}
              stroke="#E8E6E1"
              strokeOpacity="0.3"
              strokeWidth={tickW}
              strokeLinecap="round"
            />
          );
        })}

      {detail &&
        MINOR.map((i) => (
          <g key={`m${i}`}>
            <path d={F[i].lit} fill="#E8E6E1" fillOpacity="0.27" />
            <path d={F[i].shade} fill="#E8E6E1" fillOpacity="0.1" />
          </g>
        ))}

      {MAJOR.map((i) => (
        <g key={`M${i}`}>
          <path d={F[i].lit} fill={`url(#${boneLit})`} fillOpacity="0.92" />
          <path d={F[i].shade} fill="#E8E6E1" fillOpacity="0.36" />
        </g>
      ))}

      <path d={F[0].lit} fill={`url(#${amberLit})`} />
      <path d={F[0].shade} fill={`url(#${amberShade})`} />

      <circle cx="16" cy="16" r="1.85" fill="#0E1116" stroke="#F2A33A" strokeWidth={hubW} />
      <circle cx="16" cy="16" r="0.63" fill="#FFD089" />
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
