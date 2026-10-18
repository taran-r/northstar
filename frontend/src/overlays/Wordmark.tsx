import { APP_NAME, SHOW_WORDMARK } from '../branding';
import { StarIcon } from './Icons';

export function Wordmark() {
  if (!SHOW_WORDMARK) return null;
  return (
    <div className="pointer-events-none absolute left-7 top-[27px] flex h-5 items-center gap-2.5">
      <StarIcon />
      <span className="text-[13px] font-medium uppercase leading-5 tracking-[0.34em] text-bone">
        {APP_NAME}
      </span>
    </div>
  );
}
