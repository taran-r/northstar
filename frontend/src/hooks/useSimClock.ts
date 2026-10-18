import { useCallback, useMemo, useRef, useState } from 'react';

export type SimClock = {
  t: number;
  playing: boolean;
};

export type SimClockHandle = {
  ref: React.RefObject<SimClock>;
  playing: boolean;
  runId: number;
  play: () => void;
  pause: () => void;
  toggle: () => void;
  step: () => void;
  restart: () => void;
};

export const STEP_SECONDS = 0.2;

export function useSimClock(initiallyPlaying = true): SimClockHandle {
  const ref = useRef<SimClock>({ t: 0, playing: initiallyPlaying });
  const [playing, setPlaying] = useState(initiallyPlaying);
  const [runId, setRunId] = useState(0);

  const play = useCallback(() => {
    ref.current.playing = true;
    setPlaying(true);
  }, []);

  const pause = useCallback(() => {
    ref.current.playing = false;
    setPlaying(false);
  }, []);

  const toggle = useCallback(() => {
    const next = !ref.current.playing;
    ref.current.playing = next;
    setPlaying(next);
  }, []);

  const step = useCallback(() => {
    ref.current.playing = false;
    ref.current.t += STEP_SECONDS;
    setPlaying(false);
  }, []);

  const restart = useCallback(() => {
    ref.current.t = 0;
    ref.current.playing = true;
    setPlaying(true);
    setRunId((id) => id + 1);
  }, []);

  return useMemo(
    () => ({ ref, playing, runId, play, pause, toggle, step, restart }),
    [playing, runId, play, pause, toggle, step, restart],
  );
}
