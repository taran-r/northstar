import type { Intent, SimSource, StateFrame } from '../types/sim';

export function createMockSource(): SimSource & { publish: (frame: StateFrame) => void } {
  const listeners = new Set<(frame: StateFrame) => void>();
  const sent: Intent[] = [];

  return {
    live: false,
    subscribe(onFrame) {
      listeners.add(onFrame);
      return () => listeners.delete(onFrame);
    },
    send(intent) {
      sent.push(intent);
    },
    publish(frame) {
      listeners.forEach((l) => l(frame));
    },
  };
}
