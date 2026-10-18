import type { Intent, SimSource, StateFrame } from '../types/sim';

export type SocketStatus = 'connecting' | 'open' | 'closed' | 'error';

export type SocketSource = SimSource & {
  status: () => SocketStatus;
  onStatus: (fn: (status: SocketStatus) => void) => () => void;
  onControl: (fn: (message: Record<string, unknown>) => void) => () => void;
  close: () => void;
};

export const DEFAULT_WS_URL = 'ws://localhost:8000/ws';

/**
 * A SimSource backed by the planner bridge at `backend/api/server.py`.
 *
 * Same interface as createMockSource, so screens do not care which one they
 * hold. Frames arrive already in the schema in `../types/sim`: the backend
 * emits StateFrame directly and validates against this file, so there is no
 * translation here beyond dropping the socket's control messages (`hello`,
 * `ack`, `end`, `error`), which are not frames.
 *
 * Intents go up unchanged. The backend additionally accepts a `scenario` field
 * on `start` and a `policy_mode` on `start_replay`.
 *
 * Prefer `start_replay` for a demo: the live path runs the model, the safety
 * filter and the plant for real, but the filter's quadratic program measures
 * ~280 ms per step against a 100 ms control interval on CPU, so it streams
 * slower than real time. A replay streams measured frames at the right pace.
 */
export function createSocketSource(url: string = DEFAULT_WS_URL): SocketSource {
  const listeners = new Set<(frame: StateFrame) => void>();
  const statusListeners = new Set<(status: SocketStatus) => void>();
  const controlListeners = new Set<(message: Record<string, unknown>) => void>();
  const pending: Intent[] = [];

  let socket: WebSocket | null = null;
  let status: SocketStatus = 'connecting';
  let closedByCaller = false;

  const setStatus = (next: SocketStatus) => {
    status = next;
    statusListeners.forEach((fn) => fn(next));
  };

  const connect = () => {
    let ws: WebSocket;
    try {
      ws = new WebSocket(url);
    } catch {
      setStatus('error');
      return;
    }
    socket = ws;
    setStatus('connecting');

    ws.onopen = () => {
      setStatus('open');
      while (pending.length > 0) {
        const intent = pending.shift();
        if (intent) ws.send(JSON.stringify(intent));
      }
    };

    ws.onmessage = (event) => {
      let payload: unknown;
      try {
        payload = JSON.parse(event.data as string);
      } catch {
        return;
      }
      const message = payload as { type?: string };
      if (message.type !== 'state') {
        controlListeners.forEach((fn) => fn(payload as Record<string, unknown>));
        return;
      }
      listeners.forEach((fn) => fn(payload as StateFrame));
    };

    ws.onerror = () => setStatus('error');
    ws.onclose = () => {
      socket = null;
      if (!closedByCaller) setStatus('closed');
    };
  };

  connect();

  return {
    live: true,
    subscribe(onFrame) {
      listeners.add(onFrame);
      return () => listeners.delete(onFrame);
    },
    send(intent) {
      if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify(intent));
      } else {
        pending.push(intent);
      }
    },
    status: () => status,
    onStatus(fn) {
      statusListeners.add(fn);
      fn(status);
      return () => statusListeners.delete(fn);
    },
    onControl(fn) {
      controlListeners.add(fn);
      return () => controlListeners.delete(fn);
    },
    close() {
      closedByCaller = true;
      listeners.clear();
      statusListeners.clear();
      controlListeners.clear();
      socket?.close();
      socket = null;
      setStatus('closed');
    },
  };
}
