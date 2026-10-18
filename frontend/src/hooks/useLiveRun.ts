import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import type { Obstacle } from '../scene/Obstacles';
import type { Pt } from '../scene/math';
import { toSceneGround, toSceneMeshYaw, toScenePath, toSceneRover } from '../scene/coords';
import { createSocketSource, DEFAULT_WS_URL } from '../sim/socketSource';
import type { SocketSource, SocketStatus } from '../sim/socketSource';
import type { Intent, StateFrame } from '../types/sim';

export type LiveRun = {
  status: SocketStatus;
  connected: boolean;
  frame: StateFrame | null;
  rover: { x: number; z: number; yaw: number } | null;
  route: Pt[] | null;
  candidates: Pt[][];
  obstacles: Obstacle[] | null;
  notice: string | null;
  send: (intent: Intent) => void;
};

const OBSTACLE_KIND: Record<string, string> = {
  forklift: 'forklift',
  worker: 'worker',
  pallet: 'pallet',
  crate: 'crate',
};

/**
 * Subscribe to the planner bridge and project frames into scene coordinates.
 *
 * Conversion goes through `scene/coords`, which is the single tested mapping
 * from the backend frame (z up) to the Three.js frame (y up). Nothing here
 * re-derives it, per `frontend/README.md`.
 *
 * Returns nulls until a frame arrives, so a caller can fall back to its own
 * canned state and the UI still runs with the backend absent.
 */
export function useLiveRun(options?: { url?: string; enabled?: boolean }): LiveRun {
  const enabled = options?.enabled ?? true;
  const url = options?.url ?? DEFAULT_WS_URL;

  const [status, setStatus] = useState<SocketStatus>('closed');
  const [frame, setFrame] = useState<StateFrame | null>(null);
  const sourceRef = useRef<SocketSource | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  useEffect(() => {
    if (!enabled) {
      setStatus('closed');
      return;
    }
    const source = createSocketSource(url);
    sourceRef.current = source;
    const offFrame = source.subscribe(setFrame);
    const offStatus = source.onStatus(setStatus);
    const offControl = source.onControl((message) => {
      if (message.type !== 'ack') return;
      if (message.accepted === false) {
        const intent = String(message.intent ?? 'command');
        setNotice(
          intent === 'set_goal'
            ? 'That point is inside geometry.'
            : intent === 'spawn_obstacle'
              ? 'That spot is blocked.'
              : null,
        );
      } else if (message.intent === 'set_goal' || message.intent === 'spawn_obstacle') {
        setNotice(null);
      }
    });
    return () => {
      offFrame();
      offStatus();
      offControl();
      source.close();
      sourceRef.current = null;
    };
  }, [enabled, url]);

  const rover = useMemo(
    () => (frame ? toSceneRover(frame.rover) : null),
    [frame],
  );

  const route = useMemo(
    () =>
      frame && frame.selected_path.length > 1
        ? toScenePath(frame.selected_path)
        : null,
    [frame],
  );

  const candidates = useMemo(
    () => (frame ? frame.candidate_paths.map((path) => toScenePath(path)) : []),
    [frame],
  );

  const obstacles = useMemo(() => {
    if (!frame) return null;
    return frame.obstacles.map((o): Obstacle => {
      const ground = toSceneGround(o.position[0], o.position[1]);
      return {
        id: o.id,
        t: OBSTACLE_KIND[o.kind ?? ''] ?? (o.shape === 'cylinder' ? 'column' : 'crate'),
        x: ground.x,
        z: ground.z,
        yaw: toSceneMeshYaw(o.yaw ?? 0),
        length: o.size[0],
        width: o.size[1],
        height: o.size[2],
      };
    });
  }, [frame]);

  // Stable so a caller can list it in an effect's dependencies without the
  // effect re-firing every render and restarting the run.
  const send = useCallback((intent: Intent) => {
    sourceRef.current?.send(intent);
  }, []);

  return {
    status,
    connected: status === 'open',
    frame,
    rover,
    route,
    candidates,
    obstacles,
    notice,
    send,
  };
}

/**
 * Whether to connect to the planner bridge, read once from the URL.
 *
 * Opt-in by design: the repo must keep a runnable demo without a backend, and
 * an always-on socket would log connection errors whenever the server is down.
 */
export function useLiveFlag(): boolean {
  return useMemo(() => {
    if (typeof window === 'undefined') return false;
    return new URLSearchParams(window.location.search).has('live');
  }, []);
}
