import { useCallback, useEffect, useMemo, useState } from 'react';

import { WorldCanvas, type CameraMode, type QualityLevel } from '../scene/WorldCanvas';
import type { FacilityGeometry } from '../scene/FacilityHall';
import { Dock } from '../overlays/Dock';
import { StatusChip } from '../overlays/StatusChip';
import { Wordmark } from '../overlays/Wordmark';
import { InterventionToast } from '../overlays/Toasts';
import { useLiveRun } from '../hooks/useLiveRun';
import { toSceneGround, toSceneMeshYaw, toScenePath } from '../scene/coords';
import type { Pt } from '../scene/math';
import type { ToolId } from '../hooks/useCourseSetup';
import { TOOL_NAMES } from '../hooks/useCourseSetup';

const TOOLS: ToolId[] = ['pallet', 'crates', 'barrier', 'forklift'];
const API = 'http://localhost:8000';

type PlaceTool = 'goal' | ToolId;

function readNumber(value: number | null | undefined, digits = 2): string {
  if (value == null || Number.isNaN(value)) return '—';
  return value.toFixed(digits);
}

export function LiveScreen() {
  const live = useLiveRun({ enabled: true });
  const [facility, setFacility] = useState<FacilityGeometry | null>(null);
  const [checkpoint, setCheckpoint] = useState<string | null | undefined>(undefined);
  const [generation, setGeneration] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [safety, setSafety] = useState(true);
  const [camera, setCamera] = useState<CameraMode>('course');
  const [quality, setQuality] = useState<QualityLevel>('Balanced');
  const [tool, setTool] = useState<PlaceTool>('goal');
  const [yaw, setYaw] = useState(0);
  const [panelOpen, setPanelOpen] = useState(false);
  const [selected, setSelected] = useState(-1);
  const [planMs, setPlanMs] = useState<number | null>(null);

  useEffect(() => {
    let cancel = false;
    fetch(`${API}/facility`)
      .then((response) => response.json())
      .then((payload: FacilityGeometry) => {
        if (!cancel) setFacility(payload);
      })
      .catch(() => {
        if (!cancel) setFacility(null);
      });
    fetch(`${API}/health`)
      .then((response) => response.json())
      .then((payload: { checkpoint_present?: boolean; checkpoint?: string | null }) => {
        if (cancel) return;
        setCheckpoint(payload.checkpoint_present ? (payload.checkpoint ?? null) : null);
      })
      .catch(() => {
        if (!cancel) setCheckpoint(null);
      });
    return () => {
      cancel = true;
    };
  }, []);

  useEffect(() => {
    if (!live.connected) return;
    live.send({ type: 'start_interactive', seed: 1 });
    setPlaying(true);
    setSelected(-1);
  }, [live.connected, live.send, generation]);

  useEffect(() => {
    const frame = live.frame;
    if (!frame || frame.controller.mode !== 'replanning') return;
    if (frame.controller.latency_ms != null) setPlanMs(frame.controller.latency_ms);
  }, [live.frame]);

  const frame = live.frame;
  const driven = useMemo(() => {
    if (!frame) return null;
    const ground = toSceneGround(frame.rover.x, frame.rover.y);
    return { x: ground.x, z: ground.z, yaw: toSceneMeshYaw(frame.rover.heading) };
  }, [frame]);

  const goal = useMemo(
    () => (frame?.goal ? toSceneGround(frame.goal[0], frame.goal[1]) : null),
    [frame],
  );
  const route = useMemo(
    () => (frame && frame.selected_path.length > 1 ? toScenePath(frame.selected_path) : null),
    [frame],
  );
  const candidates = useMemo(
    () => (frame ? frame.candidate_paths.map((path) => toScenePath(path)) : []),
    [frame],
  );
  const fallback = useMemo(
    () =>
      frame && frame.fallback_paths.length > 1 ? toScenePath(frame.fallback_paths) : null,
    [frame],
  );
  const trail = useMemo(() => {
    if (!frame || frame.executed_trail.length < 2) return null;
    const recent = frame.executed_trail.slice(-180);
    return toScenePath(recent);
  }, [frame]);

  const mode = frame?.controller.mode ?? 'tracking';
  const ring = mode === 'estop' ? 'stop' : mode === 'intervention' ? 'warn' : 'ok';
  const focus = facility
    ? { x: facility.width_x / 2, z: -facility.depth_y / 2 }
    : { x: 30, z: -20 };

  const onFloor = useCallback(
    (point: Pt) => {
      const x = point.x;
      const y = -point.z;
      if (tool === 'goal') {
        live.send({ type: 'set_goal', x, y, z: 0 });
        return;
      }
      live.send({ type: 'spawn_obstacle', kind: tool, x, y, yaw });
    },
    [live, tool, yaw],
  );

  const togglePlay = useCallback(() => {
    live.send({ type: playing ? 'pause' : 'resume' });
    setPlaying((value) => !value);
  }, [live, playing]);

  const step = useCallback(() => {
    setPlaying(false);
    live.send({ type: 'step' });
  }, [live]);

  const toggleSafety = useCallback(() => {
    const enabled = !safety;
    setSafety(enabled);
    live.send({ type: 'toggle_safety', enabled });
  }, [live, safety]);

  const cycleQuality = useCallback(() => {
    setQuality((value) => (value === 'High' ? 'Balanced' : value === 'Balanced' ? 'Fast' : 'High'));
  }, []);

  const waiting = !frame?.goal;
  const source = frame?.controller.source;
  const plannerLabel = waiting
    ? 'Waiting for a destination'
    : source === 'learned'
      ? 'Flow matching'
      : 'Hybrid A*';

  return (
    <div className="relative h-full w-full overflow-hidden bg-[#0b0c0f]">
      <WorldCanvas
        className="absolute inset-0"
        state="hero"
        cameraMode={camera}
        quality={quality}
        facility={facility}
        focus={focus}
        driven={driven}
        ring={ring}
        goalPoint={goal}
        route={route}
        candidates={candidates}
        fallback={fallback}
        trail={trail}
        obstacles={live.obstacles ?? []}
        selected={selected}
        onFloorClick={onFloor}
        onPickObstacle={setSelected}
        labels={false}
      />

      <div
        className="pointer-events-none absolute inset-x-0 top-0 h-[150px]"
        style={{ background: 'linear-gradient(180deg, rgba(8,9,12,0.66), rgba(8,9,12,0))' }}
      />
      <div
        className="pointer-events-none absolute inset-x-0 bottom-0 h-[190px]"
        style={{ background: 'linear-gradient(0deg, rgba(8,9,12,0.7), rgba(8,9,12,0))' }}
      />

      <Wordmark />
      <StatusChip kind={mode} placement="center" />

      {!live.connected && (
        <div className="glass absolute left-1/2 top-24 -translate-x-1/2 rounded-2xl px-4 py-3 text-[13px] text-bone">
          {live.status === 'connecting' ? 'Connecting to the simulator…' : 'Simulator is not running.'}
        </div>
      )}

      {live.notice && (
        <div className="glass absolute left-1/2 top-24 -translate-x-1/2 rounded-2xl px-4 py-3 text-[13px] text-bone">
          {live.notice}
        </div>
      )}

      {mode === 'intervention' && frame && (
        <InterventionToast top={128} controller={frame.controller} />
      )}
      {mode === 'estop' && frame && (
        <div className="glass absolute left-1/2 top-[128px] w-[380px] max-w-[calc(100%-56px)] -translate-x-1/2 rounded-2xl px-[18px] py-3.5">
          <div className="text-[14px] font-medium text-bone">Emergency stop</div>
          <div className="mt-1 text-[12px] text-[rgba(232,230,225,0.7)]">
            Clearance {readNumber(frame.metrics.min_clearance, 2)} m. The filter could not certify a command.
          </div>
        </div>
      )}
      {mode === 'replanning' && frame && (
        <div className="glass absolute left-1/2 top-[128px] w-[420px] max-w-[calc(100%-56px)] -translate-x-1/2 rounded-2xl px-[18px] py-3.5">
          <div className="flex items-center justify-between">
            <span className="text-[14px] font-medium text-bone">Replanning</span>
            <span className="text-[12px] text-[rgba(232,230,225,0.7)]">
              {frame.controller.source === 'learned' ? 'Flow matching' : 'Hybrid A*'}
              {frame.controller.latency_ms != null
                ? ` · ${Math.round(frame.controller.latency_ms)} ms`
                : ''}
            </span>
          </div>
          <div className="mt-2 flex gap-4 text-[12px] text-[rgba(232,230,225,0.8)]">
            <span>Candidates {frame.candidate_paths.length}</span>
            <span className="text-amber">Selected</span>
            <span>Fallback</span>
          </div>
        </div>
      )}
      {mode === 'goal' && frame && (
        <div className="glass absolute left-1/2 top-[128px] -translate-x-1/2 rounded-2xl px-[18px] py-3.5">
          <div className="text-[14px] font-medium text-bone">Goal reached</div>
          <div className="mt-1 text-[12px] text-[rgba(232,230,225,0.7)]">
            Path efficiency {readNumber(frame.metrics.path_efficiency, 2)} · collisions{' '}
            {frame.metrics.collisions ?? '—'}
          </div>
        </div>
      )}

      <div className="glass absolute left-1/2 top-[68px] flex max-w-[calc(100%-28px)] -translate-x-1/2 items-center gap-2 overflow-x-auto rounded-[18px] px-2 py-2">
        <button
          className={`btn h-9 shrink-0 px-3 ${tool === 'goal' ? 'btn-amber' : ''}`}
          onClick={() => setTool('goal')}
        >
          Destination
        </button>
        {TOOLS.map((id) => (
          <button
            key={id}
            className={`btn h-9 shrink-0 px-2.5 ${tool === id ? 'btn-amber' : ''}`}
            onClick={() => setTool(id)}
          >
            {TOOL_NAMES[id]}
          </button>
        ))}
        <button className="btn h-9 shrink-0 px-2.5" onClick={() => setYaw((value) => value + Math.PI / 2)}>
          Rotate
        </button>
        <button
          className="btn h-9 shrink-0 px-2.5"
          onClick={() => {
            const id = live.obstacles?.[selected]?.id;
            if (id) live.send({ type: 'remove_obstacle', id });
            setSelected(-1);
          }}
        >
          Remove
        </button>
        <button
          className="btn h-9 shrink-0 px-2.5"
          onClick={() => {
            live.send({ type: 'clear_obstacles' });
            setSelected(-1);
          }}
        >
          Clear
        </button>
      </div>

      {panelOpen && (
        <div className="glass absolute right-7 top-[84px] w-[300px] rounded-[18px] p-4 text-[13px] text-bone">
          <div className="lbl text-bone">Planner</div>
          <div className="mt-2 text-[15px] font-medium">{plannerLabel}</div>
          <div className="mt-1 text-[12px] text-[rgba(232,230,225,0.66)]">
            {checkpoint === undefined
              ? 'Checking checkpoint…'
              : checkpoint
                ? `Checkpoint ${checkpoint}`
                : 'No checkpoint loaded. Hybrid A* is the planner.'}
          </div>
          <dl className="mt-3 grid grid-cols-2 gap-y-1.5 text-[12px]">
            <dt className="text-[rgba(232,230,225,0.6)]">Candidates</dt>
            <dd className="text-right">{frame ? frame.candidate_paths.length : '—'}</dd>
            <dt className="text-[rgba(232,230,225,0.6)]">Replans</dt>
            <dd className="text-right">{frame?.metrics.replans ?? '—'}</dd>
            <dt className="text-[rgba(232,230,225,0.6)]">Interventions</dt>
            <dd className="text-right">{frame?.metrics.interventions ?? '—'}</dd>
            <dt className="text-[rgba(232,230,225,0.6)]">Collisions</dt>
            <dd className="text-right">{frame?.metrics.collisions ?? '—'}</dd>
            <dt className="text-[rgba(232,230,225,0.6)]">Min clearance</dt>
            <dd className="text-right">{readNumber(frame?.metrics.min_clearance ?? null)} m</dd>
            <dt className="text-[rgba(232,230,225,0.6)]">Last plan</dt>
            <dd className="text-right">{planMs == null ? '—' : `${Math.round(planMs)} ms`}</dd>
            <dt className="text-[rgba(232,230,225,0.6)]">Sim time</dt>
            <dd className="text-right">{readNumber(frame?.sim_time, 1)} s</dd>
          </dl>
          <div className="mt-3 flex flex-col gap-1 text-[11px] text-[rgba(232,230,225,0.7)]">
            <span><span className="text-[#9ec3ff]">━</span> Candidate</span>
            <span><span className="text-amber">━</span> Selected flow-matching path</span>
            <span><span className="text-[#F2EFE6]">┄</span> Hybrid A* fallback</span>
          </div>
        </div>
      )}

      <Dock
        variant="floating"
        playing={playing}
        onTogglePlay={togglePlay}
        onStep={step}
        safety={safety}
        onToggleSafety={toggleSafety}
        networkOpen={panelOpen}
        onToggleNetwork={() => setPanelOpen((value) => !value)}
        onReplay={() => setGeneration((value) => value + 1)}
        onCompare={() => undefined}
        showCompare={false}
        camera={camera}
        onCamera={setCamera}
        quality={quality}
        onCycleQuality={cycleQuality}
        speed={readNumber(frame?.rover.speed ?? null, 2)}
        progress={0}
      />
    </div>
  );
}
