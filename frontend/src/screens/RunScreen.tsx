import { useCallback, useEffect } from 'react';

import { WorldCanvas } from '../scene/WorldCanvas';
import { Dock } from '../overlays/Dock';
import { DetailPanel } from '../overlays/DetailPanel';
import { PipView } from '../overlays/PipView';
import { StatusChip } from '../overlays/StatusChip';
import { GoalToast, InterventionToast, ReplanToast, StopToast } from '../overlays/Toasts';
import { Wordmark } from '../overlays/Wordmark';
import { SceneSwitcher } from '../overlays/SceneSwitcher';
import { NetworkPanel } from '../overlays/NetworkPanel';
import { useRunState } from '../hooks/useRunState';
import { useMediaQuery } from '../hooks/useMediaQuery';
import type { CourseSetup } from '../hooks/useCourseSetup';
import { frameFor } from '../sim/frames';
import { useLiveRun, useLiveFlag } from '../hooks/useLiveRun';

export function RunScreen({
  course,
  onReplay,
  onSetup,
}: {
  course: CourseSetup;
  onReplay: () => void;
  onSetup: () => void;
}) {
  const run = useRunState();
  const { def } = run;
  const showCornerView = useMediaQuery('(min-width: 1180px)');

  // Opt-in so the canned demo keeps working byte-identically when the planner
  // bridge is not running: add ?live=1 to stream real frames from
  // backend/api/server.py. Falls back to the canned state per-field, so a
  // dropped socket degrades instead of blanking the scene.
  const live = useLiveRun({ enabled: useLiveFlag() });

  // Ask the bridge for frames. Until an intent goes up the server has no
  // worker running, so an opened socket alone streams nothing.
  //
  // Replay rather than live: the live path runs the model, the ranking and the
  // safety filter for real, but the filter's quadratic program measures
  // ~280 ms per step against a 100 ms control interval on CPU, so it arrives
  // slower than real time. A replay streams measured frames at the right pace.
  //
  // The two safety arms are two recorded logs, so changing `safety` restarts
  // the stream instead of switching mid-run - there is no mid-stream swap to
  // make, and pretending otherwise would show frames from the wrong arm.
  // `runId` is here so the dock's replay button relaunches the run too.
  useEffect(() => {
    if (!live.connected) return;
    live.send({
      type: 'start_replay',
      seed: course.seed,
      policy_mode: run.safety ? 'safety_on' : 'safety_off',
    });
  }, [live.connected, live.send, course.seed, run.safety, run.runId]);

  // The clock still drives the canned scene, so both move together: local
  // playback for the rendering, the intent for the frame producer.
  const togglePlay = useCallback(() => {
    if (live.connected) live.send({ type: run.playing ? 'pause' : 'resume' });
    run.togglePlay();
  }, [live.connected, live.send, run.playing, run.togglePlay]);

  const step = useCallback(() => {
    if (live.connected) live.send({ type: 'step' });
    run.step();
  }, [live.connected, live.send, run.step]);

  // No `toggle_safety` intent here on purpose: the effect above relaunches the
  // run on the new arm, and that already sets `use_cbf` server-side.
  const obstacles = live.obstacles ?? course.obstacles;
  const route = live.route ?? (def.live ? course.plannedRoute : null);
  const candidates =
    live.candidates.length > 0
      ? live.candidates
      : def.live
        ? course.plannedCandidates
        : [];

  const frame = frameFor({
    world: def.world,
    mode: def.chip,
    seed: course.seed,
    speed: Number(run.speed),
    obstacles,
    route,
    candidates,
    simTime: live.frame ? live.frame.sim_time : run.progress * 44,
  });
  const wideEnoughForSwitcher = useMediaQuery('(min-width: 1024px)');
  const showSwitcher =
    wideEnoughForSwitcher &&
    typeof window !== 'undefined' &&
    new URLSearchParams(window.location.search).has('states');

  const mainCamera = run.swapped ? 'course' : run.camera;
  const pipCamera = run.swapped ? 'chase' : 'course';
  const toastTop = run.variant === 'rail' ? 26 : 78;
  const chipPlacement = run.variant === 'rail' ? 'left' : 'center';

  return (
    <div className="relative h-full w-full overflow-hidden bg-[#0b0c0f]">
      <div
        className="absolute inset-0"
        style={{ transform: def.network ? 'translateY(-96px)' : undefined }}
      >
        <WorldCanvas
          className="absolute inset-0"
          state={def.world}
          cameraMode={mainCamera}
          clock={def.live ? run.clock : undefined}
          runId={run.runId}
          attention={!!def.network && run.attentionOnMap}
          quality={run.quality}
          obstacles={obstacles}
          route={route}
          candidates={candidates}
          onTick={run.setProgress}
        />
      </div>

      {def.toast === 'stop' && (
        <div
          className="pointer-events-none absolute inset-0"
          style={{ boxShadow: 'inset 0 0 180px rgba(229,72,77,0.22)' }}
        />
      )}

      <div
        className="pointer-events-none absolute inset-x-0 top-0 h-[170px]"
        style={{ background: 'linear-gradient(180deg, rgba(8,9,12,0.66), rgba(8,9,12,0))' }}
      />
      <div
        className="pointer-events-none absolute inset-x-0 bottom-0 h-[190px]"
        style={{ background: 'linear-gradient(0deg, rgba(8,9,12,0.7), rgba(8,9,12,0))' }}
      />

      <Wordmark />
      <StatusChip kind={run.chip} placement={chipPlacement} />

      {showCornerView && (
      <PipView
        state={def.world}
        cameraMode={pipCamera}
        obstacles={obstacles}
        route={route}
        candidates={candidates}
        label={run.swapped ? 'ROVER VIEW' : 'COURSE VIEW'}
        clock={def.live ? run.clock : undefined}
        runId={run.runId}
        onSwap={run.toggleSwap}
      />
      )}

      {def.toast === 'int' && <InterventionToast top={toastTop} controller={frame.controller} />}
      {def.toast === 'stop' && (
        <StopToast top={toastTop} clearance={frame.metrics.min_clearance} />
      )}
      {def.toast === 'replan' && <ReplanToast top={toastTop} />}
      {def.toast === 'goal' && <GoalToast top={toastTop} />}

      {def.panel && (
        <DetailPanel
          open={run.panelOpen}
          onToggle={run.togglePanel}
          controller={frame.controller}
        />
      )}

      {def.network && (
        <NetworkPanel
          mode={def.network}
          attentionOnMap={run.attentionOnMap}
          onToggleAttention={run.toggleAttentionOnMap}
          onClose={run.toggleNetwork}
        />
      )}

      <Dock
        variant={run.variant}
        playing={run.playing}
        onTogglePlay={togglePlay}
        onStep={step}
        safety={run.safety}
        onToggleSafety={run.toggleSafety}
        networkOpen={!!def.network}
        onToggleNetwork={run.toggleNetwork}
        onReplay={run.restart}
        onCompare={onReplay}
        camera={run.camera}
        onCamera={run.setCamera}
        quality={run.quality}
        onCycleQuality={run.cycleQuality}
        speed={run.speed}
        progress={run.progress}
      />

      {showSwitcher && (
      <SceneSwitcher
        scene={run.scene}
        onScene={run.setScene}
        variant={run.variant}
        onVariant={run.setVariant}
        onSetup={onSetup}
        onCompare={onReplay}
      />
      )}
    </div>
  );
}
