import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import { WorldCanvas } from '../scene/WorldCanvas';
import { Dock } from '../overlays/Dock';
import { DetailPanel } from '../overlays/DetailPanel';
import { PipView } from '../overlays/PipView';
import { StatusChip } from '../overlays/StatusChip';
import { DestinationToast, GoalToast, InterventionToast, ReplanToast, StopToast } from '../overlays/Toasts';
import { Wordmark } from '../overlays/Wordmark';
import { SceneSwitcher } from '../overlays/SceneSwitcher';
import { NetworkPanel } from '../overlays/NetworkPanel';
import { SafetyStack } from '../overlays/SafetyStack';
import { useRunState } from '../hooks/useRunState';
import { useMediaQuery } from '../hooks/useMediaQuery';
import type { CourseSetup } from '../hooks/useCourseSetup';
import { frameFor } from '../sim/frames';
import { useLiveRun } from '../hooks/useLiveRun';
import {
  toBackendMeshHeading,
  toSceneGround,
  toSceneMeshYaw,
  toScenePath,
} from '../scene/coords';

export function RunScreen({
  course,
  onReplay,
  onSetup,
  liveBackend = false,
}: {
  course: CourseSetup;
  onReplay: () => void;
  onSetup: () => void;
  liveBackend?: boolean;
}) {
  const run = useRunState();
  const [arrived, setArrived] = useState(false);
  const onSetupRef = useRef(onSetup);
  onSetupRef.current = onSetup;
  const { def } = run;
  const showCornerView = useMediaQuery('(min-width: 1180px)');

  const live = useLiveRun({ enabled: liveBackend });

  useEffect(() => {
    if (!live.connected) return;
    live.send({
      type: 'start_hall',
      seed: course.seed,
      goal: { x: 3, y: -86 },
      obstacles: course.obstacles.map((obstacle) => ({
        kind: obstacle.t,
        x: obstacle.x,
        y: -obstacle.z,
        // Forklift.tsx is +Z forward. Pallets/crates/barriers are long on +X,
        // which already matches the bicycle length axis at the stored scene yaw.
        yaw:
          obstacle.t === 'forklift'
            ? toBackendMeshHeading(obstacle.yaw ?? 0)
            : (obstacle.yaw ?? 0),
      })),
    });
  }, [live.connected, live.send, course.seed, course.obstacles, run.runId]);

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

  const measured = live.frame;
  const goalReached = measured?.controller.mode === 'goal' || measured?.metrics.goal_reached === true;
  useEffect(() => {
    if (!goalReached) return;
    setArrived(true);
    const timer = window.setTimeout(() => onSetupRef.current(), 2400);
    return () => window.clearTimeout(timer);
  }, [goalReached]);
  const driven = useMemo(() => {
    if (!liveBackend) return null;
    if (!measured) return { x: -1.5, z: 8, yaw: 0 };
    const ground = toSceneGround(measured.rover.x, measured.rover.y);
    return { x: ground.x, z: ground.z, yaw: toSceneMeshYaw(measured.rover.heading) };
  }, [liveBackend, measured]);
  const agents = useMemo(() => {
    if (!measured) return [];
    return measured.obstacles
      .filter((obstacle) => obstacle.kind === 'forklift' || obstacle.id.includes('forklift'))
      .map((obstacle) => {
        const ground = toSceneGround(obstacle.position[0], obstacle.position[1]);
        return {
          x: ground.x,
          z: ground.z,
          yaw: toSceneMeshYaw(obstacle.yaw ?? 0),
        };
      });
  }, [measured]);
  const liveRoute = useMemo(
    () => (measured && measured.selected_path.length > 1 ? toScenePath(measured.selected_path) : null),
    [measured],
  );
  const liveFallback = useMemo(
    () => (measured && measured.fallback_paths.length > 1 ? toScenePath(measured.fallback_paths) : null),
    [measured],
  );
  const liveCandidates = useMemo(
    () => (measured ? measured.candidate_paths.map((path) => toScenePath(path)) : []),
    [measured],
  );
  const liveTrail = useMemo(() => {
    if (!measured || measured.executed_trail.length < 2) return null;
    return toScenePath(measured.executed_trail.slice(-180));
  }, [measured]);
  const liveGoal = useMemo(
    () => (measured?.goal ? toSceneGround(measured.goal[0], measured.goal[1]) : null),
    [measured],
  );

  // Stacks stay as the setup meshes. Forklifts move from the bridge, so the
  // placed forklift props are dropped here and drawn from `agents` instead.
  const obstacles = useMemo(
    () =>
      liveBackend && measured
        ? course.obstacles.filter((obstacle) => obstacle.t !== 'forklift')
        : course.obstacles,
    [liveBackend, measured, course.obstacles],
  );
  const route = measured ? liveRoute : def.live ? course.plannedRoute : null;
  const candidates = measured
    ? liveCandidates
    : def.live
      ? course.plannedCandidates
      : [];
  // While Hybrid A* is the route, the neural samples are not what the rover
  // is following. Drawing them is what made the floor look like the network path.
  const onFallback = Boolean(measured && liveFallback && !liveRoute);
  const drawnCandidates = onFallback ? [] : candidates;

  const canned = frameFor({
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

  const shown = measured ?? canned;
  const chip = measured ? measured.controller.mode : run.chip;
  const ring = chip === 'estop' ? 'stop' : chip === 'intervention' ? 'warn' : 'ok';
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
          clock={liveBackend ? undefined : def.live ? run.clock : undefined}
          runId={run.runId}
          attention={!liveBackend && !!def.network && run.attentionOnMap}
          quality={run.quality}
          obstacles={obstacles}
          route={route}
          candidates={drawnCandidates}
          fallback={measured ? liveFallback : null}
          trail={measured ? liveTrail : null}
          goalPoint={measured ? liveGoal : null}
          driven={driven}
          agents={agents}
          ring={ring}
          onTick={liveBackend ? undefined : run.setProgress}
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

      <Wordmark onClick={onSetup} />
      <StatusChip kind={chip} placement={chipPlacement} />

      {showCornerView && (
      <PipView
        state={def.world}
        cameraMode={pipCamera}
        obstacles={obstacles}
        route={route}
        candidates={drawnCandidates}
        fallback={measured ? liveFallback : null}
        trail={measured ? liveTrail : null}
        goalPoint={measured ? liveGoal : null}
        driven={driven}
        agents={agents}
        ring={ring}
        label={run.swapped ? 'ROVER VIEW' : 'COURSE VIEW'}
        clock={liveBackend ? undefined : def.live ? run.clock : undefined}
        runId={run.runId}
        onSwap={run.toggleSwap}
      />
      )}

      {!measured && def.toast === 'int' && (
        <InterventionToast top={toastTop} controller={canned.controller} />
      )}
      {!measured && def.toast === 'stop' && (
        <StopToast top={toastTop} clearance={canned.metrics.min_clearance} />
      )}
      {!measured && def.toast === 'replan' && <ReplanToast top={toastTop} />}
      {!measured && def.toast === 'goal' && <GoalToast top={toastTop} />}
      {measured?.controller.mode === 'estop' && (
        <StopToast top={toastTop} clearance={measured.metrics.min_clearance} measured />
      )}
      {measured?.controller.mode === 'replanning' && !arrived && (
        <ReplanToast top={toastTop} latencyMs={measured.controller.latency_ms} measured />
      )}
      {arrived && <DestinationToast top={toastTop} />}
      {measured && !arrived && <SafetyStack controller={measured.controller} />}

      {def.panel && (
        <DetailPanel
          open={run.panelOpen}
          onToggle={run.togglePanel}
          controller={shown.controller}
        />
      )}

      {def.network && (
        <NetworkPanel
          curves={measured?.flow ?? []}
          conditioning={measured?.conditioning ?? null}
          source={measured?.controller.source ?? null}
          latencyMs={
            measured?.controller.mode === 'replanning' ? measured.controller.latency_ms : null
          }
          checkpoint={null}
          onClose={run.toggleNetwork}
        />
      )}

      <Dock
        variant={run.variant}
        playing={run.playing}
        onTogglePlay={togglePlay}
        onStep={step}
        safety={liveBackend ? true : run.safety}
        onToggleSafety={liveBackend ? () => undefined : run.toggleSafety}
        safetyLocked={liveBackend}
        cbfMode={chip}
        networkOpen={!!def.network}
        onToggleNetwork={run.toggleNetwork}
        onReplay={run.restart}
        onCompare={onReplay}
        showCompare={!liveBackend}
        camera={run.camera}
        onCamera={run.setCamera}
        quality={run.quality}
        onCycleQuality={run.cycleQuality}
        speed={measured ? measured.rover.speed.toFixed(2) : run.speed}
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
