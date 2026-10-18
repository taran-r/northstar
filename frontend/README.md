# Frontend

React + TypeScript + React Three Fiber + Tailwind. Visual only: nothing talks to a backend yet,
but the screens already consume the state-frame shape the simulator will send.

```bash
npm install
npm run dev        # http://localhost:5173
npm run build
npm run typecheck
npm run test
```

## Layout

| Path | Contents |
|------|----------|
| `src/scene/` | The three.js world: hall, rover, forklift, obstacles, path ribbons, floor markers, camera rig |
| `src/overlays/` | HUD pieces: status chip, dock, corner view, toasts, detail panel, policy-network panel |
| `src/screens/` | `SetupScreen`, `RunScreen`, `ReplayScreen` |
| `src/hooks/` | `useRunState` (HUD state), `useCourseSetup` (obstacles and route preview, shared by both screens) |
| `src/types/sim.ts` | The state-frame and intent schema, matching Handoff.md section 10 |
| `src/sim/` | `frames.ts` builds a frame; `mockSource.ts` is the seam a real client replaces |
| `src/data/` | `replayMetrics.ts` — the replay table's values |
| `src/index.css` | Design tokens and the shared `glass` / control classes |

## Coordinates

`src/scene/coords.ts` is the single conversion between the backend frame (x, y, z with **z up**,
heading in the XY plane) and the Three.js frame (**y up**): `(x, y, z) → (x, z, -y)`, with matching
helpers for heading, ground points and paths. `src/scene/coords.test.ts` covers it. Convert at the
boundary only — scene components work in Three.js coordinates and never re-derive the mapping.

## Transport

`useSimClock` owns the run clock: `t` in seconds plus a `playing` flag, held in a ref the canvas
reads each frame. Pause stops the clock, so the rover, its trail and the crossing forklift all
freeze where they are; step advances `t` by 0.2 s while paused; the dock's circular-arrow button
restarts the run at `t = 0`. The HUD gets a throttled publish of the clock (150 ms while playing,
immediately after a step) and never drives the animation itself.

The replay comparison screen is reached from the review switcher, since the dock's arrow now means
restart.

## Where the data comes in

`RunScreen` builds a `StateFrame` with `frameFor(...)` and the overlays render from it, so the
switch to a live simulator is: subscribe to the socket, publish frames through `SimSource`, drop
`frameFor`. The frame carries `measured: false` — nothing in the UI presents these numbers as
recorded results.

The course you build in `SetupScreen` is shared state: its obstacles appear in the run, and the
rover drives the planned route rather than a canned loop.

Every forklift placed in setup is a moving agent during a run: it patrols the lane it faces
(across the aisle when turned sideways, up and down the hall otherwise), about 5.5 m either side of
where it was placed, each with its own speed and phase so they do not march in step. They hold at
the edge of their lane while the rover is within ~6 m, so nothing drives through the rover while
the backend is not there to make it brake. A course with no forklift gets one staged crossing agent
so the aisle is not empty. Route planning still uses their placed positions, so the preview's
clearance number is a snapshot, not a guarantee.

## Performance notes

The run screen holds two WebGL canvases, so the scene is built to keep draw calls and React work
low:

- the static hall is merged into one geometry per material (≈1500 boxes → ~15 draws);
- lighting is four directional/ambient sources, not per-fixture point lights;
- `Hall`, `Rover`, `Forklift`, `Obstacles`, `PathRibbons` and `WorldCanvas` are memoised;
- the drive loop animates inside the canvas against refs, so it never re-renders React; the HUD
  gets a throttled tick (150 ms) instead;
- the corner view renders at half resolution and 12 fps.

Quality (High / Balanced / Fast) changes pixel ratio, shadows and rack detail.

## Placeholders waiting on real data

- `src/data/replayMetrics.ts` holds the replay comparison table. `REPLAY_METRICS` is all em dashes
  because measured runs do not exist yet; `DESIGN_SAMPLE_METRICS` carries the design's sample
  values for a visual check. Swap which one `ReplayScreen` imports once runs are logged.
- `src/sim/frames.ts` carries the per-scene controller values. They are illustrative, flagged by
  `measured: false`, and go away when the simulator sends real ones.
- `src/overlays/networkGraph.ts` draws the policy-network panel from a fixed seed. Action
  probabilities, layer sizes and the "why" line are illustrative until inference is connected.
- `src/overlays/SceneSwitcher.tsx` is a review aid, not part of the design, and is hidden unless
  you load the page with `?states=1`. It steps through the design's canned HUD states; once the
  simulator is connected those come from `controller.mode` on each frame instead.
- `src/branding.ts` holds the wordmark. `CLAUDE.md` says the project name is undecided; set
  `SHOW_WORDMARK` to `false` or change `APP_NAME` in one place.
- The canned scenes (intervention, emergency stop, replanning) stage their own forklift, which can
  overlap obstacles you placed in setup. They are demo states; they disappear once the simulator
  drives the run.
