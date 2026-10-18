import { memo, useEffect, useMemo } from 'react';
import * as THREE from 'three';
import { curve, spline, type Pt } from './math';
import { PATH_STYLE } from './palette';
import type { PathSpec } from './scenes';
import { dashSegments, resample, ribbonGeometry } from './ribbon';

export function pathPoints(spec: PathSpec, start: Pt, goal: Pt | null): Pt[] {
  if (spec.pts) return spec.pts;
  if (spec.wps) return spline(spec.wps.map((w) => ({ x: w[0], z: w[1] })), 10);
  if (!goal) return [];
  return curve(start, goal, spec.off ?? 0, spec.upto);
}

function Ribbon({
  pts,
  width,
  color,
  opacity,
  y = 0.03,
  additive = false,
}: {
  pts: Pt[];
  width: number;
  color: string;
  opacity: number;
  y?: number;
  additive?: boolean;
}) {
  const geo = useMemo(() => ribbonGeometry(pts, width, y), [pts, width, y]);
  useEffect(() => () => geo.dispose(), [geo]);
  if (pts.length < 2) return null;
  return (
    <mesh geometry={geo}>
      <meshBasicMaterial
        color={color}
        transparent
        opacity={opacity}
        depthWrite={false}
        side={THREE.DoubleSide}
        blending={additive ? THREE.AdditiveBlending : THREE.NormalBlending}
      />
    </mesh>
  );
}

export const PathRibbons = memo(function PathRibbons({
  paths,
  start,
  goal,
}: {
  paths: PathSpec[];
  start: Pt;
  goal: Pt | null;
}) {
  return (
    <group>
      {paths.map((spec, i) => {
        const style = PATH_STYLE[spec.k];
        const pts = pathPoints(spec, start, goal);
        if (pts.length < 2) return null;
        const opacity = spec.a ?? style.opacity;

        if (spec.k === 'fb') {
          const even = resample(pts, 0.4);
          return (
            <group key={i}>
              {dashSegments(even, 3, 3).map((seg, j) => (
                <Ribbon key={j} pts={seg} width={style.width} color={style.color} opacity={opacity} />
              ))}
            </group>
          );
        }

        const glow = spec.k === 'sel' || spec.k === 'cand';
        return (
          <group key={i}>
            <Ribbon pts={pts} width={style.width} color={style.color} opacity={opacity} />
            {glow && (
              <Ribbon
                pts={pts}
                width={style.width * (spec.k === 'sel' ? 3.2 : 4.5)}
                color={spec.k === 'sel' ? '#F2A33A' : '#5B8DEF'}
                opacity={spec.k === 'sel' ? 0.18 : 0.08}
                y={0.028}
                additive
              />
            )}
          </group>
        );
      })}
    </group>
  );
});

export const WheelTrail = memo(function WheelTrail({ pts, scale = 1.5 }: { pts: Pt[]; scale?: number }) {
  if (pts.length < 2) return null;
  const chunks = 4;
  const per = Math.ceil(pts.length / chunks);
  return (
    <group>
      {[-0.38, 0.38].map((off) => {
        const track = pts.map((q, i) => {
          const a = pts[Math.max(0, i - 1)];
          const b = pts[Math.min(pts.length - 1, i + 1)];
          let tx = b.x - a.x;
          let tz = b.z - a.z;
          const l = Math.hypot(tx, tz) || 1;
          tx /= l;
          tz /= l;
          return { x: q.x - (tz * off * scale) / 1.5, z: q.z + (tx * off * scale) / 1.5 };
        });
        return (
          <group key={off}>
            {Array.from({ length: chunks }, (_, i) => {
              const seg = track.slice(Math.max(0, i * per - 1), Math.min(track.length, (i + 1) * per + 1));
              if (seg.length < 2) return null;
              return (
                <Ribbon
                  key={i}
                  pts={seg}
                  width={0.12}
                  color="#FFE2B0"
                  opacity={0.1 + (0.4 * (i + 1)) / chunks}
                  y={0.02}
                />
              );
            })}
          </group>
        );
      })}
    </group>
  );
});
