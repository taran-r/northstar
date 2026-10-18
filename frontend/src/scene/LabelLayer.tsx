import { useRef } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';

export type WorldLabelSpec = {
  id: string;
  pos: [number, number, number];
  text: string;
  color: string;
  dashed?: boolean;
};

export function LabelTracker({
  labels,
  nodes,
}: {
  labels: WorldLabelSpec[];
  nodes: Record<string, HTMLDivElement | null>;
}) {
  const { camera, size } = useThree();
  const v = useRef(new THREE.Vector3());

  useFrame(() => {
    for (const label of labels) {
      const el = nodes[label.id];
      if (!el) continue;
      v.current.set(label.pos[0], label.pos[1], label.pos[2]).project(camera);
      const behind = v.current.z > 1;
      if (behind) {
        el.style.opacity = '0';
        continue;
      }
      const x = (v.current.x * 0.5 + 0.5) * size.width;
      const y = (-v.current.y * 0.5 + 0.5) * size.height;
      el.style.opacity = '1';
      el.style.transform = `translate(${x}px, ${y}px) translate(-50%, -100%)`;
    }
  });

  return null;
}

export function LabelOverlay({
  labels,
  nodes,
}: {
  labels: WorldLabelSpec[];
  nodes: Record<string, HTMLDivElement | null>;
}) {
  return (
    <div className="pointer-events-none absolute inset-0 overflow-hidden">
      {labels.map((label) => (
        <div
          key={label.id}
          ref={(el) => {
            nodes[label.id] = el;
          }}
          className="absolute left-0 top-0 flex h-[22px] items-center gap-2 whitespace-nowrap rounded-[11px] px-2.5 text-[10.5px] font-medium uppercase leading-[22px] tracking-[0.14em] text-bone"
          style={{ background: 'rgba(20,23,28,0.72)', opacity: 0 }}
        >
          <span
            className="block h-0 w-3.5"
            style={{ borderTop: `2px ${label.dashed ? 'dashed' : 'solid'} ${label.color}` }}
          />
          <span>{label.text}</span>
        </div>
      ))}
    </div>
  );
}
