import * as THREE from 'three';
import type { Pt } from './math';

export function ribbonGeometry(pts: Pt[], width: number, y = 0.03): THREE.BufferGeometry {
  const geo = new THREE.BufferGeometry();
  const n = pts.length;
  if (n < 2) return geo;

  const positions = new Float32Array(n * 2 * 3);
  const uvs = new Float32Array(n * 2 * 2);

  for (let i = 0; i < n; i++) {
    const a = pts[Math.max(0, i - 1)];
    const b = pts[Math.min(n - 1, i + 1)];
    let tx = b.x - a.x;
    let tz = b.z - a.z;
    const len = Math.hypot(tx, tz) || 1;
    tx /= len;
    tz /= len;
    const nx = -tz;
    const nz = tx;
    const h = width / 2;
    const o = i * 6;
    positions[o] = pts[i].x + nx * h;
    positions[o + 1] = y;
    positions[o + 2] = pts[i].z + nz * h;
    positions[o + 3] = pts[i].x - nx * h;
    positions[o + 4] = y;
    positions[o + 5] = pts[i].z - nz * h;
    const v = i / (n - 1);
    uvs[i * 4] = 0;
    uvs[i * 4 + 1] = v;
    uvs[i * 4 + 2] = 1;
    uvs[i * 4 + 3] = v;
  }

  const index: number[] = [];
  for (let i = 0; i < n - 1; i++) {
    const a = i * 2;
    index.push(a, a + 1, a + 2, a + 1, a + 3, a + 2);
  }

  geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geo.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
  geo.setIndex(index);
  geo.computeVertexNormals();
  return geo;
}

export function dashSegments(pts: Pt[], on: number, off: number): Pt[][] {
  const segs: Pt[][] = [];
  let i = 0;
  while (i < pts.length - 1) {
    segs.push(pts.slice(i, Math.min(pts.length, i + on + 1)));
    i += on + off;
  }
  return segs.filter((s) => s.length > 1);
}

export function resample(pts: Pt[], step: number): Pt[] {
  if (pts.length < 2) return pts;
  const out: Pt[] = [pts[0]];
  let carry = 0;
  for (let i = 1; i < pts.length; i++) {
    const a = pts[i - 1];
    const b = pts[i];
    const d = Math.hypot(b.x - a.x, b.z - a.z);
    let t = step - carry;
    while (t <= d) {
      const u = t / d;
      out.push({ x: a.x + (b.x - a.x) * u, z: a.z + (b.z - a.z) * u });
      t += step;
    }
    carry = (carry + d) % step;
  }
  out.push(pts[pts.length - 1]);
  return out;
}
