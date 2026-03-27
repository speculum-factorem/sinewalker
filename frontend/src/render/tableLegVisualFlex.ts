import * as THREE from 'three'
import type { Genome } from '../actuators/ActuatorSystem'

/** Копия позиций вершин до деформации (локальные координаты геометрии). */
export function snapshotGeometryPositions(geometry: THREE.BufferGeometry): Float32Array {
  const pos = geometry.getAttribute('position')
  return new Float32Array(pos.array as Float32Array)
}

/** Верхняя граница по Y для «ножек» в локальном bbox (ниже — можно гнуть визуально). */
export function legRegionYMax(geometry: THREE.BufferGeometry, spanFraction = 0.38): number {
  geometry.computeBoundingBox()
  const b = geometry.boundingBox
  if (!b) return 0.2
  return b.min.y + (b.max.y - b.min.y) * spanFraction
}

/** Единичные направления «в сторону» от центра опор в плоскости xz (локально). */
export function footOutwardDirsXZ(feet: THREE.Vector3[]): THREE.Vector3[] {
  const c = new THREE.Vector3()
  for (const f of feet) c.add(f)
  if (feet.length > 0) c.multiplyScalar(1 / feet.length)
  return feet.map((f) => {
    const d = new THREE.Vector3(f.x - c.x, 0, f.z - c.z)
    if (d.lengthSq() < 1e-10) d.set(1, 0, 0)
    else d.normalize()
    return d
  })
}

/**
 * Лёгкий чисто визуальный изгиб ножек (после sync rigid body).
 * Не влияет на Bullet — только на отображаемые вершины.
 */
export function applyLegVertexFlex(
  geometry: THREE.BufferGeometry,
  base: Float32Array,
  feet: THREE.Vector3[],
  outward: THREE.Vector3[],
  genome: Genome | null,
  tSeconds: number,
  legYMax: number,
  bboxMinY: number,
  visualGain = 0.062
) {
  if (!genome || feet.length === 0 || outward.length !== feet.length) return
  const posAttr = geometry.getAttribute('position')
  const arr = posAttr.array as Float32Array
  const n = posAttr.count
  const { amplitudes, omegas, phases } = genome
  const denom = Math.max(1e-4, legYMax - bboxMinY)

  for (let vi = 0; vi < n; vi++) {
    const i3 = vi * 3
    const bx = base[i3]
    const by = base[i3 + 1]
    const bz = base[i3 + 2]

    if (by > legYMax) {
      arr[i3] = bx
      arr[i3 + 1] = by
      arr[i3 + 2] = bz
      continue
    }

    const falloff = Math.pow(Math.max(0, Math.min(1, (legYMax - by) / denom)), 1.35)

    let ox = 0
    let oy = 0
    let oz = 0
    let wsum = 0
    const m = Math.min(feet.length, amplitudes.length, outward.length)
    for (let fi = 0; fi < m; fi++) {
      const fp = feet[fi]
      const dx = bx - fp.x
      const dy = by - fp.y
      const dz = bz - fp.z
      const w = 1 / (dx * dx + dy * dy + dz * dz + 2e-4)
      const s = amplitudes[fi] * Math.sin(omegas[fi] * tSeconds + phases[fi])
      const dir = outward[fi]
      ox += w * dir.x * s
      oy += w * dir.y * s
      oz += w * dir.z * s
      wsum += w
    }
    if (wsum > 1e-8) {
      const g = (visualGain * falloff) / wsum
      arr[i3] = bx + ox * g
      arr[i3 + 1] = by + oy * g
      arr[i3 + 2] = bz + oz * g
    } else {
      arr[i3] = bx
      arr[i3 + 1] = by
      arr[i3 + 2] = bz
    }
  }
  posAttr.needsUpdate = true
  geometry.computeVertexNormals()
}
