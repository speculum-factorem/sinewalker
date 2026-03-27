import * as THREE from 'three'

/** Плоский массив вершин геометрии в локальных координатах (для convex hull). */
export function geometryPositionsFlat(geometry: THREE.BufferGeometry): number[] {
  const pos = geometry.getAttribute('position')
  if (!pos) return []
  const out: number[] = []
  for (let i = 0; i < pos.count; i++) {
    out.push(pos.getX(i), pos.getY(i), pos.getZ(i))
  }
  return out
}

/** 4 точки «ножек» в локальных координатах геометрии (нормализованный стол). */
export function computeFootPivotsLocal(geometry: THREE.BufferGeometry, count = 4): THREE.Vector3[] {
  geometry.computeBoundingBox()
  const pos = geometry.getAttribute('position')
  if (!pos) return []

  const n = pos.count
  const ys: number[] = []
  for (let i = 0; i < n; i += 2) ys.push(pos.getY(i))
  ys.sort((a, b) => a - b)
  const yCut = ys[Math.floor(ys.length * 0.15)] ?? 0.12

  const best: Array<{ x: number; y: number; z: number; score: number } | null> = [null, null, null, null]

  for (let i = 0; i < n; i++) {
    const y = pos.getY(i)
    if (y > yCut) continue
    const x = pos.getX(i)
    const z = pos.getZ(i)
    const q = (x >= 0 ? 0 : 1) + (z >= 0 ? 0 : 2)
    const score = Math.abs(x) + Math.abs(z) - y * 0.2
    const cur = best[q]
    if (!cur || score > cur.score) best[q] = { x, y, z, score }
  }

  const out: THREE.Vector3[] = []
  for (const b of best) {
    if (b) out.push(new THREE.Vector3(b.x, b.y, b.z))
  }
  return out.slice(0, count)
}

/** Ближайшие узлы soft body к заданным мировым точкам (разные индексы, где возможно). */
export function nearestNodeIndicesForWorldPoints(softBody: any, targets: THREE.Vector3[]): number[] {
  const nodes = softBody.get_m_nodes()
  const n = nodes.size()
  const used = new Set<number>()
  const out: number[] = []

  const distSq = (i: number, t: THREE.Vector3) => {
    const p = nodes.at(i).get_m_x()
    const dx = p.x() - t.x
    const dy = p.y() - t.y
    const dz = p.z() - t.z
    return dx * dx + dy * dy + dz * dz
  }

  for (const target of targets) {
    let best = -1
    let bestD = Infinity
    for (let i = 0; i < n; i++) {
      if (used.has(i)) continue
      const d = distSq(i, target)
      if (d < bestD) {
        bestD = d
        best = i
      }
    }
    if (best < 0) {
      for (let i = 0; i < n; i++) {
        const d = distSq(i, target)
        if (d < bestD) {
          bestD = d
          best = i
        }
      }
    }
    if (best >= 0) {
      used.add(best)
      out.push(best)
    }
  }
  return out
}
