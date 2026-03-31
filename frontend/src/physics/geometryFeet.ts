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

/** Точки «ножек» в локальных координатах геометрии (любое количество). */
export function computeFootPivotsLocal(geometry: THREE.BufferGeometry, count = 4): THREE.Vector3[] {
  geometry.computeBoundingBox()
  const pos = geometry.getAttribute('position')
  if (!pos) return []
  const desired = Math.max(1, Math.floor(count))

  const n = pos.count
  const ys: number[] = []
  for (let i = 0; i < n; i += 2) ys.push(pos.getY(i))
  ys.sort((a, b) => a - b)
  const yCut = ys[Math.floor(ys.length * 0.15)] ?? 0.12

  const candidates: Array<{ p: THREE.Vector3; angle: number; score: number }> = []

  for (let i = 0; i < n; i++) {
    const y = pos.getY(i)
    if (y > yCut) continue
    const x = pos.getX(i)
    const z = pos.getZ(i)
    const radial = Math.hypot(x, z)
    if (radial < 1e-5) continue
    const score = radial - y * 0.2
    const angle = Math.atan2(z, x)
    candidates.push({ p: new THREE.Vector3(x, y, z), angle, score })
  }

  if (candidates.length === 0) return []
  const sectorSize = (Math.PI * 2) / desired
  const bestBySector: Array<{ p: THREE.Vector3; score: number } | null> = new Array(desired).fill(null)
  for (const c of candidates) {
    const normalized = c.angle < 0 ? c.angle + Math.PI * 2 : c.angle
    let sector = Math.floor(normalized / sectorSize)
    if (sector >= desired) sector = desired - 1
    const prev = bestBySector[sector]
    if (!prev || c.score > prev.score) bestBySector[sector] = { p: c.p, score: c.score }
  }

  const out: THREE.Vector3[] = []
  for (const b of bestBySector) {
    if (b) out.push(b.p.clone())
  }
  if (out.length >= desired) return out.slice(0, desired)

  // Fill missing sectors with next best unique points.
  candidates.sort((a, b) => b.score - a.score)
  for (const c of candidates) {
    if (out.length >= desired) break
    const tooClose = out.some((p) => p.distanceToSquared(c.p) < 1e-4)
    if (!tooClose) out.push(c.p.clone())
  }
  return out
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
