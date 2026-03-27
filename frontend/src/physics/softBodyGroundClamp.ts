/**
 * Bullet/Ammo нередко пропускает узлы soft body сквозь статический пол.
 * — Поднимаем узлы чуть выше плоскости (liftMargin), чтобы уменьшить выталкивание.
 * — У узлов, уже «стоящих» около пола, слегка гасим касательную скорость (опора скользит меньше).
 */
export function clampSoftBodyNodesAbovePlane(
  softBody: any,
  planeY: number,
  slip = 0.45,
  liftMargin = 0.0035
) {
  const nodes = softBody.get_m_nodes()
  const n = nodes.size()
  const vyKill = 1 - Math.min(0.95, Math.max(0, slip))
  const ySurf = planeY + liftMargin
  /** Выше пола, но ещё в зоне контакта — доп. трение по xz. */
  const nearBand = 0.055
  const nearFriction = 0.9

  for (let i = 0; i < n; i++) {
    const node = nodes.at(i)
    const p = node.get_m_x()
    const py = p.y()

    if (py < ySurf) {
      p.setY(ySurf)
      node.set_m_x(p)

      const v = node.get_m_v()
      if (v.y() < 0) v.setY(0)
      v.setX(v.x() * vyKill)
      v.setZ(v.z() * vyKill)
      node.set_m_v(v)
    } else if (py < ySurf + nearBand) {
      const v = node.get_m_v()
      v.setX(v.x() * nearFriction)
      v.setZ(v.z() * nearFriction)
      if (v.y() < -0.15) v.setY(v.y() * 0.35)
      node.set_m_v(v)
    }
  }
}
