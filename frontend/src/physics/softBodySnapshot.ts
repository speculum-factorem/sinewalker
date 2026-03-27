/** Снимок узлов soft body для сброса между эпизодами (один мир Ammo). */

export function captureSoftBodyNodeState(softBody: any): { pos: Float32Array; vel: Float32Array } {
  const nodes = softBody.get_m_nodes()
  const n = nodes.size()
  const pos = new Float32Array(n * 3)
  const vel = new Float32Array(n * 3)
  for (let i = 0; i < n; i++) {
    const node = nodes.at(i)
    const p = node.get_m_x()
    const v = node.get_m_v()
    const ix = i * 3
    pos[ix] = p.x()
    pos[ix + 1] = p.y()
    pos[ix + 2] = p.z()
    vel[ix] = v.x()
    vel[ix + 1] = v.y()
    vel[ix + 2] = v.z()
  }
  return { pos, vel }
}

export function restoreSoftBodyNodeState(softBody: any, pos: Float32Array, vel: Float32Array) {
  const nodes = softBody.get_m_nodes()
  const n = nodes.size()
  for (let i = 0; i < n; i++) {
    const ix = i * 3
    const node = nodes.at(i)
    const p = node.get_m_x()
    const v = node.get_m_v()
    p.setValue(pos[ix], pos[ix + 1], pos[ix + 2])
    v.setValue(vel[ix], vel[ix + 1], vel[ix + 2])
    node.set_m_x(p)
    node.set_m_v(v)
  }
}
