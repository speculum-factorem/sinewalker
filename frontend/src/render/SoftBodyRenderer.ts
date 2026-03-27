import * as THREE from 'three'

export class SoftBodyRenderer {
  private readonly softBody: any
  private readonly geometry: THREE.BufferGeometry
  private readonly positions: Float32Array
  private readonly nodeCount: number

  constructor(softBody: any, geometry: THREE.BufferGeometry, nodeCount?: number) {
    this.softBody = softBody
    this.geometry = geometry

    const positionAttr = this.geometry.getAttribute('position')
    if (!positionAttr) throw new Error('Missing geometry position attribute')
    this.positions = positionAttr.array as Float32Array
    const verts = this.positions.length / 3
    const simNodes = softBody.get_m_nodes()?.size?.() ?? verts
    this.nodeCount = Math.min(nodeCount ?? verts, verts, simNodes)
  }

  syncVertices() {
    const nodes = this.softBody.get_m_nodes()
    let ptr = 0
    for (let i = 0; i < this.nodeCount; i++) {
      const node = nodes.at(i)
      const nodePos = node.get_m_x()
      this.positions[ptr++] = nodePos.x()
      this.positions[ptr++] = nodePos.y()
      this.positions[ptr++] = nodePos.z()
    }
    this.geometry.attributes.position.needsUpdate = true
  }
}

