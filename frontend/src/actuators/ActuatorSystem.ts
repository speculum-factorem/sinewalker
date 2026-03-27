import * as THREE from 'three'
import { AmmoWorld } from '../physics/AmmoWorld'

export interface Genome {
  amplitudes: number[]
  omegas: number[]
  phases: number[]
}

export interface Actuator {
  nodeIndex: number
  basePos: THREE.Vector3
  localOffset: THREE.Vector3
  dir: THREE.Vector3
  influence: number
  radius: number
  mesh: THREE.Mesh
  body: any
}

export interface ActuatorSystemOptions {
  maxInfluence?: number
  visualize?: boolean
}

export class ActuatorSystem {
  private readonly world: AmmoWorld
  private readonly softBody: any
  private readonly visualize: boolean
  /** y плоскости пола; центр сферы не ниже (y + радиус + зазор). `null` — не ограничивать. */
  private readonly actuatorGroundPlane: number | null

  private readonly actuators: Actuator[] = []
  private readonly tmpTransform: any
  private readonly tmpOrigin: any
  private readonly tmpQuat: any

  constructor(world: AmmoWorld, softBody: any, visualize = true, actuatorGroundPlane: number | null = 0) {
    this.world = world
    this.softBody = softBody
    this.visualize = visualize
    this.actuatorGroundPlane = actuatorGroundPlane
    this.tmpTransform = new world.ammo.btTransform()
    this.tmpOrigin = new world.ammo.btVector3(0, 0, 0)
    this.tmpQuat = new world.ammo.btQuaternion(0, 0, 0, 1)
  }

  private computeCOM(sampleStride = 6): THREE.Vector3 {
    const nodes = this.softBody.get_m_nodes()
    const nodeCount = nodes.size ? nodes.size() : 0
    const stride = Math.max(1, Math.floor(sampleStride))
    const com = new THREE.Vector3()
    let n = 0

    // Fallback if bindings don't expose size(): use actuator nodes only.
    if (!nodeCount || !Number.isFinite(nodeCount)) {
      for (const a of this.actuators) {
        const node = nodes.at(a.nodeIndex)
        const p = node.get_m_x()
        com.add(new THREE.Vector3(p.x(), p.y(), p.z()))
        n++
      }
      return n ? com.multiplyScalar(1 / n) : com
    }

    for (let i = 0; i < nodeCount; i += stride) {
      const node = nodes.at(i)
      const p = node.get_m_x()
      com.add(new THREE.Vector3(p.x(), p.y(), p.z()))
      n++
    }
    return n ? com.multiplyScalar(1 / n) : com
  }

  get count() {
    return this.actuators.length
  }

  get nodeIndices() {
    return this.actuators.map((a) => a.nodeIndex)
  }

  createForNodes(
    nodeIndices: number[],
    opts: { scene?: THREE.Scene; influence?: number; radius?: number; dirMode?: 'radial' | 'normal' } = {}
  ) {
    const {
      scene,
      influence = 0.52,
      radius = 0.035,
      dirMode = 'radial'
    } = opts

    // Compute a center in world coords based on selected nodes.
    // This keeps radial directions consistent without needing full node iteration.
    const center = new THREE.Vector3()
    const nodes = this.softBody.get_m_nodes()
    for (let i = 0; i < nodeIndices.length; i++) {
      const node = nodes.at(nodeIndices[i])
      const p = node.get_m_x()
      center.add(new THREE.Vector3(p.x(), p.y(), p.z()))
    }
    if (nodeIndices.length > 0) center.multiplyScalar(1 / nodeIndices.length)

    // If normal mode is selected, use geometry normals as direction.
    // For MVP we approximate normals via current node positions; users can switch later.
    const sceneToUse = scene ?? null
    const com0 = this.computeCOM(6)
    for (let idx = 0; idx < nodeIndices.length; idx++) {
      const nodeIndex = nodeIndices[idx]
      const node = nodes.at(nodeIndex)
      const p = node.get_m_x()
      const basePos = new THREE.Vector3(p.x(), p.y(), p.z())
      const localOffset = basePos.clone().sub(com0)

      const dir =
        dirMode === 'radial'
          ? basePos.clone().sub(center).normalize()
          : new THREE.Vector3(0, 1, 0)

      const body = this.world.createKinematicBodySphere(radius, basePos).body
      this.softBody.appendAnchor(nodeIndex, body, false, influence)

      const mesh = this.visualize && sceneToUse
        ? new THREE.Mesh(
            new THREE.SphereGeometry(radius * 1.25, 10, 10),
            new THREE.MeshStandardMaterial({ color: 0xffcc66, emissive: 0x2a1b00 })
          )
        : new THREE.Mesh(
            new THREE.SphereGeometry(radius * 1.25, 10, 10),
            new THREE.MeshStandardMaterial({ color: 0xffcc66, emissive: 0x2a1b00, visible: false })
          )

      mesh.position.copy(basePos)
      if (this.visualize && sceneToUse) sceneToUse.add(mesh)

      this.actuators.push({
        nodeIndex,
        basePos,
        localOffset,
        dir,
        influence,
        radius,
        mesh,
        body
      })
    }
  }

  setGenome(genome: { amplitudes: number[]; omegas: number[]; phases: number[] }) {
    if (genome.amplitudes.length !== this.actuators.length) {
      throw new Error('Genome length does not match actuator count')
    }
    this.currentGenome = genome
  }

  private currentGenome: Genome | null = null

  // Move all kinematic bodies based on a genome at given simulation time.
  applyAtTime(tSeconds: number) {
    if (!this.currentGenome) return

    const { amplitudes, omegas, phases } = this.currentGenome
    const com = this.computeCOM(6)

    for (let i = 0; i < this.actuators.length; i++) {
      const a = this.actuators[i]
      const amp = amplitudes[i]
      const omega = omegas[i]
      const phase = phases[i]

      const offset = a.dir.clone().multiplyScalar(amp * Math.sin(omega * tSeconds + phase))
      // Critical: actuators move relative to COM, not world-fixed.
      const newPos = com.clone().add(a.localOffset).add(offset)

      const gp = this.actuatorGroundPlane
      if (gp !== null && Number.isFinite(gp)) {
        const minCenterY = gp + a.radius + 0.004
        if (newPos.y < minCenterY) newPos.y = minCenterY
      }

      this.tmpTransform.setIdentity()
      this.tmpOrigin.setValue(newPos.x, newPos.y, newPos.z)
      this.tmpTransform.setOrigin(this.tmpOrigin)
      this.tmpTransform.setRotation(this.tmpQuat)

      a.body.setWorldTransform(this.tmpTransform)
      const ms = a.body.getMotionState?.()
      ms?.setWorldTransform(this.tmpTransform)

      if (this.visualize) a.mesh.position.copy(newPos)
    }
  }

  dispose() {
    // For MVP: remove meshes only. Physics bodies are owned by AmmoWorld and will be freed on world rebuild.
    for (const a of this.actuators) {
      if (this.visualize) a.mesh.parent?.remove(a.mesh)
    }
    this.actuators.length = 0
  }
}

