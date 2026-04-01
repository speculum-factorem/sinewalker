import * as THREE from 'three'
import { AmmoWorld } from '../physics/AmmoWorld'
import { comWorldFromRigidBody } from '../physics/rigidBodyThree'
import type { Genome } from './ActuatorSystem'
import type { MotorCommand, RobotState } from '../robotio/types'

interface FootActuator {
  pivotBodyLocal: THREE.Vector3
  dir: THREE.Vector3
  phaseBias: number
  radius: number
  mesh: THREE.Mesh
}

/**
 * Приводы без kinematic-тел и point2point: пружинная сила в точке ноги (applyForce).
 * Раньше констрейнты + двигаемые кинематики разгоняли стола в Bullet.
 */
export class ActuatorSystemRigid {
  private readonly visualize: boolean
  private readonly actuatorGroundPlane: number | null
  private springGain: number
  private maxForcePerFoot: number
  private tableBody: any = null
  private readonly forceBt: any
  private readonly relBt: any
  private readonly actuators: FootActuator[] = []
  private currentGenome: Genome | null = null

  constructor(
    world: AmmoWorld,
    visualize = true,
    actuatorGroundPlane: number | null = 0,
    springGain = 42,
    maxForcePerFoot = 48
  ) {
    this.visualize = visualize
    this.actuatorGroundPlane = actuatorGroundPlane
    this.springGain = springGain
    this.maxForcePerFoot = maxForcePerFoot
    const { ammo } = world
    this.forceBt = new ammo.btVector3(0, 0, 0)
    this.relBt = new ammo.btVector3(0, 0, 0)
  }

  private bodyLocalToWorld(body: any, local: THREE.Vector3): THREE.Vector3 {
    const t = body.getCenterOfMassTransform()
    const o = t.getOrigin()
    const r = t.getRotation()
    const q = new THREE.Quaternion(r.x(), r.y(), r.z(), r.w())
    return local.clone().applyQuaternion(q).add(new THREE.Vector3(o.x(), o.y(), o.z()))
  }

  createForFeet(
    tableBody: any,
    footLocals: THREE.Vector3[],
    opts: { scene?: THREE.Scene; radius?: number; tableSpawnMatrix?: THREE.Matrix4 } = {}
  ) {
    this.disposeInternal()
    this.tableBody = tableBody
    this.actuators.length = 0

    const { scene, radius = 0.034, tableSpawnMatrix } = opts
    const spawn = tableSpawnMatrix ?? new THREE.Matrix4().identity()

    const com0 = comWorldFromRigidBody(tableBody)
    const t0 = tableBody.getCenterOfMassTransform()
    const ro = t0.getRotation()
    const qBodyInv = new THREE.Quaternion(ro.x(), ro.y(), ro.z(), ro.w()).invert()

    const footWorldPts: THREE.Vector3[] = []
    for (const fl of footLocals) {
      footWorldPts.push(fl.clone().applyMatrix4(spawn))
    }

    const center = new THREE.Vector3()
    for (const p of footWorldPts) center.add(p)
    if (footWorldPts.length > 0) center.multiplyScalar(1 / footWorldPts.length)

    for (let idx = 0; idx < footLocals.length; idx++) {
      const footWorld = footWorldPts[idx]
      const pivotBodyLocal = footWorld.clone().sub(com0).applyQuaternion(qBodyInv)
      const basePos = footWorld.clone()
      // Locomotion-oriented direction:
      // mostly along forward (+x), with small lateral component by side.
      const sideSign = footWorld.z >= center.z ? 1 : -1
      const dir = new THREE.Vector3(1.0, 0, 0.28 * sideSign).normalize()
      // Diagonal alternating gait bias for a steadier step cycle.
      const phaseBias = footWorld.z * footWorld.x >= 0 ? 0 : Math.PI

      const mesh = this.visualize && scene
        ? new THREE.Mesh(
            new THREE.SphereGeometry(radius * 1.25, 10, 10),
            new THREE.MeshStandardMaterial({ color: 0xffcc66, emissive: 0x2a1b00 })
          )
        : new THREE.Mesh(
            new THREE.SphereGeometry(radius * 1.25, 10, 10),
            new THREE.MeshStandardMaterial({ color: 0xffcc66, emissive: 0x2a1b00, visible: false })
          )
      mesh.position.copy(basePos)
      if (this.visualize && scene) scene.add(mesh)

      this.actuators.push({
        pivotBodyLocal,
        dir,
        phaseBias,
        radius,
        mesh
      })
    }
  }

  setGenome(genome: Genome) {
    if (genome.amplitudes.length !== this.actuators.length) {
      throw new Error('Genome length does not match actuator count')
    }
    this.currentGenome = genome
  }

  /** Жёсткость «пружинных» приводов: сила ∝ springGain × синус; предел — maxForcePerFoot. */
  setDriveStiffness(springGain: number, maxForcePerFoot: number) {
    this.springGain = Math.max(1e-6, springGain)
    this.maxForcePerFoot = Math.max(1e-6, maxForcePerFoot)
  }

  /** Видимость жёлтых маркеров в точках приложения сил (не влияет на физику). */
  setMarkerSpheresVisible(visible: boolean) {
    for (const a of this.actuators) {
      a.mesh.visible = visible
    }
  }

  applyAtTime(tSeconds: number) {
    if (!this.currentGenome || !this.tableBody) return
    const state: RobotState = {
      simTimeSeconds: tSeconds,
      com: { x: 0, y: 0, z: 0 },
      linearVelocity: { x: 0, y: 0, z: 0 }
    }
    const commands = this.computeMotorCommands(this.currentGenome, tSeconds, state)
    this.applyMotorCommands(commands)
  }

  computeMotorCommands(genome: Genome, tSeconds: number, _state: RobotState): MotorCommand[] {
    const { amplitudes, omegas, phases } = genome
    const commands: MotorCommand[] = []
    for (let i = 0; i < this.actuators.length; i++) {
      const a = this.actuators[i]
      const phase = omegas[i] * tSeconds + phases[i] + a.phaseBias
      const signal = Math.sin(phase)
      const signalVel = omegas[i] * Math.cos(phase)
      commands.push({
        jointId: `leg-${i}`,
        index: i,
        targetPosition: amplitudes[i] * signal,
        targetVelocity: amplitudes[i] * signalVel,
        targetEffort: amplitudes[i] * signal * this.springGain
      })
    }
    return commands
  }

  applyMotorCommands(commands: MotorCommand[]) {
    if (!this.tableBody) return
    const com = comWorldFromRigidBody(this.tableBody)
    const k = this.springGain
    const fCap = this.maxForcePerFoot

    for (let i = 0; i < this.actuators.length; i++) {
      const a = this.actuators[i]
      const cmd = commands[i]
      if (!cmd) continue
      const footWorld = this.bodyLocalToWorld(this.tableBody, a.pivotBodyLocal)
      let fx = a.dir.x * (cmd.targetEffort * k)
      let fy = a.dir.y * (cmd.targetEffort * k)
      let fz = a.dir.z * (cmd.targetEffort * k)
      const len = Math.hypot(fx, fy, fz)
      if (len > fCap && len > 1e-8) {
        const s = fCap / len
        fx *= s
        fy *= s
        fz *= s
      }

      const relx = footWorld.x - com.x
      const rely = footWorld.y - com.y
      const relz = footWorld.z - com.z

      this.forceBt.setValue(fx, fy, fz)
      this.relBt.setValue(relx, rely, relz)
      this.tableBody.applyForce(this.forceBt, this.relBt)

      // Маркеры двигаем всегда — при повторном показе позы не скачут.
      const glued = footWorld.clone().add(a.dir.clone().multiplyScalar(cmd.targetPosition))
      const gp = this.actuatorGroundPlane
      if (gp !== null && Number.isFinite(gp)) {
        const minCenterY = gp + a.radius + 0.004
        if (glued.y < minCenterY) glued.y = minCenterY
      }
      a.mesh.position.copy(glued)
    }
  }

  private disposeInternal() {
    for (const a of this.actuators) {
      a.mesh.parent?.remove(a.mesh)
    }
    this.actuators.length = 0
    this.tableBody = null
  }

  dispose() {
    this.disposeInternal()
  }

  get count() {
    return this.actuators.length
  }
}
