import * as THREE from 'three'
import type { ActuatorSystemRigid } from '../actuators/ActuatorSystemRigid'
import type { Genome } from '../actuators/ActuatorSystem'
import { comWorldFromRigidBody } from '../physics/rigidBodyThree'
import { HardwareWebSocketBackend, SimBackend } from './backends'
import type { HardwareTelemetry, MotorCommand, PolicyTickInput, RobotBackend, RobotBackendId, RobotState } from './types'

export class RobotIoLayer {
  private readonly actuators: ActuatorSystemRigid
  private backend: RobotBackend
  private backendId: RobotBackendId = 'sim'
  private wsUrl = 'ws://localhost:8765'
  private readonly velTmp = new THREE.Vector3()

  constructor(actuators: ActuatorSystemRigid) {
    this.actuators = actuators
    this.backend = new SimBackend(this.actuators)
  }

  setBackend(id: RobotBackendId, wsUrl?: string) {
    this.backend.dispose()
    this.backendId = id
    if (wsUrl && wsUrl.trim().length > 0) this.wsUrl = wsUrl.trim()
    this.backend = id === 'hardware-websocket' ? new HardwareWebSocketBackend(this.wsUrl) : new SimBackend(this.actuators)
  }

  getBackendId() {
    return this.backendId
  }

  getWsUrl() {
    return this.wsUrl
  }

  tick(genome: Genome, timeSeconds: number, tableBody: any) {
    const state = this.makeState(tableBody, timeSeconds)
    const commands = this.actuators.computeMotorCommands(genome, timeSeconds, state)
    const input: PolicyTickInput = { genome, timeSeconds, state }
    this.backend.send(input, commands)
    return commands
  }

  sendManualCommands(commands: MotorCommand[], tableBody: any, timeSeconds: number) {
    const state = this.makeState(tableBody, timeSeconds)
    const input: PolicyTickInput = {
      genome: {
        amplitudes: commands.map((c) => Math.abs(c.targetPosition)),
        omegas: commands.map(() => 0),
        phases: commands.map(() => 0)
      },
      timeSeconds,
      state
    }
    this.backend.send(input, commands)
  }

  getLatestTelemetry(): HardwareTelemetry | null {
    return this.backend.getLatestTelemetry()
  }

  dispose() {
    this.backend.dispose()
  }

  private makeState(tableBody: any, timeSeconds: number): RobotState {
    const com = comWorldFromRigidBody(tableBody)
    let lvx = 0
    let lvy = 0
    let lvz = 0
    try {
      const lv = tableBody.getLinearVelocity?.()
      if (lv) {
        lvx = lv.x()
        lvy = lv.y()
        lvz = lv.z()
      }
    } catch {
      // ignore
    }
    this.velTmp.set(lvx, lvy, lvz)
    return {
      simTimeSeconds: timeSeconds,
      com: { x: com.x, y: com.y, z: com.z },
      linearVelocity: { x: this.velTmp.x, y: this.velTmp.y, z: this.velTmp.z }
    }
  }
}
