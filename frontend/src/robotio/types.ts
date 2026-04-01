import type { Genome } from '../actuators/ActuatorSystem'

export type RobotBackendId = 'sim' | 'hardware-websocket'

export interface RobotState {
  simTimeSeconds: number
  com: { x: number; y: number; z: number }
  linearVelocity: { x: number; y: number; z: number }
}

export interface HardwareTelemetry {
  imu?: unknown
  jointState?: {
    name?: string[]
    position?: number[]
    velocity?: number[]
    effort?: number[]
  }
  twist?: unknown
  driverHeartbeat?: {
    lastAckTs?: number
    lastAckSeq?: number
    lastPingSeq?: number
  }
}

export interface MotorCommand {
  jointId: string
  index: number
  targetPosition: number
  targetVelocity: number
  targetEffort: number
}

export interface PolicyTickInput {
  genome: Genome
  timeSeconds: number
  state: RobotState
}

export interface RobotBackend {
  readonly id: RobotBackendId
  send(input: PolicyTickInput, commands: MotorCommand[]): void
  getLatestTelemetry(): HardwareTelemetry | null
  dispose(): void
}
