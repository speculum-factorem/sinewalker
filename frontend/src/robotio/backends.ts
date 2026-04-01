import type { ActuatorSystemRigid } from '../actuators/ActuatorSystemRigid'
import type { HardwareTelemetry, MotorCommand, PolicyTickInput, RobotBackend } from './types'

export class SimBackend implements RobotBackend {
  readonly id = 'sim' as const
  private readonly actuators: ActuatorSystemRigid

  constructor(actuators: ActuatorSystemRigid) {
    this.actuators = actuators
  }

  send(_input: PolicyTickInput, commands: MotorCommand[]) {
    this.actuators.applyMotorCommands(commands)
  }

  getLatestTelemetry(): HardwareTelemetry | null {
    return null
  }

  dispose() {}
}

export class HardwareWebSocketBackend implements RobotBackend {
  readonly id = 'hardware-websocket' as const
  private ws: WebSocket | null = null
  private readonly url: string
  private latestTelemetry: HardwareTelemetry | null = null

  constructor(url: string) {
    this.url = url
    this.open()
  }

  private open() {
    this.dispose()
    try {
      this.ws = new WebSocket(this.url)
      this.ws.addEventListener('message', (ev) => {
        try {
          const payload = JSON.parse(String(ev.data)) as { type?: string; state?: HardwareTelemetry; sensorState?: HardwareTelemetry }
          if (payload.type === 'sensor-state' && payload.state) {
            this.latestTelemetry = payload.state
          } else if (payload.type === 'ack' && payload.sensorState) {
            this.latestTelemetry = payload.sensorState
          }
        } catch {
          // ignore malformed telemetry
        }
      })
    } catch {
      this.ws = null
    }
  }

  send(input: PolicyTickInput, commands: MotorCommand[]) {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return
    const payload = {
      type: 'robot-io-tick',
      ts: Date.now(),
      timeSeconds: input.timeSeconds,
      state: input.state,
      commands
    }
    this.ws.send(JSON.stringify(payload))
  }

  getLatestTelemetry(): HardwareTelemetry | null {
    return this.latestTelemetry
  }

  dispose() {
    if (!this.ws) return
    try {
      this.ws.close()
    } catch {
      // ignore
    }
    this.ws = null
  }
}
