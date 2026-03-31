import * as THREE from 'three'
import { comWorldFromRigidBody } from '../physics/rigidBodyThree'

export interface RewardConfig {
  fallMinY?: number
  fallPenalty?: number
  comAxis?: 'x' | 'z'
  lateralPenaltyWeight?: number
  backwardPenaltyWeight?: number
  backtrackPenaltyWeight?: number
}

export interface EpisodeReward {
  reward: number
  progress: number
  fell: boolean
  comStart: THREE.Vector3
  comEnd: THREE.Vector3
}

function computeCOM(softBody: any, nodeCount: number): THREE.Vector3 {
  const nodes = softBody.get_m_nodes()
  const com = new THREE.Vector3()
  for (let i = 0; i < nodeCount; i++) {
    const node = nodes.at(i)
    const p = node.get_m_x()
    com.add(new THREE.Vector3(p.x(), p.y(), p.z()))
  }
  if (nodeCount > 0) com.multiplyScalar(1 / nodeCount)
  return com
}

export class RewardCalculator {
  private readonly fallMinY: number
  private readonly fallPenalty: number
  private readonly comAxis: 'x' | 'z'
  private readonly lateralPenaltyWeight: number
  private readonly backwardPenaltyWeight: number
  private readonly backtrackPenaltyWeight: number

  private comStart: THREE.Vector3 | null = null
  private comEnd: THREE.Vector3 | null = null
  private fell = false

  private minY = Infinity
  private nodeCount = 0
  private lastAxisPos: number | null = null
  private backwardTravel = 0
  private maxAxisPos: number | null = null
  private backtrackFromPeak = 0

  constructor(config: RewardConfig = {}) {
    this.fallMinY = config.fallMinY ?? 0.12
    this.fallPenalty = config.fallPenalty ?? 5
    this.comAxis = config.comAxis ?? 'x'
    this.lateralPenaltyWeight = Math.max(0, config.lateralPenaltyWeight ?? 0.65)
    this.backwardPenaltyWeight = Math.max(0, config.backwardPenaltyWeight ?? 2.4)
    this.backtrackPenaltyWeight = Math.max(0, config.backtrackPenaltyWeight ?? 4.5)
  }

  start(softBody: any, nodeCount: number) {
    this.nodeCount = nodeCount
    this.fell = false
    this.minY = Infinity

    this.comStart = computeCOM(softBody, nodeCount)
    this.lastAxisPos = this.comAxis === 'x' ? this.comStart.x : this.comStart.z
    this.maxAxisPos = this.lastAxisPos
    this.backwardTravel = 0
    this.backtrackFromPeak = 0
    this.comEnd = null
  }

  observeStep(softBody: any, sampleStride = 8) {
    if (this.nodeCount <= 0) return

    const nodes = softBody.get_m_nodes()
    const stride = Math.max(1, Math.floor(sampleStride))
    for (let i = 0; i < this.nodeCount; i += stride) {
      const node = nodes.at(i)
      const p = node.get_m_x()
      if (p.y() < this.minY) this.minY = p.y()
      if (p.y() < this.fallMinY) this.fell = true
    }
    const com = computeCOM(softBody, this.nodeCount)
    const axisPos = this.comAxis === 'x' ? com.x : com.z
    if (this.lastAxisPos !== null) {
      const delta = axisPos - this.lastAxisPos
      if (delta < 0) this.backwardTravel += -delta
    }
    if (this.maxAxisPos === null) {
      this.maxAxisPos = axisPos
    } else {
      this.maxAxisPos = Math.max(this.maxAxisPos, axisPos)
      this.backtrackFromPeak = Math.max(this.backtrackFromPeak, this.maxAxisPos - axisPos)
    }
    this.lastAxisPos = axisPos
  }

  finish(softBody: any): EpisodeReward {
    if (!this.comStart) throw new Error('RewardCalculator.finish() called before start()')
    this.comEnd = computeCOM(softBody, this.nodeCount)

    const start = this.comStart
    const end = this.comEnd

    const progress = this.comAxis === 'x' ? end.x - start.x : end.z - start.z
    const lateralDrift = this.comAxis === 'x' ? Math.abs(end.z - start.z) : Math.abs(end.x - start.x)
    const reward =
      progress -
      this.lateralPenaltyWeight * lateralDrift -
      this.backwardPenaltyWeight * this.backwardTravel -
      this.backtrackPenaltyWeight * this.backtrackFromPeak -
      (this.fell ? this.fallPenalty : 0)

    return {
      reward,
      progress,
      fell: this.fell,
      comStart: start.clone(),
      comEnd: end.clone()
    }
  }

  startRigid(body: any) {
    this.nodeCount = 0
    this.fell = false
    this.minY = Infinity
    this.comStart = comWorldFromRigidBody(body)
    this.lastAxisPos = this.comAxis === 'x' ? this.comStart.x : this.comStart.z
    this.maxAxisPos = this.lastAxisPos
    this.backwardTravel = 0
    this.backtrackFromPeak = 0
    this.comEnd = null
  }

  observeRigidStep(body: any) {
    const com = comWorldFromRigidBody(body)
    if (com.y < this.minY) this.minY = com.y
    if (com.y < this.fallMinY) this.fell = true
    const axisPos = this.comAxis === 'x' ? com.x : com.z
    if (this.lastAxisPos !== null) {
      const delta = axisPos - this.lastAxisPos
      if (delta < 0) this.backwardTravel += -delta
    }
    if (this.maxAxisPos === null) {
      this.maxAxisPos = axisPos
    } else {
      this.maxAxisPos = Math.max(this.maxAxisPos, axisPos)
      this.backtrackFromPeak = Math.max(this.backtrackFromPeak, this.maxAxisPos - axisPos)
    }
    this.lastAxisPos = axisPos
  }

  finishRigid(body: any): EpisodeReward {
    if (!this.comStart) throw new Error('RewardCalculator.finishRigid() called before startRigid()')
    this.comEnd = comWorldFromRigidBody(body)

    const start = this.comStart
    const end = this.comEnd

    const progress = this.comAxis === 'x' ? end.x - start.x : end.z - start.z
    const lateralDrift = this.comAxis === 'x' ? Math.abs(end.z - start.z) : Math.abs(end.x - start.x)
    const reward =
      progress -
      this.lateralPenaltyWeight * lateralDrift -
      this.backwardPenaltyWeight * this.backwardTravel -
      this.backtrackPenaltyWeight * this.backtrackFromPeak -
      (this.fell ? this.fallPenalty : 0)

    return {
      reward,
      progress,
      fell: this.fell,
      comStart: start.clone(),
      comEnd: end.clone()
    }
  }
}

