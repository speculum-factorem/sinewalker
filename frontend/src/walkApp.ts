import * as THREE from 'three'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js'
import { mergeGeometries, mergeVertices } from 'three/addons/utils/BufferGeometryUtils.js'
import spiderObjRaw from './assets/models/spider3dmodel/TRANTULA/TRANTULA.OBJ?raw'
import spiderObjUrl from './assets/models/spider3dmodel/TRANTULA/TRANTULA.OBJ?url'

import { AmmoWorld, type RigidBodyHandle } from './physics/AmmoWorld'
import { computeFootPivotsLocal, geometryPositionsFlat } from './physics/geometryFeet'
import {
  applyRigidBodyElasticityMaterial,
  comWorldFromRigidBody,
  createRigidBodyResetPool,
  resetRigidBodyToPose,
  syncMeshFromRigidBody,
  type RigidBodyResetPool
} from './physics/rigidBodyThree'
import type { Genome } from './actuators/ActuatorSystem'
import { ActuatorSystemRigid } from './actuators/ActuatorSystemRigid'
import { RewardCalculator } from './training/RewardCalculator'
import { TrainerEvolution, type EvaluationStage } from './training/TrainerEvolution'
import { TrainerCmaEs } from './training/TrainerCmaEs'
import { TrainerDe } from './training/TrainerDe'
import { TrainerPso } from './training/TrainerPso'
import { TrainerCem } from './training/TrainerCem'
import { RobotIoLayer } from './robotio/RobotIoLayer'
import type { MotorCommand, RobotBackendId } from './robotio/types'
import {
  applyLegVertexFlex,
  footOutwardDirsXZ,
  legRegionYMax,
  snapshotGeometryPositions
} from './render/tableLegVisualFlex'

/** Fixed evolution / actuator layout (no UI). */
const TRAINING_POPULATION = 20
const DEFAULT_TRAINING_GENERATIONS = 22
const TRAINING_GENERATIONS_MIN = 1
const TRAINING_GENERATIONS_MAX = 2000
const EPISODE_STEPS = 230
const DT_SECONDS = 0.016
const VIZ_SPEED = 6
const EVAL_ROLLOUTS = 3
const EVAL_MIN_STEP_RATIO = 0.42
const EARLY_STOP_GRACE_STEPS = 70
const EARLY_STOP_WINDOW_STEPS = 55
const EARLY_STOP_MIN_PROGRESS = 0.03
const EARLY_STOP_FALL_MIN_Y = 0.12
const EARLY_STOP_BACKTRACK_THRESHOLD = 0.08
const PLAYBACK_BACKTRACK_THRESHOLD = 0.07
const TRAINING_LATERAL_PENALTY_WEIGHT = 0.8
const TRAINING_BACKWARD_PENALTY_WEIGHT = 2.6
const TRAINING_BACKTRACK_PENALTY_WEIGHT = 4.8

type TrainingAlgorithmId = 'evolution' | 'cmaes' | 'de' | 'pso' | 'cem'
type TerrainId = 'flat' | 'ice' | 'gravel' | 'slope' | 'steps' | 'rough'
const GROUND_PLANE_Y = 0
/** Ползунок и визуальный изгиб, и физика приводов; это значение = множитель 1.0 к базовой жёсткости. */
const DEFAULT_ELASTICITY_SLIDER = 0.086
const BASE_ACTUATOR_SPRING_GAIN = 38
const BASE_ACTUATOR_MAX_FORCE = 46
const ELASTICITY_MULT_MIN = 0.25
const ELASTICITY_MULT_MAX = 2.2

const TABLE_RIGID_MASS = 14
type FigureId = 'table' | 'stool' | 'bench' | 'spider' | 'turtle' | 'beetle' | 'crab' | 'rocket'
const POLICY_FILE_VERSION = 1

interface PolicySnapshot {
  version: number
  createdAt: string
  algorithm: TrainingAlgorithmId
  figure: FigureId
  trainingGenerations: number
  elasticitySlider: number
  showActuatorSpheres: boolean
  terrain: TerrainId
  randomTerrainPerEpisode: boolean
  robotBackend: RobotBackendId
  robotWsUrl: string
  genome: Genome
}

interface TerrainBodySpec {
  halfExtents: THREE.Vector3
  position: THREE.Vector3
  rotation?: THREE.Quaternion
  color: number
}

interface TerrainProfile {
  id: TerrainId
  label: string
  friction: number
  restitution: number
  spawnYOffset: number
  specs: TerrainBodySpec[]
}

interface TerrainBodyInstance {
  handle: RigidBodyHandle
  mesh: THREE.Mesh
  activePosition: THREE.Vector3
  activeRotation: THREE.Quaternion
  hiddenPosition: THREE.Vector3
}

const TERRAIN_PHYSICS_DEBUG_MESHES = false

function normalizeGeometry(geometry: THREE.BufferGeometry): THREE.BufferGeometry {
  const cloned = geometry.clone()
  cloned.computeBoundingBox()
  const bbox = cloned.boundingBox
  if (!bbox) return cloned

  const size = new THREE.Vector3()
  bbox.getSize(size)
  const maxDim = Math.max(size.x, size.y, size.z)
  const scale = maxDim > 0 ? 1 / maxDim : 1

  cloned.scale(scale, scale, scale)

  const center = new THREE.Vector3()
  bbox.getCenter(center)
  cloned.translate(-center.x, -center.y, -center.z)
  return cloned
}

function randomGenome(actuatorCount: number): Genome {
  const amplitudes: number[] = []
  const omegas: number[] = []
  const phases: number[] = []
  for (let i = 0; i < actuatorCount; i++) {
    amplitudes.push(0.05 + Math.random() * 0.1)
    omegas.push(1.0 + Math.random() * 4.0)
    const gait = i % 2 === 0 ? 0 : Math.PI
    phases.push(gait + (Math.random() - 0.5) * 0.45)
  }
  return { amplitudes, omegas, phases }
}

function mergeAndNormalize(pieces: THREE.BufferGeometry[], weldTolerance = 0.06): THREE.BufferGeometry {
  const normalizedPieces = pieces.map((g) => {
    // Ensure compatible attributes for mergeGeometries.
    const n = g.index ? g.toNonIndexed() : g.clone()
    const attrs = Object.keys(n.attributes)
    for (const k of attrs) {
      if (k !== 'position') n.deleteAttribute(k)
    }
    return n
  })
  const merged = mergeGeometries(normalizedPieces, false)
  if (!merged) throw new Error('Failed to merge figure parts')
  for (const p of pieces) p.dispose()
  for (const p of normalizedPieces) p.dispose()
  const welded = mergeVertices(merged, weldTolerance)
  merged.dispose()
  welded.computeVertexNormals()
  return normalizeGeometry(welded)
}

function mergedGeometryFromObject(root: THREE.Object3D): THREE.BufferGeometry | null {
  const parts: THREE.BufferGeometry[] = []
  root.traverse((node) => {
    const m = node as THREE.Mesh
    const g = m.geometry as THREE.BufferGeometry | undefined
    if (!g || !g.getAttribute('position')) return
    const cloned = g.index ? g.toNonIndexed() : g.clone()
    cloned.applyMatrix4(m.matrixWorld)
    // OBJ parts may have mismatched attributes (e.g. uv on some parts only).
    const attrs = Object.keys(cloned.attributes)
    for (const k of attrs) {
      if (k !== 'position') cloned.deleteAttribute(k)
    }
    parts.push(cloned)
  })
  if (parts.length === 0) return null
  // Imported meshes already have dense topology; coarse weld destroys silhouette.
  return mergeAndNormalize(parts, 0.0005)
}

function createTableGeometry(): THREE.BufferGeometry {
  const pieces: THREE.BufferGeometry[] = []
  const top = new THREE.BoxGeometry(1.6, 0.18, 1.0, 2, 1, 2)
  top.applyMatrix4(new THREE.Matrix4().makeTranslation(0, 0.65, 0))
  pieces.push(top.toNonIndexed())
  top.dispose()
  const legTpl = new THREE.CylinderGeometry(0.12, 0.12, 1.15, 8, 2)
  const legXZ: Array<[number, number]> = [
    [0.65, 0.35],
    [-0.65, 0.35],
    [0.65, -0.35],
    [-0.65, -0.35]
  ]
  for (const [x, z] of legXZ) {
    const leg = legTpl.clone()
    leg.applyMatrix4(new THREE.Matrix4().makeTranslation(x, 0.18, z))
    pieces.push(leg.toNonIndexed())
    leg.dispose()
  }
  legTpl.dispose()
  return mergeAndNormalize(pieces)
}

function createStoolGeometry(): THREE.BufferGeometry {
  const pieces: THREE.BufferGeometry[] = []
  const seat = new THREE.CylinderGeometry(0.75, 0.75, 0.16, 20, 1)
  seat.applyMatrix4(new THREE.Matrix4().makeTranslation(0, 0.62, 0))
  pieces.push(seat.toNonIndexed())
  seat.dispose()
  const legTpl = new THREE.CylinderGeometry(0.095, 0.095, 1.05, 10, 2)
  const legXZ: Array<[number, number]> = [
    [0.42, 0.42],
    [-0.42, 0.42],
    [0.42, -0.42],
    [-0.42, -0.42]
  ]
  for (const [x, z] of legXZ) {
    const leg = legTpl.clone()
    leg.applyMatrix4(new THREE.Matrix4().makeTranslation(x, 0.12, z))
    pieces.push(leg.toNonIndexed())
    leg.dispose()
  }
  legTpl.dispose()
  return mergeAndNormalize(pieces)
}

function createBenchGeometry(): THREE.BufferGeometry {
  const pieces: THREE.BufferGeometry[] = []
  const top = new THREE.BoxGeometry(2.0, 0.16, 0.72, 3, 1, 2)
  top.applyMatrix4(new THREE.Matrix4().makeTranslation(0, 0.66, 0))
  pieces.push(top.toNonIndexed())
  top.dispose()
  const legTpl = new THREE.BoxGeometry(0.18, 1.08, 0.18, 1, 2, 1)
  const legXZ: Array<[number, number]> = [
    [0.82, 0.24],
    [-0.82, 0.24],
    [0.82, -0.24],
    [-0.82, -0.24]
  ]
  for (const [x, z] of legXZ) {
    const leg = legTpl.clone()
    leg.applyMatrix4(new THREE.Matrix4().makeTranslation(x, 0.12, z))
    pieces.push(leg.toNonIndexed())
    leg.dispose()
  }
  legTpl.dispose()
  return mergeAndNormalize(pieces)
}

function createSpiderGeometry(): THREE.BufferGeometry {
  const pieces: THREE.BufferGeometry[] = []
  // Realistic-ish spider silhouette: big abdomen, compact cephalothorax, long low legs.
  const abdomen = new THREE.SphereGeometry(0.44, 24, 14)
  abdomen.applyMatrix4(new THREE.Matrix4().makeScale(1.18, 0.9, 1.05))
  abdomen.applyMatrix4(new THREE.Matrix4().makeTranslation(-0.2, 0.4, 0))
  pieces.push(abdomen.toNonIndexed())
  abdomen.dispose()

  const cephalothorax = new THREE.SphereGeometry(0.24, 20, 12)
  cephalothorax.applyMatrix4(new THREE.Matrix4().makeScale(1.0, 0.82, 0.9))
  cephalothorax.applyMatrix4(new THREE.Matrix4().makeTranslation(0.3, 0.36, 0))
  pieces.push(cephalothorax.toNonIndexed())
  cephalothorax.dispose()

  const pedicel = new THREE.CylinderGeometry(0.048, 0.04, 0.2, 10, 1)
  pedicel.rotateZ(Math.PI / 2)
  pedicel.applyMatrix4(new THREE.Matrix4().makeTranslation(0.05, 0.37, 0))
  pieces.push(pedicel.toNonIndexed())
  pedicel.dispose()

  const fangs = new THREE.ConeGeometry(0.045, 0.16, 8, 1)
  for (const side of [1, -1] as const) {
    const fang = fangs.clone()
    fang.rotateZ(Math.PI / 2)
    fang.rotateY(side * 0.28)
    fang.applyMatrix4(new THREE.Matrix4().makeTranslation(0.5, 0.28, side * 0.07))
    pieces.push(fang.toNonIndexed())
    fang.dispose()
  }
  fangs.dispose()

  const coxaTpl = new THREE.CylinderGeometry(0.035, 0.03, 0.18, 8, 1)
  const femurTpl = new THREE.CylinderGeometry(0.03, 0.024, 0.48, 8, 1)
  const tibiaTpl = new THREE.CylinderGeometry(0.024, 0.016, 0.42, 8, 1)
  const tarsusTpl = new THREE.CylinderGeometry(0.017, 0.012, 0.26, 8, 1)
  for (let i = 0; i < 8; i++) {
    const side = i < 4 ? 1 : -1
    const idx = i % 4
    const rootX = 0.24 - idx * 0.16
    const rootZ = side * (0.15 + idx * 0.09)
    const yaw = side * (Math.PI * 0.62 - idx * 0.11)

    const shoulder = new THREE.SphereGeometry(0.042, 9, 7)
    shoulder.applyMatrix4(new THREE.Matrix4().makeTranslation(rootX, 0.31, rootZ))
    pieces.push(shoulder.toNonIndexed())
    shoulder.dispose()

    const coxa = coxaTpl.clone()
    coxa.rotateZ(Math.PI / 2)
    coxa.rotateY(yaw)
    coxa.applyMatrix4(
      new THREE.Matrix4().makeTranslation(rootX + Math.cos(yaw) * 0.07, 0.29, rootZ + Math.sin(yaw) * 0.07)
    )
    pieces.push(coxa.toNonIndexed())
    coxa.dispose()

    const femurX = rootX + Math.cos(yaw) * 0.19
    const femurZ = rootZ + Math.sin(yaw) * 0.19
    const femur = femurTpl.clone()
    femur.rotateZ(Math.PI / 2)
    femur.rotateY(yaw + side * 0.14)
    femur.applyMatrix4(new THREE.Matrix4().makeTranslation(femurX, 0.2, femurZ))
    pieces.push(femur.toNonIndexed())
    femur.dispose()

    const kneeX = rootX + Math.cos(yaw) * 0.36
    const kneeZ = rootZ + Math.sin(yaw) * 0.36
    const knee = new THREE.SphereGeometry(0.032, 8, 6)
    knee.applyMatrix4(new THREE.Matrix4().makeTranslation(kneeX, 0.12, kneeZ))
    pieces.push(knee.toNonIndexed())
    knee.dispose()

    const tibia = tibiaTpl.clone()
    tibia.rotateZ(Math.PI / 2)
    tibia.rotateY(yaw + side * 0.28)
    tibia.applyMatrix4(
      new THREE.Matrix4().makeTranslation(kneeX + Math.cos(yaw) * 0.14, 0.07, kneeZ + Math.sin(yaw) * 0.14)
    )
    pieces.push(tibia.toNonIndexed())
    tibia.dispose()

    const tarsus = tarsusTpl.clone()
    tarsus.rotateZ(Math.PI / 2)
    tarsus.rotateY(yaw + side * 0.34)
    tarsus.applyMatrix4(
      new THREE.Matrix4().makeTranslation(kneeX + Math.cos(yaw) * 0.27, 0.04, kneeZ + Math.sin(yaw) * 0.27)
    )
    pieces.push(tarsus.toNonIndexed())
    tarsus.dispose()
  }
  coxaTpl.dispose()
  femurTpl.dispose()
  tibiaTpl.dispose()
  tarsusTpl.dispose()
  return mergeAndNormalize(pieces)
}

function createTurtleGeometry(): THREE.BufferGeometry {
  const pieces: THREE.BufferGeometry[] = []
  const abdomen = new THREE.SphereGeometry(0.36, 20, 12)
  abdomen.applyMatrix4(new THREE.Matrix4().makeScale(1.35, 0.85, 1.0))
  abdomen.applyMatrix4(new THREE.Matrix4().makeTranslation(-0.34, 0.42, 0))
  pieces.push(abdomen.toNonIndexed())
  abdomen.dispose()

  const thorax = new THREE.SphereGeometry(0.24, 18, 11)
  thorax.applyMatrix4(new THREE.Matrix4().makeScale(1.15, 0.85, 0.95))
  thorax.applyMatrix4(new THREE.Matrix4().makeTranslation(0.03, 0.39, 0))
  pieces.push(thorax.toNonIndexed())
  thorax.dispose()

  const head = new THREE.SphereGeometry(0.18, 16, 10)
  head.applyMatrix4(new THREE.Matrix4().makeScale(1.1, 0.9, 0.9))
  head.applyMatrix4(new THREE.Matrix4().makeTranslation(0.43, 0.39, 0))
  pieces.push(head.toNonIndexed())
  head.dispose()

  const petiole = new THREE.CylinderGeometry(0.065, 0.05, 0.16, 10, 1)
  petiole.rotateZ(Math.PI / 2)
  petiole.applyMatrix4(new THREE.Matrix4().makeTranslation(-0.16, 0.41, 0))
  pieces.push(petiole.toNonIndexed())
  petiole.dispose()

  const upperLegTpl = new THREE.CylinderGeometry(0.052, 0.044, 0.27, 8, 1)
  const lowerLegTpl = new THREE.CylinderGeometry(0.044, 0.032, 0.34, 8, 1)
  const legRoots: Array<[number, number]> = [
    [0.14, 0.28],
    [-0.02, 0.32],
    [-0.26, 0.28],
    [0.14, -0.28],
    [-0.02, -0.32],
    [-0.26, -0.28]
  ]
  for (const [x, z] of legRoots) {
    const side = z >= 0 ? 1 : -1
    const yaw = side * (Math.PI * 0.43 + x * 0.22)
    const hip = new THREE.SphereGeometry(0.053, 9, 7)
    hip.applyMatrix4(new THREE.Matrix4().makeTranslation(x, 0.29, z * 0.94))
    pieces.push(hip.toNonIndexed())
    hip.dispose()

    const upper = upperLegTpl.clone()
    upper.rotateZ(Math.PI / 2)
    upper.rotateY(yaw)
    upper.applyMatrix4(new THREE.Matrix4().makeTranslation(x + Math.cos(yaw) * 0.1, 0.22, z * 0.98 + Math.sin(yaw) * 0.1))
    pieces.push(upper.toNonIndexed())
    upper.dispose()

    const lower = lowerLegTpl.clone()
    lower.rotateZ(Math.PI / 2)
    lower.rotateY(yaw + side * 0.17)
    lower.applyMatrix4(new THREE.Matrix4().makeTranslation(x + Math.cos(yaw) * 0.22, 0.12, z * 0.98 + Math.sin(yaw) * 0.22))
    pieces.push(lower.toNonIndexed())
    lower.dispose()
  }
  upperLegTpl.dispose()
  lowerLegTpl.dispose()

  const antennaTpl = new THREE.CylinderGeometry(0.012, 0.008, 0.22, 6, 1)
  for (const side of [1, -1] as const) {
    const ant = antennaTpl.clone()
    ant.rotateZ(Math.PI / 3.2)
    ant.rotateY(side * 0.36)
    ant.applyMatrix4(new THREE.Matrix4().makeTranslation(0.56, 0.53, side * 0.06))
    pieces.push(ant.toNonIndexed())
    ant.dispose()
  }
  antennaTpl.dispose()
  return mergeAndNormalize(pieces)
}


function createBeetleGeometry(): THREE.BufferGeometry {
  const pieces: THREE.BufferGeometry[] = []
  const thorax = new THREE.CapsuleGeometry(0.24, 0.66, 10, 16)
  thorax.rotateZ(Math.PI / 2)
  thorax.applyMatrix4(new THREE.Matrix4().makeScale(1.0, 0.72, 0.85))
  thorax.applyMatrix4(new THREE.Matrix4().makeTranslation(0.08, 0.42, 0))
  pieces.push(thorax.toNonIndexed())
  thorax.dispose()

  const abdomen = new THREE.SphereGeometry(0.34, 18, 12)
  abdomen.applyMatrix4(new THREE.Matrix4().makeScale(1.26, 0.76, 0.9))
  abdomen.applyMatrix4(new THREE.Matrix4().makeTranslation(-0.36, 0.43, 0))
  pieces.push(abdomen.toNonIndexed())
  abdomen.dispose()

  const head = new THREE.SphereGeometry(0.18, 14, 10)
  head.applyMatrix4(new THREE.Matrix4().makeScale(1.05, 0.9, 0.85))
  head.applyMatrix4(new THREE.Matrix4().makeTranslation(0.58, 0.4, 0))
  pieces.push(head.toNonIndexed())
  head.dispose()

  const elytraCut = new THREE.BoxGeometry(0.64, 0.04, 0.05, 1, 1, 1)
  elytraCut.applyMatrix4(new THREE.Matrix4().makeTranslation(-0.14, 0.64, 0))
  pieces.push(elytraCut.toNonIndexed())
  elytraCut.dispose()

  const upperLegTpl = new THREE.CylinderGeometry(0.052, 0.044, 0.28, 8, 1)
  const lowerLegTpl = new THREE.CylinderGeometry(0.045, 0.034, 0.34, 8, 1)
  const legRoots: Array<[number, number]> = [
    [0.38, 0.31],
    [0.02, 0.34],
    [-0.34, 0.31],
    [0.38, -0.31],
    [0.02, -0.34],
    [-0.34, -0.31]
  ]
  for (const [x, z] of legRoots) {
    const side = z >= 0 ? 1 : -1
    const yaw = side * (Math.PI * 0.42 + x * 0.22)

    const hip = new THREE.SphereGeometry(0.055, 10, 8)
    hip.applyMatrix4(new THREE.Matrix4().makeTranslation(x, 0.29, z * 0.92))
    pieces.push(hip.toNonIndexed())
    hip.dispose()

    const upper = upperLegTpl.clone()
    upper.rotateZ(Math.PI / 2)
    upper.rotateY(yaw)
    upper.applyMatrix4(new THREE.Matrix4().makeTranslation(x + Math.cos(yaw) * 0.11, 0.22, z * 0.98 + Math.sin(yaw) * 0.11))
    pieces.push(upper.toNonIndexed())
    upper.dispose()

    const kneeX = x + Math.cos(yaw) * 0.2
    const kneeZ = z * 0.98 + Math.sin(yaw) * 0.2
    const lower = lowerLegTpl.clone()
    lower.rotateZ(Math.PI / 2)
    lower.rotateY(yaw + side * 0.16)
    lower.applyMatrix4(new THREE.Matrix4().makeTranslation(kneeX + Math.cos(yaw) * 0.12, 0.12, kneeZ + Math.sin(yaw) * 0.12))
    pieces.push(lower.toNonIndexed())
    lower.dispose()
  }
  upperLegTpl.dispose()
  lowerLegTpl.dispose()
  return mergeAndNormalize(pieces)
}

function createCrabGeometry(): THREE.BufferGeometry {
  const pieces: THREE.BufferGeometry[] = []
  const body = new THREE.CylinderGeometry(0.58, 0.66, 0.24, 22, 1)
  body.rotateX(Math.PI / 2)
  body.applyMatrix4(new THREE.Matrix4().makeScale(1.06, 0.52, 0.95))
  body.applyMatrix4(new THREE.Matrix4().makeTranslation(0, 0.4, 0))
  pieces.push(body.toNonIndexed())
  body.dispose()

  const carapaceRim = new THREE.TorusGeometry(0.54, 0.035, 10, 28)
  carapaceRim.rotateX(Math.PI / 2)
  carapaceRim.applyMatrix4(new THREE.Matrix4().makeScale(1.05, 0.55, 1.0))
  carapaceRim.applyMatrix4(new THREE.Matrix4().makeTranslation(0, 0.43, 0))
  pieces.push(carapaceRim.toNonIndexed())
  carapaceRim.dispose()

  const upperLegTpl = new THREE.CylinderGeometry(0.056, 0.046, 0.28, 8, 1)
  const lowerLegTpl = new THREE.CylinderGeometry(0.048, 0.036, 0.3, 8, 1)
  const roots: Array<[number, number, number]> = [
    [0.4, 0.22, 1],
    [0.0, 0.2, 1],
    [-0.4, 0.22, 1],
    [0.4, 0.22, -1],
    [0.0, 0.2, -1],
    [-0.4, 0.22, -1]
  ]
  for (const [x, y, side] of roots) {
    const yaw = side > 0 ? Math.PI / 2 : -Math.PI / 2
    const shoulder = new THREE.SphereGeometry(0.058, 9, 7)
    shoulder.applyMatrix4(new THREE.Matrix4().makeTranslation(x, y, side * 0.44))
    pieces.push(shoulder.toNonIndexed())
    shoulder.dispose()

    const upper = upperLegTpl.clone()
    upper.rotateX(Math.PI / 2)
    upper.rotateY(yaw)
    upper.applyMatrix4(new THREE.Matrix4().makeTranslation(x, y - 0.04, side * 0.57))
    pieces.push(upper.toNonIndexed())
    upper.dispose()

    const lower = lowerLegTpl.clone()
    lower.rotateX(Math.PI / 2)
    lower.rotateY(yaw + side * 0.18)
    lower.applyMatrix4(new THREE.Matrix4().makeTranslation(x + side * 0.02, y - 0.12, side * 0.7))
    pieces.push(lower.toNonIndexed())
    lower.dispose()
  }
  upperLegTpl.dispose()
  lowerLegTpl.dispose()

  const clawTpl = new THREE.CylinderGeometry(0.052, 0.04, 0.32, 8, 1)
  for (const side of [1, -1] as const) {
    const claw = clawTpl.clone()
    claw.rotateX(Math.PI / 2)
    claw.rotateY(side > 0 ? Math.PI / 2 : -Math.PI / 2)
    claw.applyMatrix4(new THREE.Matrix4().makeTranslation(0.6, 0.28, side * 0.5))
    pieces.push(claw.toNonIndexed())
    claw.dispose()
  }
  clawTpl.dispose()
  return mergeAndNormalize(pieces)
}

function createRocketGeometry(): THREE.BufferGeometry {
  const pieces: THREE.BufferGeometry[] = []
  // Mantis-like tall profile.
  const abdomen = new THREE.CapsuleGeometry(0.2, 0.72, 10, 12)
  abdomen.rotateZ(Math.PI / 2)
  abdomen.applyMatrix4(new THREE.Matrix4().makeScale(1.22, 0.78, 0.84))
  abdomen.applyMatrix4(new THREE.Matrix4().makeTranslation(-0.24, 0.48, 0))
  pieces.push(abdomen.toNonIndexed())
  abdomen.dispose()

  const thorax = new THREE.CapsuleGeometry(0.16, 0.64, 10, 12)
  thorax.applyMatrix4(new THREE.Matrix4().makeScale(0.82, 1.34, 0.72))
  thorax.applyMatrix4(new THREE.Matrix4().makeTranslation(0.18, 0.78, 0))
  pieces.push(thorax.toNonIndexed())
  thorax.dispose()

  const head = new THREE.SphereGeometry(0.16, 14, 10)
  head.applyMatrix4(new THREE.Matrix4().makeScale(0.95, 1.05, 0.85))
  head.applyMatrix4(new THREE.Matrix4().makeTranslation(0.22, 1.24, 0))
  pieces.push(head.toNonIndexed())
  head.dispose()

  const eyeL = new THREE.SphereGeometry(0.055, 10, 8)
  eyeL.applyMatrix4(new THREE.Matrix4().makeTranslation(0.28, 1.28, 0.09))
  pieces.push(eyeL.toNonIndexed())
  eyeL.dispose()
  const eyeR = new THREE.SphereGeometry(0.055, 10, 8)
  eyeR.applyMatrix4(new THREE.Matrix4().makeTranslation(0.28, 1.28, -0.09))
  pieces.push(eyeR.toNonIndexed())
  eyeR.dispose()

  const forearmTpl = new THREE.CylinderGeometry(0.05, 0.036, 0.34, 8, 1)
  for (const side of [1, -1] as const) {
    const upper = forearmTpl.clone()
    upper.rotateZ(Math.PI / 2.4)
    upper.rotateY(side * 0.5)
    upper.applyMatrix4(new THREE.Matrix4().makeTranslation(0.28, 0.98, side * 0.14))
    pieces.push(upper.toNonIndexed())
    upper.dispose()

    const lower = forearmTpl.clone()
    lower.rotateZ(Math.PI / 2.1)
    lower.rotateY(side * 0.72)
    lower.applyMatrix4(new THREE.Matrix4().makeTranslation(0.42, 0.76, side * 0.22))
    pieces.push(lower.toNonIndexed())
    lower.dispose()
  }
  forearmTpl.dispose()

  const rearLegTpl = new THREE.CylinderGeometry(0.052, 0.038, 0.38, 8, 1)
  const rearRoots: Array<[number, number]> = [
    [0.06, 0.22],
    [-0.18, 0.26],
    [0.06, -0.22],
    [-0.18, -0.26]
  ]
  for (const [x, z] of rearRoots) {
    const side = z >= 0 ? 1 : -1
    const yaw = side * (Math.PI * 0.44 + x * 0.25)
    const leg = rearLegTpl.clone()
    leg.rotateZ(Math.PI / 2)
    leg.rotateY(yaw)
    leg.applyMatrix4(new THREE.Matrix4().makeTranslation(x + Math.cos(yaw) * 0.1, 0.26, z + Math.sin(yaw) * 0.1))
    pieces.push(leg.toNonIndexed())
    leg.dispose()
  }
  rearLegTpl.dispose()

  const antennaTpl = new THREE.CylinderGeometry(0.01, 0.007, 0.28, 6, 1)
  for (const side of [1, -1] as const) {
    const ant = antennaTpl.clone()
    ant.rotateZ(Math.PI / 4)
    ant.rotateY(side * 0.26)
    ant.applyMatrix4(new THREE.Matrix4().makeTranslation(0.28, 1.42, side * 0.05))
    pieces.push(ant.toNonIndexed())
    ant.dispose()
  }
  antennaTpl.dispose()
  return mergeAndNormalize(pieces)
}

function createGeometryByFigure(kind: FigureId): THREE.BufferGeometry {
  if (kind === 'spider') return createSpiderGeometry()
  if (kind === 'turtle') return createTurtleGeometry()
  if (kind === 'beetle') return createBeetleGeometry()
  if (kind === 'crab') return createCrabGeometry()
  if (kind === 'rocket') return createRocketGeometry()
  if (kind === 'stool') return createStoolGeometry()
  if (kind === 'bench') return createBenchGeometry()
  return createTableGeometry()
}

function figureLabel(kind: FigureId): string {
  if (kind === 'spider') return 'Паук'
  if (kind === 'turtle') return 'Муравей'
  if (kind === 'beetle') return 'Жук'
  if (kind === 'crab') return 'Краб'
  if (kind === 'rocket') return 'Богомол'
  if (kind === 'stool') return 'Табурет'
  if (kind === 'bench') return 'Скамья'
  return 'Стол'
}

function actuatorCountForFigure(kind: FigureId): number {
  if (kind === 'spider') return 8
  if (kind === 'beetle' || kind === 'crab' || kind === 'turtle') return 6
  return 4
}

function terrainLabel(kind: TerrainId): string {
  if (kind === 'ice') return 'Лед'
  if (kind === 'gravel') return 'Гравий'
  if (kind === 'slope') return 'Уклон'
  if (kind === 'steps') return 'Ступеньки'
  if (kind === 'rough') return 'Шумовой рельеф'
  return 'Плоский'
}

function buildTerrainProfiles(): TerrainProfile[] {
  const flat: TerrainProfile = {
    id: 'flat',
    label: 'Плоский',
    friction: 1.0,
    restitution: 0.0,
    spawnYOffset: 0,
    specs: [{ halfExtents: new THREE.Vector3(10, 1.5, 10), position: new THREE.Vector3(0, -1.5, 0), color: 0xffffff }]
  }
  const ice: TerrainProfile = {
    id: 'ice',
    label: 'Лед',
    friction: 0.06,
    restitution: 0.03,
    spawnYOffset: 0,
    specs: [{ halfExtents: new THREE.Vector3(10, 1.5, 10), position: new THREE.Vector3(0, -1.5, 0), color: 0xcdeeff }]
  }
  const gravel: TerrainProfile = {
    id: 'gravel',
    label: 'Гравий',
    friction: 1.8,
    restitution: 0.0,
    spawnYOffset: 0.01,
    specs: [{ halfExtents: new THREE.Vector3(10, 1.5, 10), position: new THREE.Vector3(0, -1.5, 0), color: 0xcec4b7 }]
  }
  const slopeAngle = THREE.MathUtils.degToRad(8)
  const slopeQuat = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), slopeAngle)
  const slope: TerrainProfile = {
    id: 'slope',
    label: 'Уклон',
    friction: 0.9,
    restitution: 0.0,
    spawnYOffset: 0.14,
    specs: [
      {
        halfExtents: new THREE.Vector3(10, 1.5, 10),
        position: new THREE.Vector3(0, -1.5, 0),
        rotation: slopeQuat,
        color: 0xf5e7cc
      }
    ]
  }
  const stepsSpecs: TerrainBodySpec[] = []
  for (let i = 0; i < 9; i++) {
    const x = -4 + i
    const h = 0.05 + i * 0.04
    stepsSpecs.push({
      halfExtents: new THREE.Vector3(0.5, h, 6),
      position: new THREE.Vector3(x, -h, 0),
      color: 0xd9d9d9
    })
  }
  const steps: TerrainProfile = {
    id: 'steps',
    label: 'Ступеньки',
    friction: 1.1,
    restitution: 0.0,
    spawnYOffset: 0.05,
    specs: stepsSpecs
  }
  const roughSpecs: TerrainBodySpec[] = []
  const roughCountX = 8
  const roughCountZ = 5
  for (let ix = 0; ix < roughCountX; ix++) {
    for (let iz = 0; iz < roughCountZ; iz++) {
      const jitter = (Math.sin(ix * 2.31 + iz * 1.73) + Math.cos(ix * 1.17 - iz * 2.09)) * 0.5
      const h = 0.22 + Math.max(0.02, (jitter + 1) * 0.14)
      roughSpecs.push({
        halfExtents: new THREE.Vector3(1.25, h, 1.25),
        position: new THREE.Vector3((ix - (roughCountX - 1) * 0.5) * 2.5, -h, (iz - (roughCountZ - 1) * 0.5) * 2.5),
        color: 0xbfc7cf
      })
    }
  }
  const rough: TerrainProfile = {
    id: 'rough',
    label: 'Шумовой рельеф',
    friction: 1.0,
    restitution: 0.0,
    spawnYOffset: 0.11,
    specs: roughSpecs
  }
  return [flat, ice, gravel, slope, steps, rough]
}

export function createWalkApp(root: HTMLDivElement) {
  root.innerHTML = `
    <div id="viewport"></div>
    <div id="hud">
      <h2>Стол на плоскости</h2>
      <p class="hud-desc">Жёсткий convex hull на полу. Ползунок: приводы, визуальный изгиб и материал всего стола (отскок, демпфирование).</p>
      <label for="figureType">Фигура</label>
      <select id="figureType">
        <option value="table">Стол</option>
        <option value="stool">Табурет</option>
        <option value="bench">Скамья</option>
        <option value="spider">Паук</option>
        <option value="turtle">Муравей</option>
        <option value="beetle">Жук</option>
        <option value="crab">Краб</option>
        <option value="rocket">Богомол</option>
      </select>
      <label for="flexGain">Упругость объекта</label>
      <div class="row">
        <input type="range" id="flexGain" min="0.04" max="0.14" step="0.002" value="${DEFAULT_ELASTICITY_SLIDER}" />
        <span class="flex-gain-val" id="flexGainVal">${DEFAULT_ELASTICITY_SLIDER.toFixed(3)}</span>
      </div>
      <label for="sphereViz">Сферы приводов</label>
      <div class="row">
        <input type="range" id="sphereViz" min="0" max="1" step="1" value="1" />
        <span class="flex-gain-val" id="sphereVizVal">вкл</span>
      </div>
      <label>Симуляция</label>
      <div class="row">
        <button type="button" id="simStart">Запустить</button>
        <button type="button" id="simStop" disabled>Остановить</button>
      </div>
      <label for="trainGenerations">Поколений обучения</label>
      <input type="number" id="trainGenerations" min="${TRAINING_GENERATIONS_MIN}" max="${TRAINING_GENERATIONS_MAX}" step="1" value="${DEFAULT_TRAINING_GENERATIONS}" />
      <label for="trainAlgorithm">Алгоритм обучения</label>
      <select id="trainAlgorithm">
        <option value="evolution">Элита + мутация</option>
        <option value="cmaes">CMA-ES</option>
        <option value="de">Differential Evolution (DE)</option>
        <option value="pso">Particle Swarm Optimization (PSO)</option>
        <option value="cem">Cross-Entropy Method (CEM)</option>
      </select>
      <label for="terrainType">Terrain Lab</label>
      <select id="terrainType">
        <option value="flat">Плоский</option>
        <option value="ice">Лед</option>
        <option value="gravel">Гравий</option>
        <option value="slope">Уклон</option>
        <option value="steps">Ступеньки</option>
        <option value="rough">Шумовой рельеф</option>
      </select>
      <div class="row">
        <label for="terrainRandomEpisodes">Случайно на эпизоде</label>
        <input type="checkbox" id="terrainRandomEpisodes" />
      </div>
      <label for="robotBackend">Robot I/O backend</label>
      <select id="robotBackend">
        <option value="sim">sim backend</option>
        <option value="hardware-websocket">hardware backend (WebSocket)</option>
      </select>
      <label for="robotWsUrl">Hardware endpoint</label>
      <input type="text" id="robotWsUrl" value="ws://localhost:8765" />
      <label>Calibration UI</label>
      <div class="row">
        <select id="calibJoint"></select>
        <button type="button" id="identifyJoint">Identify</button>
      </div>
      <div class="row">
        <button type="button" id="calibStepZero">Step 1: Zero offset</button>
        <button type="button" id="calibStepDirection">Step 2: Direction</button>
      </div>
      <div class="row">
        <button type="button" id="calibStepLimits">Step 3: Limits</button>
        <button type="button" id="calibSaveYaml">Save calibrated YAML</button>
      </div>
      <canvas id="calibChart" width="300" height="84"></canvas>
      <div class="row">
        <button type="button" id="trainStart">Обучить политику</button>
      </div>
      <label>Политика (JSON)</label>
      <div class="row">
        <button type="button" id="policySave">Save JSON</button>
        <button type="button" id="policyLoad">Load JSON</button>
      </div>
      <canvas id="rewardChart" width="300" height="80"></canvas>
      <div class="muted" id="status"></div>
    </div>
  `

  const viewport = root.querySelector<HTMLDivElement>('#viewport')
  const statusEl = root.querySelector<HTMLDivElement>('#status')
  const rewardChart = root.querySelector<HTMLCanvasElement>('#rewardChart')
  const flexGainEl = root.querySelector<HTMLInputElement>('#flexGain')
  const flexGainValEl = root.querySelector<HTMLSpanElement>('#flexGainVal')
  const sphereVizEl = root.querySelector<HTMLInputElement>('#sphereViz')
  const sphereVizValEl = root.querySelector<HTMLSpanElement>('#sphereVizVal')
  const figureTypeEl = root.querySelector<HTMLSelectElement>('#figureType')
  const simStartBtn = root.querySelector<HTMLButtonElement>('#simStart')
  const simStopBtn = root.querySelector<HTMLButtonElement>('#simStop')
  const trainStartBtn = root.querySelector<HTMLButtonElement>('#trainStart')
  const calibJointEl = root.querySelector<HTMLSelectElement>('#calibJoint')
  const identifyJointBtn = root.querySelector<HTMLButtonElement>('#identifyJoint')
  const calibStepZeroBtn = root.querySelector<HTMLButtonElement>('#calibStepZero')
  const calibStepDirectionBtn = root.querySelector<HTMLButtonElement>('#calibStepDirection')
  const calibStepLimitsBtn = root.querySelector<HTMLButtonElement>('#calibStepLimits')
  const calibSaveYamlBtn = root.querySelector<HTMLButtonElement>('#calibSaveYaml')
  const calibChart = root.querySelector<HTMLCanvasElement>('#calibChart')
  const terrainTypeEl = root.querySelector<HTMLSelectElement>('#terrainType')
  const terrainRandomEpisodesEl = root.querySelector<HTMLInputElement>('#terrainRandomEpisodes')
  const robotBackendEl = root.querySelector<HTMLSelectElement>('#robotBackend')
  const robotWsUrlEl = root.querySelector<HTMLInputElement>('#robotWsUrl')
  const policySaveBtn = root.querySelector<HTMLButtonElement>('#policySave')
  const policyLoadBtn = root.querySelector<HTMLButtonElement>('#policyLoad')
  const trainGenerationsEl = root.querySelector<HTMLInputElement>('#trainGenerations')
  const trainAlgorithmEl = root.querySelector<HTMLSelectElement>('#trainAlgorithm')

  if (
    !viewport ||
    !statusEl ||
    !rewardChart ||
    !flexGainEl ||
    !flexGainValEl ||
    !sphereVizEl ||
    !sphereVizValEl ||
    !figureTypeEl ||
    !simStartBtn ||
    !simStopBtn ||
    !trainStartBtn ||
    !calibJointEl ||
    !identifyJointBtn ||
    !calibStepZeroBtn ||
    !calibStepDirectionBtn ||
    !calibStepLimitsBtn ||
    !calibSaveYamlBtn ||
    !calibChart ||
    !terrainTypeEl ||
    !terrainRandomEpisodesEl ||
    !robotBackendEl ||
    !robotWsUrlEl ||
    !policySaveBtn ||
    !policyLoadBtn ||
    !trainGenerationsEl ||
    !trainAlgorithmEl
  ) {
    throw new Error('HUD elements missing')
  }

  const readTrainingGenerations = (): number => {
    const raw = parseInt(trainGenerationsEl.value, 10)
    if (!Number.isFinite(raw)) return DEFAULT_TRAINING_GENERATIONS
    return Math.min(TRAINING_GENERATIONS_MAX, Math.max(TRAINING_GENERATIONS_MIN, Math.round(raw)))
  }

  let showActuatorSpheres = true
  let currentFigure: FigureId = 'table'
  const terrainProfiles = buildTerrainProfiles()
  let currentTerrain: TerrainId = 'flat'
  let randomTerrainPerEpisode = false

  let simTimeSeconds = 0
  let isPlaying = false
  let externalStepping = false
  let playbackStartX = 0
  let playbackBestX = 0

  const updateSimControlButtons = () => {
    simStartBtn.disabled = isPlaying
    simStopBtn.disabled = !isPlaying
  }

  simStartBtn.addEventListener('click', () => {
    if (tableRigidBody) {
      const com = comWorldFromRigidBody(tableRigidBody)
      playbackStartX = com.x
      playbackBestX = com.x
    }
    isPlaying = true
    updateSimControlButtons()
  })

  simStopBtn.addEventListener('click', () => {
    isPlaying = false
    updateSimControlButtons()
  })

  let elasticitySlider = DEFAULT_ELASTICITY_SLIDER

  const elasticityMultiplier = () => {
    const m = elasticitySlider / DEFAULT_ELASTICITY_SLIDER
    return Math.min(ELASTICITY_MULT_MAX, Math.max(ELASTICITY_MULT_MIN, m))
  }

  const syncElasticityFromSlider = () => {
    const k = elasticityMultiplier()
    actuatorSystem?.setDriveStiffness(BASE_ACTUATOR_SPRING_GAIN * k, BASE_ACTUATOR_MAX_FORCE * k)
    if (tableRigidBody) applyRigidBodyElasticityMaterial(tableRigidBody, k)
  }

  flexGainEl.addEventListener('input', () => {
    elasticitySlider = parseFloat(flexGainEl.value)
    flexGainValEl.textContent = elasticitySlider.toFixed(3)
    syncElasticityFromSlider()
  })

  const syncActuatorSphereVisibility = () => {
    actuatorSystem?.setMarkerSpheresVisible(showActuatorSpheres)
  }

  sphereVizEl.addEventListener('input', () => {
    showActuatorSpheres = sphereVizEl.value === '1'
    sphereVizValEl.textContent = showActuatorSpheres ? 'вкл' : 'выкл'
    syncActuatorSphereVisibility()
  })

  let ammo: any = null
  let ammoWorld: AmmoWorld | null = null
  let tableRigidBody: any = null
  let tableVisualMesh: THREE.Mesh | null = null
  let actuatorSystem: ActuatorSystemRigid | null = null
  const terrainVisualGroup = new THREE.Group()
  const terrainDecorById = new Map<TerrainId, THREE.Group>()
  const terrainBodiesById = new Map<TerrainId, TerrainBodyInstance[]>()
  let terrainSpawnYOffset = 0
  let robotIoLayer: RobotIoLayer | null = null
  let calibCmdHistory: number[] = []
  let calibMeasHistory: number[] = []
  let identificationActive = false
  let identificationJointIdx = 0
  let identificationStartTime = 0
  let calibrationYamlObject: {
    generatedAt: string
    robotBackend: RobotBackendId
    robotWsUrl: string
    joints: Array<{
      input: string
      output_name: string
      output_id: number
      sign: number
      scale: number
      position_offset: number
      limits: {
        position_min: number
        position_max: number
        velocity_abs_max: number
        effort_abs_max: number
      }
    }>
  } = {
    generatedAt: new Date().toISOString(),
    robotBackend: 'sim',
    robotWsUrl: 'ws://localhost:8765',
    joints: []
  }

  const initialTablePos = new THREE.Vector3()
  const initialTableQuat = new THREE.Quaternion()
  const resetPosePos = new THREE.Vector3()
  const initialScaleScratch = new THREE.Vector3()
  let tableResetPool: RigidBodyResetPool | null = null

  let meshBasePositions: Float32Array | null = null
  let legYMaxForFlex = 0.2
  let bboxMinYForFlex = -1
  let footOutwardLocal: THREE.Vector3[] = []

  let previewGenome: Genome | null = null
  let rewardHistory: number[] = []
  let spiderObjGeometryTemplate: THREE.BufferGeometry | null = null
  let spiderObjLastError = ''

  let scene: THREE.Scene
  let camera: THREE.PerspectiveCamera
  let renderer: THREE.WebGLRenderer
  let controls: any
  let baseFloorMesh: THREE.Mesh | null = null
  let baseGridHelper: THREE.GridHelper | null = null

  let meshGeometry: THREE.BufferGeometry | null = null
  let footPivotsLocal: THREE.Vector3[] = []
  let meshTransform = new THREE.Matrix4().makeTranslation(0, 0.55, 0)
  const policyFileInput = document.createElement('input')
  policyFileInput.type = 'file'
  policyFileInput.accept = 'application/json,.json'
  policyFileInput.style.display = 'none'
  root.appendChild(policyFileInput)

  const parseSpiderObjGeometry = async () => {
    spiderObjLastError = ''
    try {
      if (spiderObjRaw && spiderObjRaw.length > 0) {
        const obj = new OBJLoader().parse(spiderObjRaw)
        obj.updateMatrixWorld(true)
        const merged = mergedGeometryFromObject(obj)
        if (merged) {
          spiderObjGeometryTemplate?.dispose()
          spiderObjGeometryTemplate = merged
          return spiderObjGeometryTemplate
        }
      }
    } catch (e) {
      spiderObjLastError = `raw: ${(e as Error).message}`
    }

    try {
      const res = await fetch(spiderObjUrl)
      if (!res.ok) {
        spiderObjLastError = `url: HTTP ${res.status}`
        return null
      }
      const text = await res.text()
      const obj = new OBJLoader().parse(text)
      obj.updateMatrixWorld(true)
      const merged = mergedGeometryFromObject(obj)
      if (!merged) {
        spiderObjLastError = 'url: parsed empty geometry'
        return null
      }
      spiderObjGeometryTemplate?.dispose()
      spiderObjGeometryTemplate = merged
      return spiderObjGeometryTemplate
    } catch (e) {
      spiderObjLastError = `url: ${(e as Error).message}`
      return null
    }
  }

  const setStaticBodyPose = (body: any, pos: THREE.Vector3, rot: THREE.Quaternion) => {
    if (!ammoWorld) return
    const a = ammoWorld.ammo
    const tr = new a.btTransform()
    const origin = new a.btVector3(pos.x, pos.y, pos.z)
    const quat = new a.btQuaternion(rot.x, rot.y, rot.z, rot.w)
    tr.setIdentity()
    tr.setOrigin(origin)
    tr.setRotation(quat)
    body.setWorldTransform(tr)
    const ms = body.getMotionState?.()
    if (ms) ms.setWorldTransform(tr)
    body.activate?.()
    try {
      a.destroy(tr)
      a.destroy(origin)
      a.destroy(quat)
    } catch {
      // ignore
    }
  }

  const makeTerrainTexture = (
    base: string,
    dots: Array<{ color: string; count: number; radiusMin: number; radiusMax: number; alpha: number }>,
    strokes: Array<{ color: string; count: number; widthMin: number; widthMax: number; alpha: number }>
  ) => {
    const canvas = document.createElement('canvas')
    canvas.width = 256
    canvas.height = 256
    const ctx = canvas.getContext('2d')
    if (!ctx) return null
    ctx.fillStyle = base
    ctx.fillRect(0, 0, canvas.width, canvas.height)
    for (const d of dots) {
      ctx.fillStyle = d.color
      ctx.globalAlpha = d.alpha
      for (let i = 0; i < d.count; i++) {
        const x = Math.random() * canvas.width
        const y = Math.random() * canvas.height
        const r = d.radiusMin + Math.random() * (d.radiusMax - d.radiusMin)
        ctx.beginPath()
        ctx.arc(x, y, r, 0, Math.PI * 2)
        ctx.fill()
      }
    }
    for (const s of strokes) {
      ctx.strokeStyle = s.color
      ctx.globalAlpha = s.alpha
      for (let i = 0; i < s.count; i++) {
        const x = Math.random() * canvas.width
        const y = Math.random() * canvas.height
        const len = 12 + Math.random() * 44
        const a = Math.random() * Math.PI * 2
        const w = s.widthMin + Math.random() * (s.widthMax - s.widthMin)
        ctx.lineWidth = w
        ctx.beginPath()
        ctx.moveTo(x, y)
        ctx.lineTo(x + Math.cos(a) * len, y + Math.sin(a) * len)
        ctx.stroke()
      }
    }
    ctx.globalAlpha = 1
    const tex = new THREE.CanvasTexture(canvas)
    tex.wrapS = THREE.RepeatWrapping
    tex.wrapT = THREE.RepeatWrapping
    tex.colorSpace = THREE.SRGBColorSpace
    tex.anisotropy = 8
    return tex
  }

  const createTerrainDecorGroup = (id: TerrainId): THREE.Group => {
    const group = new THREE.Group()
    if (id === 'steps') {
      const profile = terrainProfiles.find((t) => t.id === 'steps')
      if (!profile) return group
      for (let i = 0; i < profile.specs.length; i++) {
        const spec = profile.specs[i]
        const geom = new THREE.BoxGeometry(spec.halfExtents.x * 2, spec.halfExtents.y * 2, spec.halfExtents.z * 2)
        const mat = new THREE.MeshStandardMaterial({
          color: i % 2 === 0 ? 0xbebebe : 0xa9a9a9,
          roughness: 0.88,
          metalness: 0.04
        })
        const m = new THREE.Mesh(geom, mat)
        m.position.copy(spec.position)
        m.quaternion.copy(spec.rotation ?? new THREE.Quaternion())
        group.add(m)
      }
      return group
    }

    const planeGeom = new THREE.PlaneGeometry(24, 24, id === 'rough' ? 80 : id === 'gravel' ? 56 : 20, id === 'rough' ? 80 : id === 'gravel' ? 56 : 20)
    planeGeom.rotateX(-Math.PI / 2)
    const posAttr = planeGeom.getAttribute('position') as THREE.BufferAttribute
    for (let i = 0; i < posAttr.count; i++) {
      const x = posAttr.getX(i)
      const z = posAttr.getZ(i)
      if (id === 'rough') {
        const y = Math.sin(x * 0.75) * 0.16 + Math.cos(z * 0.64) * 0.13 + Math.sin((x + z) * 1.4) * 0.05
        posAttr.setY(i, Math.max(-0.05, y))
      } else if (id === 'gravel') {
        const y = (Math.sin(x * 4.7 + z * 2.3) + Math.cos(x * 3.8 - z * 4.1)) * 0.012
        posAttr.setY(i, y)
      } else if (id === 'ice') {
        const y = Math.sin((x + z) * 0.5) * 0.003
        posAttr.setY(i, y)
      }
    }
    posAttr.needsUpdate = true
    planeGeom.computeVertexNormals()

    let mat: THREE.MeshStandardMaterial
    if (id === 'ice') {
      const iceTex = makeTerrainTexture(
        '#c7e6ff',
        [{ color: '#e8f7ff', count: 850, radiusMin: 0.5, radiusMax: 2.5, alpha: 0.22 }],
        [{ color: '#8fbad6', count: 85, widthMin: 0.5, widthMax: 1.4, alpha: 0.26 }]
      )
      if (iceTex) iceTex.repeat.set(6, 6)
      mat = new THREE.MeshStandardMaterial({
        color: 0xb7e3ff,
        map: iceTex ?? undefined,
        roughness: 0.08,
        metalness: 0.1
      })
    } else if (id === 'gravel') {
      const gravelTex = makeTerrainTexture(
        '#8b7f72',
        [
          { color: '#746b60', count: 1900, radiusMin: 0.8, radiusMax: 2.8, alpha: 0.35 },
          { color: '#a19484', count: 1200, radiusMin: 0.8, radiusMax: 2.5, alpha: 0.28 }
        ],
        [{ color: '#675e56', count: 90, widthMin: 0.4, widthMax: 1.2, alpha: 0.2 }]
      )
      if (gravelTex) gravelTex.repeat.set(11, 11)
      mat = new THREE.MeshStandardMaterial({
        color: 0x9a8d7d,
        map: gravelTex ?? undefined,
        roughness: 0.97,
        metalness: 0.0
      })
    } else if (id === 'slope') {
      const slopeTex = makeTerrainTexture(
        '#9f8a6e',
        [{ color: '#8a765f', count: 1500, radiusMin: 1.0, radiusMax: 2.8, alpha: 0.24 }],
        [{ color: '#7e6c57', count: 120, widthMin: 1.0, widthMax: 2.2, alpha: 0.22 }]
      )
      if (slopeTex) slopeTex.repeat.set(7, 7)
      mat = new THREE.MeshStandardMaterial({
        color: 0xa8906f,
        map: slopeTex ?? undefined,
        roughness: 0.9,
        metalness: 0.03
      })
      group.quaternion.setFromAxisAngle(new THREE.Vector3(0, 0, 1), THREE.MathUtils.degToRad(8))
    } else if (id === 'rough') {
      const roughTex = makeTerrainTexture(
        '#6f7d73',
        [
          { color: '#5f6e65', count: 1800, radiusMin: 1.0, radiusMax: 3.2, alpha: 0.3 },
          { color: '#8a988f', count: 1100, radiusMin: 0.8, radiusMax: 2.6, alpha: 0.24 }
        ],
        [{ color: '#4d5a53', count: 120, widthMin: 0.6, widthMax: 1.7, alpha: 0.2 }]
      )
      if (roughTex) roughTex.repeat.set(10, 10)
      mat = new THREE.MeshStandardMaterial({
        color: 0x718076,
        map: roughTex ?? undefined,
        roughness: 0.93,
        metalness: 0.02
      })
    } else {
      const flatTex = makeTerrainTexture(
        '#e9ecef',
        [{ color: '#d4d8dd', count: 1300, radiusMin: 0.8, radiusMax: 2.2, alpha: 0.2 }],
        [{ color: '#f6f7f9', count: 60, widthMin: 1.2, widthMax: 2.0, alpha: 0.25 }]
      )
      if (flatTex) flatTex.repeat.set(8, 8)
      mat = new THREE.MeshStandardMaterial({
        color: 0xe7eaee,
        map: flatTex ?? undefined,
        roughness: 0.88,
        metalness: 0.01
      })
    }

    const plane = new THREE.Mesh(planeGeom, mat)
    plane.position.y = 0.006
    group.add(plane)
    return group
  }

  const initTerrainBodies = () => {
    if (!ammoWorld || !scene) return
    terrainBodiesById.clear()
    terrainDecorById.clear()
    terrainVisualGroup.clear()
    scene.add(terrainVisualGroup)

    const hiddenBaseY = -220
    let hiddenIndex = 0
    for (const profile of terrainProfiles) {
      const items: TerrainBodyInstance[] = []
      for (const spec of profile.specs) {
        const activeRot = spec.rotation ? spec.rotation.clone() : new THREE.Quaternion()
        const hiddenPos = new THREE.Vector3(spec.position.x, hiddenBaseY - hiddenIndex * 4, spec.position.z)
        hiddenIndex++
        const handle = ammoWorld.createStaticBox(spec.halfExtents, hiddenPos, activeRot, {
          friction: profile.friction,
          restitution: profile.restitution
        })
        const geom = new THREE.BoxGeometry(spec.halfExtents.x * 2, spec.halfExtents.y * 2, spec.halfExtents.z * 2)
        const mat = new THREE.MeshStandardMaterial({
          color: spec.color,
          roughness: 0.95,
          metalness: 0.02
        })
        const mesh = new THREE.Mesh(geom, mat)
        mesh.visible = false
        terrainVisualGroup.add(mesh)
        items.push({
          handle,
          mesh,
          activePosition: spec.position.clone(),
          activeRotation: activeRot,
          hiddenPosition: hiddenPos
        })
      }
      terrainBodiesById.set(profile.id, items)
      const decor = createTerrainDecorGroup(profile.id)
      decor.visible = false
      terrainVisualGroup.add(decor)
      terrainDecorById.set(profile.id, decor)
    }
  }

  const applyTerrainById = (id: TerrainId) => {
    if (!ammoWorld) return
    currentTerrain = id
    const profile = terrainProfiles.find((t) => t.id === id) ?? terrainProfiles[0]
    terrainSpawnYOffset = profile.spawnYOffset
    for (const [terrainId, items] of terrainBodiesById.entries()) {
      const active = terrainId === id
      for (const item of items) {
        const targetPos = active ? item.activePosition : item.hiddenPosition
        setStaticBodyPose(item.handle.body, targetPos, item.activeRotation)
        item.mesh.visible = active && TERRAIN_PHYSICS_DEBUG_MESHES
        if (active) {
          item.mesh.position.copy(item.activePosition)
          item.mesh.quaternion.copy(item.activeRotation)
        }
      }
    }
    for (const [terrainId, decor] of terrainDecorById.entries()) {
      decor.visible = terrainId === id
    }
    if (baseFloorMesh) baseFloorMesh.visible = false
    if (baseGridHelper) baseGridHelper.visible = id === 'flat'
  }

  const chooseTerrainForEpisode = (): TerrainId => {
    if (!randomTerrainPerEpisode) return currentTerrain
    const idx = Math.floor(Math.random() * terrainProfiles.length)
    return terrainProfiles[idx]?.id ?? currentTerrain
  }

  const disposeCurrentPhysics = () => {
    robotIoLayer?.dispose()
    robotIoLayer = null
    actuatorSystem?.dispose()
    actuatorSystem = null
    if (tableVisualMesh) {
      tableVisualMesh.parent?.remove(tableVisualMesh)
      const mat = tableVisualMesh.material
      if (mat instanceof THREE.Material) mat.dispose()
      tableVisualMesh = null
    }
    tableRigidBody = null
    tableResetPool = null
    if (ammoWorld) {
      ammoWorld.dispose()
      ammoWorld = null
    }
    for (const items of terrainBodiesById.values()) {
      for (const item of items) {
        item.mesh.parent?.remove(item.mesh)
        item.mesh.geometry.dispose()
        const mat = item.mesh.material
        if (mat instanceof THREE.Material) mat.dispose()
      }
    }
    for (const decor of terrainDecorById.values()) {
      decor.traverse((obj) => {
        const mesh = obj as THREE.Mesh
        if (!mesh.isMesh) return
        const geom = mesh.geometry as THREE.BufferGeometry | undefined
        if (geom) geom.dispose()
        const mat = mesh.material
        if (Array.isArray(mat)) {
          for (const m of mat) m.dispose()
        } else if (mat instanceof THREE.Material) {
          const maybeMap = (mat as THREE.MeshStandardMaterial).map
          maybeMap?.dispose()
          mat.dispose()
        }
      })
      decor.parent?.remove(decor)
    }
    terrainBodiesById.clear()
    terrainDecorById.clear()
    terrainVisualGroup.clear()
    if (meshGeometry) {
      meshGeometry.dispose()
      meshGeometry = null
    }
  }

  const rebuildFigure = async (kind: FigureId) => {
    if (!ammo || !scene) return
    currentFigure = kind
    isPlaying = false
    externalStepping = false
    updateSimControlButtons()
    disposeCurrentPhysics()
    if (currentFigure === 'spider' && !spiderObjGeometryTemplate) {
      await parseSpiderObjGeometry()
    }
    meshGeometry =
      currentFigure === 'spider' && spiderObjGeometryTemplate
        ? spiderObjGeometryTemplate.clone()
        : createGeometryByFigure(currentFigure)
    meshGeometry.computeBoundingBox()
    const bbox = meshGeometry.boundingBox
    const lift = bbox ? Math.max(0.02, -bbox.min.y + 0.025) : 0.55
    meshTransform = new THREE.Matrix4().makeTranslation(0, lift, 0)
    footPivotsLocal = computeFootPivotsLocal(meshGeometry, actuatorCountForFigure(currentFigure))
    previewGenome = null
    initPhysicsWorldOnce()
  }

  const setStatus = (line: string) => {
    statusEl.textContent = line
  }

  const selectedTrainingAlgorithm = (): TrainingAlgorithmId => {
    const algoRaw = trainAlgorithmEl.value
    return algoRaw === 'cmaes'
      ? 'cmaes'
      : algoRaw === 'de'
        ? 'de'
        : algoRaw === 'pso'
          ? 'pso'
          : algoRaw === 'cem'
            ? 'cem'
            : 'evolution'
  }

  const loadPersistedGenomeForAlgorithm = (algo: TrainingAlgorithmId, actuatorCount: number): Genome | null => {
    if (actuatorCount <= 0) return null
    if (algo === 'cmaes') return new TrainerCmaEs().loadBestGenome(actuatorCount)
    if (algo === 'de') return new TrainerDe().loadBestGenome(actuatorCount)
    if (algo === 'pso') return new TrainerPso().loadBestGenome(actuatorCount)
    if (algo === 'cem') return new TrainerCem().loadBestGenome(actuatorCount)
    const g = new TrainerEvolution().loadBestGenome()
    return g && g.amplitudes.length === actuatorCount ? g : null
  }

  const isFiniteNumberArray = (value: unknown): value is number[] =>
    Array.isArray(value) && value.every((x) => typeof x === 'number' && Number.isFinite(x))

  const isFigureId = (value: unknown): value is FigureId =>
    value === 'table' ||
    value === 'stool' ||
    value === 'bench' ||
    value === 'spider' ||
    value === 'turtle' ||
    value === 'beetle' ||
    value === 'crab' ||
    value === 'rocket'

  const isTrainingAlgorithmId = (value: unknown): value is TrainingAlgorithmId =>
    value === 'evolution' || value === 'cmaes' || value === 'de' || value === 'pso' || value === 'cem'

  const isRobotBackendId = (value: unknown): value is RobotBackendId =>
    value === 'sim' || value === 'hardware-websocket'

  const isTerrainId = (value: unknown): value is TerrainId =>
    value === 'flat' || value === 'ice' || value === 'gravel' || value === 'slope' || value === 'steps' || value === 'rough'

  const sanitizeGenomeLengths = (genome: Genome, actuatorCount: number): Genome | null => {
    if (
      genome.amplitudes.length !== actuatorCount ||
      genome.omegas.length !== actuatorCount ||
      genome.phases.length !== actuatorCount
    ) {
      return null
    }
    const clamp = (x: number, min: number, max: number) => Math.max(min, Math.min(max, x))
    return {
      amplitudes: genome.amplitudes.map((x) => clamp(x, 0.0, 0.22)),
      omegas: genome.omegas.map((x) => clamp(x, 0.2, 8.5)),
      phases: genome.phases.slice()
    }
  }

  const parsePolicySnapshot = (raw: string): PolicySnapshot => {
    const payload = JSON.parse(raw) as Partial<PolicySnapshot>
    if (!payload || typeof payload !== 'object') throw new Error('JSON должен быть объектом')
    if (!isTrainingAlgorithmId(payload.algorithm)) throw new Error('Некорректный algorithm')
    if (!isFigureId(payload.figure)) throw new Error('Некорректный figure')
    if (typeof payload.trainingGenerations !== 'number' || !Number.isFinite(payload.trainingGenerations)) {
      throw new Error('Некорректный trainingGenerations')
    }
    if (typeof payload.elasticitySlider !== 'number' || !Number.isFinite(payload.elasticitySlider)) {
      throw new Error('Некорректный elasticitySlider')
    }
    if (typeof payload.showActuatorSpheres !== 'boolean') throw new Error('Некорректный showActuatorSpheres')
    const terrain = isTerrainId(payload.terrain) ? payload.terrain : 'flat'
    const randomTerrainPerEpisode =
      typeof payload.randomTerrainPerEpisode === 'boolean' ? payload.randomTerrainPerEpisode : false
    const robotBackend = isRobotBackendId(payload.robotBackend) ? payload.robotBackend : 'sim'
    const robotWsUrl = typeof payload.robotWsUrl === 'string' ? payload.robotWsUrl : 'ws://localhost:8765'
    const g = payload.genome
    if (!g || !isFiniteNumberArray(g.amplitudes) || !isFiniteNumberArray(g.omegas) || !isFiniteNumberArray(g.phases)) {
      throw new Error('Некорректный genome')
    }
    return {
      version: typeof payload.version === 'number' ? payload.version : 0,
      createdAt: typeof payload.createdAt === 'string' ? payload.createdAt : new Date().toISOString(),
      algorithm: payload.algorithm,
      figure: payload.figure,
      trainingGenerations: payload.trainingGenerations,
      elasticitySlider: payload.elasticitySlider,
      showActuatorSpheres: payload.showActuatorSpheres,
      terrain,
      randomTerrainPerEpisode,
      robotBackend,
      robotWsUrl,
      genome: {
        amplitudes: g.amplitudes.slice(),
        omegas: g.omegas.slice(),
        phases: g.phases.slice()
      }
    }
  }

  const buildPolicySnapshot = (): PolicySnapshot | null => {
    const actuatorCount = footPivotsLocal.length
    if (!previewGenome || actuatorCount <= 0) return null
    const sanitized = sanitizeGenomeLengths(previewGenome, actuatorCount)
    if (!sanitized) return null
    return {
      version: POLICY_FILE_VERSION,
      createdAt: new Date().toISOString(),
      algorithm: selectedTrainingAlgorithm(),
      figure: currentFigure,
      trainingGenerations: readTrainingGenerations(),
      elasticitySlider,
      showActuatorSpheres,
      terrain: currentTerrain,
      randomTerrainPerEpisode,
      robotBackend: (robotBackendEl.value as RobotBackendId) || 'sim',
      robotWsUrl: robotWsUrlEl.value.trim() || 'ws://localhost:8765',
      genome: sanitized
    }
  }

  const triggerPolicyDownload = (snapshot: PolicySnapshot) => {
    const body = `${JSON.stringify(snapshot, null, 2)}\n`
    const blob = new Blob([body], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const stamp = snapshot.createdAt.replace(/[:.]/g, '-')
    const fileName = `walking-policy-${snapshot.figure}-${snapshot.algorithm}-${stamp}.json`
    const a = document.createElement('a')
    a.href = url
    a.download = fileName
    document.body.appendChild(a)
    a.click()
    a.remove()
    URL.revokeObjectURL(url)
  }

  const drawRewardChart = () => {
    const ctx = rewardChart.getContext('2d')
    if (!ctx) return
    const w = rewardChart.width
    const h = rewardChart.height
    ctx.clearRect(0, 0, w, h)
    ctx.strokeStyle = 'rgba(255,255,255,0.1)'
    for (let i = 1; i <= 4; i++) {
      const y = (h * i) / 5
      ctx.beginPath()
      ctx.moveTo(0, y)
      ctx.lineTo(w, y)
      ctx.stroke()
    }
    if (rewardHistory.length < 2) {
      ctx.fillStyle = 'rgba(255,255,255,0.55)'
      ctx.font = '11px ui-monospace, Menlo, monospace'
      ctx.fillText('Лучший reward по поколениям', 8, 16)
      return
    }
    const min = Math.min(...rewardHistory)
    const max = Math.max(...rewardHistory)
    const pad = 0.12
    const range = Math.max(1e-6, max - min)
    const yMin = min - range * pad
    const yMax = max + range * pad
    const xFor = (i: number) => (i / (rewardHistory.length - 1)) * (w - 16) + 8
    const yFor = (v: number) => {
      const t = (v - yMin) / (yMax - yMin)
      return (1 - t) * (h - 18) + 10
    }
    ctx.strokeStyle = 'rgba(110,231,255,0.95)'
    ctx.lineWidth = 2
    ctx.beginPath()
    ctx.moveTo(xFor(0), yFor(rewardHistory[0]))
    for (let i = 1; i < rewardHistory.length; i++) ctx.lineTo(xFor(i), yFor(rewardHistory[i]))
    ctx.stroke()
    ctx.fillStyle = 'rgba(255,255,255,0.8)'
    ctx.font = '11px ui-monospace, Menlo, monospace'
    ctx.fillText(`best: ${rewardHistory[rewardHistory.length - 1].toFixed(3)}`, 8, h - 4)
  }

  const rebuildCalibrationJointSelect = () => {
    calibJointEl.innerHTML = ''
    const count = footPivotsLocal.length
    for (let i = 0; i < count; i++) {
      const o = document.createElement('option')
      o.value = String(i)
      o.textContent = `leg-${i}`
      calibJointEl.appendChild(o)
    }
    if (count > 0) calibJointEl.value = '0'
  }

  const pushCalibrationSample = (cmdPos: number, measPos: number | null) => {
    calibCmdHistory.push(cmdPos)
    calibMeasHistory.push(measPos ?? cmdPos)
    const maxN = 180
    if (calibCmdHistory.length > maxN) calibCmdHistory.splice(0, calibCmdHistory.length - maxN)
    if (calibMeasHistory.length > maxN) calibMeasHistory.splice(0, calibMeasHistory.length - maxN)
  }

  const drawCalibrationChart = () => {
    const ctx = calibChart.getContext('2d')
    if (!ctx) return
    const w = calibChart.width
    const h = calibChart.height
    ctx.clearRect(0, 0, w, h)
    ctx.fillStyle = 'rgba(20,20,20,0.03)'
    ctx.fillRect(0, 0, w, h)
    ctx.strokeStyle = 'rgba(100,100,100,0.18)'
    ctx.beginPath()
    ctx.moveTo(6, h * 0.5)
    ctx.lineTo(w - 6, h * 0.5)
    ctx.stroke()
    const n = Math.min(calibCmdHistory.length, calibMeasHistory.length)
    if (n < 2) {
      ctx.fillStyle = 'rgba(80,80,80,0.8)'
      ctx.font = '11px ui-monospace, Menlo, monospace'
      ctx.fillText('Calibration: command vs measured', 8, 16)
      return
    }
    const all = calibCmdHistory.concat(calibMeasHistory)
    const min = Math.min(...all)
    const max = Math.max(...all)
    const r = Math.max(1e-6, max - min)
    const x = (i: number) => 8 + (i / (n - 1)) * (w - 16)
    const y = (v: number) => (1 - (v - min) / r) * (h - 14) + 7
    ctx.lineWidth = 1.8
    ctx.strokeStyle = '#2d8cff'
    ctx.beginPath()
    ctx.moveTo(x(0), y(calibCmdHistory[0]!))
    for (let i = 1; i < n; i++) ctx.lineTo(x(i), y(calibCmdHistory[i]!))
    ctx.stroke()
    ctx.strokeStyle = '#f97316'
    ctx.beginPath()
    ctx.moveTo(x(0), y(calibMeasHistory[0]!))
    for (let i = 1; i < n; i++) ctx.lineTo(x(i), y(calibMeasHistory[i]!))
    ctx.stroke()
  }

  const measuredJointPosition = (idx: number): number | null => {
    const t = robotIoLayer?.getLatestTelemetry()
    const js = t?.jointState
    if (!js?.name || !js?.position) return null
    const desired = `leg-${idx}`
    const i = js.name.indexOf(desired)
    if (i >= 0 && i < js.position.length) return js.position[i] ?? null
    return idx < js.position.length ? js.position[idx] ?? null : null
  }

  const escapeYaml = (s: string) => s.replace(/\\/g, '\\\\').replace(/"/g, '\\"')

  const triggerCalibratedYamlDownload = () => {
    const lines: string[] = []
    lines.push(`generatedAt: "${escapeYaml(calibrationYamlObject.generatedAt)}"`)
    lines.push(`robotBackend: "${escapeYaml(calibrationYamlObject.robotBackend)}"`)
    lines.push(`robotWsUrl: "${escapeYaml(calibrationYamlObject.robotWsUrl)}"`)
    lines.push('joints:')
    for (const j of calibrationYamlObject.joints) {
      lines.push(`  - input: ${j.input}`)
      lines.push(`    output_name: ${j.output_name}`)
      lines.push(`    output_id: ${j.output_id}`)
      lines.push(`    sign: ${j.sign.toFixed(6)}`)
      lines.push(`    scale: ${j.scale.toFixed(6)}`)
      lines.push(`    position_offset: ${j.position_offset.toFixed(6)}`)
      lines.push('    limits:')
      lines.push(`      position_min: ${j.limits.position_min.toFixed(6)}`)
      lines.push(`      position_max: ${j.limits.position_max.toFixed(6)}`)
      lines.push(`      velocity_abs_max: ${j.limits.velocity_abs_max.toFixed(6)}`)
      lines.push(`      effort_abs_max: ${j.limits.effort_abs_max.toFixed(6)}`)
    }
    const body = `${lines.join('\n')}\n`
    const blob = new Blob([body], { type: 'text/yaml' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `joint_mapping.calibrated.${new Date().toISOString().replace(/[:.]/g, '-')}.yaml`
    document.body.appendChild(a)
    a.click()
    a.remove()
    URL.revokeObjectURL(url)
  }

  const yieldToUI = () => new Promise<void>((r) => setTimeout(r, 0))

  const applyVisualLegFlex = () => {
    if (!meshGeometry || !meshBasePositions || !previewGenome) return
    applyLegVertexFlex(
      meshGeometry,
      meshBasePositions,
      footPivotsLocal,
      footOutwardLocal,
      previewGenome,
      simTimeSeconds,
      legYMaxForFlex,
      bboxMinYForFlex,
      elasticitySlider
    )
  }

  const rebuildActuators = (): ActuatorSystemRigid | null => {
    if (!ammoWorld || !tableRigidBody) {
      actuatorSystem = null
      return null
    }
    actuatorSystem?.dispose()
    actuatorSystem = null
    if (footPivotsLocal.length === 0) return null
    const actuatorCount = Math.max(1, footPivotsLocal.length)
    const perFootScale = Math.min(1.2, Math.max(0.55, 4 / actuatorCount))
    const sys = new ActuatorSystemRigid(
      ammoWorld,
      true,
      GROUND_PLANE_Y,
      BASE_ACTUATOR_SPRING_GAIN * perFootScale,
      BASE_ACTUATOR_MAX_FORCE * perFootScale
    )
    sys.createForFeet(tableRigidBody, footPivotsLocal, { scene, tableSpawnMatrix: meshTransform })
    if (!previewGenome || previewGenome.amplitudes.length !== footPivotsLocal.length) {
      previewGenome =
        loadPersistedGenomeForAlgorithm(selectedTrainingAlgorithm(), footPivotsLocal.length) ??
        randomGenome(footPivotsLocal.length)
    }
    sys.setGenome(previewGenome)
    actuatorSystem = sys
    robotIoLayer?.dispose()
    robotIoLayer = new RobotIoLayer(sys)
    robotIoLayer.setBackend((robotBackendEl.value as RobotBackendId) || 'sim', robotWsUrlEl.value)
    rebuildCalibrationJointSelect()
    calibrationYamlObject.generatedAt = new Date().toISOString()
    calibrationYamlObject.robotBackend = (robotBackendEl.value as RobotBackendId) || 'sim'
    calibrationYamlObject.robotWsUrl = robotWsUrlEl.value.trim() || 'ws://localhost:8765'
    calibrationYamlObject.joints = Array.from({ length: footPivotsLocal.length }, (_, i) => ({
      input: `leg-${i}`,
      output_name: `joint_${i}`,
      output_id: i + 1,
      sign: 1,
      scale: 1,
      position_offset: 0,
      limits: {
        position_min: -1.2,
        position_max: 1.2,
        velocity_abs_max: 4.0,
        effort_abs_max: 18.0
      }
    }))
    syncElasticityFromSlider()
    syncActuatorSphereVisibility()
    return sys
  }

  const initPhysicsWorldOnce = () => {
    if (ammoWorld || !scene || !ammo || !meshGeometry) return

    meshTransform.decompose(initialTablePos, initialTableQuat, initialScaleScratch)

    meshGeometry.computeBoundingBox()
    const b = meshGeometry.boundingBox
    bboxMinYForFlex = b ? b.min.y : -1
    legYMaxForFlex = legRegionYMax(meshGeometry, 0.38)
    meshBasePositions = snapshotGeometryPositions(meshGeometry)
    footOutwardLocal = footOutwardDirsXZ(footPivotsLocal)

    ammoWorld = new AmmoWorld(ammo, { gravityY: -9.8, solverIterations: 44 })
    initTerrainBodies()
    applyTerrainById(currentTerrain)
    tableResetPool = createRigidBodyResetPool(ammoWorld.ammo)

    const flat = geometryPositionsFlat(meshGeometry)
    tableRigidBody = ammoWorld.createDynamicConvexHull(TABLE_RIGID_MASS, flat, meshTransform)

    tableVisualMesh = new THREE.Mesh(
      meshGeometry,
      new THREE.MeshStandardMaterial({
        color: 0xc9b8a8,
        metalness: 0.18,
        roughness: 0.58
      })
    )
    scene.add(tableVisualMesh)
    syncMeshFromRigidBody(tableVisualMesh, tableRigidBody)

    controls.target.set(0, 1, 0)

    rebuildActuators()
    applyVisualLegFlex()

    simTimeSeconds = 0
    setStatus(
      `Симуляция выключена — «Запустить». ${figureLabel(currentFigure)}, terrain: ${terrainLabel(currentTerrain)}, приводов: ${footPivotsLocal.length}.`
    )
    if (currentFigure === 'spider' && !spiderObjGeometryTemplate) {
      const why = spiderObjLastError ? ` (${spiderObjLastError})` : ''
      setStatus(`Паук: OBJ не загружен, используется fallback-геометрия${why}.`)
    } else if (currentFigure === 'spider') {
      setStatus(`Паук: загружен OBJ из бандла, приводов: ${footPivotsLocal.length}.`)
    }
    updateSimControlButtons()
  }

  const resetSimulationForEpisode = (genome?: Genome, randomizeTerrain = false) => {
    if (!ammoWorld || !tableRigidBody || !tableResetPool || !actuatorSystem) {
      throw new Error('Физика не инициализирована')
    }
    if (randomizeTerrain) {
      applyTerrainById(chooseTerrainForEpisode())
    }
    resetPosePos.copy(initialTablePos)
    resetPosePos.y += terrainSpawnYOffset
    resetRigidBodyToPose(tableRigidBody, resetPosePos, initialTableQuat, tableResetPool)
    if (genome) previewGenome = genome
    if (!previewGenome) previewGenome = randomGenome(footPivotsLocal.length)
    actuatorSystem.setGenome(previewGenome)
    if (previewGenome && robotIoLayer) {
      robotIoLayer.tick(previewGenome, 0, tableRigidBody)
    } else {
      actuatorSystem.applyAtTime(0)
    }
    if (tableVisualMesh) syncMeshFromRigidBody(tableVisualMesh, tableRigidBody)
    applyVisualLegFlex()
    simTimeSeconds = 0
    const com = comWorldFromRigidBody(tableRigidBody)
    playbackStartX = com.x
    playbackBestX = com.x
  }

  const initThree = () => {
    scene = new THREE.Scene()
    scene.background = new THREE.Color(0xf7f9fc)

    scene.add(new THREE.AmbientLight(0xffffff, 0.8))
    const dir = new THREE.DirectionalLight(0xffffff, 0.95)
    dir.position.set(-4, 9, 5)
    scene.add(dir)

    camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.05, 200)
    camera.position.set(-3.2, 3.4, 6.2)

    renderer = new THREE.WebGLRenderer({ antialias: true })
    renderer.setPixelRatio(window.devicePixelRatio)
    renderer.setSize(window.innerWidth, window.innerHeight)
    viewport.appendChild(renderer.domElement)

    controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.dampingFactor = 0.06

    const floorY = 0.002
    const floor = new THREE.Mesh(
      new THREE.PlaneGeometry(24, 24),
      new THREE.MeshStandardMaterial({
        color: 0xffffff,
        metalness: 0.0,
        roughness: 0.96
      })
    )
    floor.rotation.x = -Math.PI / 2
    floor.position.y = floorY
    scene.add(floor)
    baseFloorMesh = floor

    const grid = new THREE.GridHelper(24, 48, 0xcbd5e1, 0xe2e8f0)
    grid.position.y = 0.004
    scene.add(grid)
    baseGridHelper = grid

    window.addEventListener('resize', () => {
      camera.aspect = window.innerWidth / window.innerHeight
      camera.updateProjectionMatrix()
      renderer.setSize(window.innerWidth, window.innerHeight)
      renderer.setPixelRatio(window.devicePixelRatio)
    })
  }

  const animate = () => {
    requestAnimationFrame(animate)
    const dt = 1 / 60
    if (ammoWorld && tableRigidBody && tableVisualMesh && isPlaying && !externalStepping) {
      if (previewGenome && robotIoLayer) {
        robotIoLayer.tick(previewGenome, simTimeSeconds, tableRigidBody)
      } else {
        actuatorSystem?.applyAtTime(simTimeSeconds)
      }
      ammoWorld.step(dt, 32)
      syncMeshFromRigidBody(tableVisualMesh, tableRigidBody)
      applyVisualLegFlex()
      simTimeSeconds += dt
      const com = comWorldFromRigidBody(tableRigidBody)
      const progressX = com.x - playbackStartX
      playbackBestX = Math.max(playbackBestX, com.x)
      if (playbackBestX - com.x > PLAYBACK_BACKTRACK_THRESHOLD) {
        isPlaying = false
        updateSimControlButtons()
        setStatus(`Авто-стоп: обнаружен откат назад (progress=${progressX.toFixed(3)}).`)
      }
    }
    if (ammoWorld && tableRigidBody && identificationActive && !externalStepping) {
      const localT = simTimeSeconds - identificationStartTime
      const cmd: MotorCommand[] = Array.from({ length: footPivotsLocal.length }, (_, i) => ({
        jointId: `leg-${i}`,
        index: i,
        targetPosition: i === identificationJointIdx ? 0.18 * Math.sin(localT * 2.4) : 0,
        targetVelocity: i === identificationJointIdx ? 0.18 * 2.4 * Math.cos(localT * 2.4) : 0,
        targetEffort: i === identificationJointIdx ? 8.0 * Math.sin(localT * 2.4) : 0
      }))
      robotIoLayer?.sendManualCommands(cmd, tableRigidBody, simTimeSeconds)
      ammoWorld.step(dt, 32)
      if (tableVisualMesh) syncMeshFromRigidBody(tableVisualMesh, tableRigidBody)
      applyVisualLegFlex()
      simTimeSeconds += dt
      const meas = measuredJointPosition(identificationJointIdx)
      pushCalibrationSample(cmd[identificationJointIdx]!.targetPosition, meas)
    }
    drawCalibrationChart()
    controls.update()
    renderer.render(scene, camera)
  }

  const stepPreviewOnce = (dtSeconds: number) => {
    if (!ammoWorld || !tableRigidBody || !tableVisualMesh) return
    if (previewGenome && robotIoLayer) {
      robotIoLayer.tick(previewGenome, simTimeSeconds, tableRigidBody)
    } else {
      actuatorSystem?.applyAtTime(simTimeSeconds)
    }
    ammoWorld.step(dtSeconds, 32)
    syncMeshFromRigidBody(tableVisualMesh, tableRigidBody)
    applyVisualLegFlex()
    simTimeSeconds += dtSeconds
  }

  const runVisibleEpisode = async (genome: Genome, steps: number, dt: number, speed: number) => {
    if (!ammo || !meshGeometry) throw new Error('No ammo/meshGeometry')
    externalStepping = true
    isPlaying = false
    updateSimControlButtons()
    resetSimulationForEpisode(genome, true)
    if (!tableRigidBody) throw new Error('Missing rigid table')

    const reward = new RewardCalculator({
      fallMinY: 0.12,
      fallPenalty: 5,
      comAxis: 'x',
      lateralPenaltyWeight: TRAINING_LATERAL_PENALTY_WEIGHT,
      backwardPenaltyWeight: TRAINING_BACKWARD_PENALTY_WEIGHT,
      backtrackPenaltyWeight: TRAINING_BACKTRACK_PENALTY_WEIGHT
    })
    reward.startRigid(tableRigidBody)

    const strideObs = 10
    const k = Math.max(1, Math.floor(speed))
    for (let s = 0; s < steps; s++) {
      for (let j = 0; j < k; j++) stepPreviewOnce(dt)
      if (s % strideObs === 0) reward.observeRigidStep(tableRigidBody)
      await new Promise<void>((r) => requestAnimationFrame(() => r()))
    }
    const ep = reward.finishRigid(tableRigidBody)
    externalStepping = false
    return ep.reward
  }

  const runTraining = async () => {
    if (!ammo || !meshGeometry) return
    const nAct = footPivotsLocal.length
    if (nAct === 0) {
      setStatus('Не удалось определить ножки')
      isPlaying = false
      updateSimControlButtons()
      return
    }

    isPlaying = false
    updateSimControlButtons()
    rewardHistory = []
    drawRewardChart()
    setStatus('Обучение…')

    const populationSize = TRAINING_POPULATION
    const generations = readTrainingGenerations()
    trainGenerationsEl.value = String(generations)
    const maxSteps = EPISODE_STEPS
    const dt = DT_SECONDS
    const algo = selectedTrainingAlgorithm()
    const batchSize = populationSize

    const median = (values: number[]): number => {
      const sorted = values.slice().sort((a, b) => a - b)
      const mid = Math.floor(sorted.length / 2)
      return sorted.length % 2 === 1 ? sorted[mid] : 0.5 * (sorted[mid - 1] + sorted[mid])
    }

    const fidelityForGeneration = (generationIndex: number) => {
      const clampedIndex = Math.max(0, generationIndex)
      const t = generations <= 1 ? 1 : Math.min(1, clampedIndex / Math.max(1, generations - 1))
      const rolloutCount = Math.max(1, 1 + Math.floor(t * (EVAL_ROLLOUTS - 1)))
      const stepRatio = EVAL_MIN_STEP_RATIO + (1 - EVAL_MIN_STEP_RATIO) * (t * t)
      const steps = Math.max(40, Math.round(maxSteps * stepRatio))
      const minProgress = EARLY_STOP_MIN_PROGRESS * (0.85 + 0.55 * t)
      return { rolloutCount, steps, minProgress }
    }

    const runFastEpisode = async (
      genome: Genome,
      rolloutIndex: number,
      rolloutCount: number,
      evalSteps: number,
      minWindowProgress: number
    ): Promise<number> => {
      resetSimulationForEpisode(genome, true)
      // Multi-seed surrogate: vary initial phase/time offset for robustness.
      simTimeSeconds = (rolloutIndex / Math.max(1, rolloutCount)) * 0.45

      const reward = new RewardCalculator({
        fallMinY: EARLY_STOP_FALL_MIN_Y,
        fallPenalty: 5,
        comAxis: 'x',
        lateralPenaltyWeight: TRAINING_LATERAL_PENALTY_WEIGHT,
        backwardPenaltyWeight: TRAINING_BACKWARD_PENALTY_WEIGHT,
        backtrackPenaltyWeight: TRAINING_BACKTRACK_PENALTY_WEIGHT
      })
      reward.startRigid(tableRigidBody!)
      const comStart = comWorldFromRigidBody(tableRigidBody!)
      let windowStartStep = EARLY_STOP_GRACE_STEPS
      let progressAtWindowStart = 0
      let bestProgress = 0

      for (let step = 0; step < evalSteps; step++) {
        if (previewGenome && robotIoLayer) {
          robotIoLayer.tick(previewGenome, simTimeSeconds, tableRigidBody!)
        } else {
          actuatorSystem?.applyAtTime(simTimeSeconds)
        }
        ammoWorld!.step(dt, 32)
        reward.observeRigidStep(tableRigidBody!)
        simTimeSeconds += dt

        const comNow = comWorldFromRigidBody(tableRigidBody!)
        const progressNow = comNow.x - comStart.x
        bestProgress = Math.max(bestProgress, progressNow)
        if (step === EARLY_STOP_GRACE_STEPS) {
          progressAtWindowStart = progressNow
        }

        const shouldCheckWindow = step > EARLY_STOP_GRACE_STEPS && step - windowStartStep >= EARLY_STOP_WINDOW_STEPS
        if (shouldCheckWindow) {
          const windowProgress = progressNow - progressAtWindowStart
          if (windowProgress < minWindowProgress) break
          windowStartStep = step
          progressAtWindowStart = progressNow
        }

        if (step > 16 && comNow.y < EARLY_STOP_FALL_MIN_Y) break
        if (step > EARLY_STOP_GRACE_STEPS && bestProgress - progressNow > EARLY_STOP_BACKTRACK_THRESHOLD) break
        if (step % 80 === 0) await yieldToUI()
      }
      return reward.finishRigid(tableRigidBody!).reward
    }

    const evaluateOne = async (
      genome: Genome,
      individualIndex: number,
      generationIndex: number,
      stage?: EvaluationStage
    ) => {
      const fidelity = fidelityForGeneration(generationIndex)
      const rolloutCount = stage?.rolloutCount ?? fidelity.rolloutCount
      const steps = Math.max(40, Math.round(maxSteps * (stage?.stepRatio ?? 1) * (fidelity.steps / maxSteps)))
      const minProgress = fidelity.minProgress * (stage?.minProgressScale ?? 1)
      const stageTag = stage ? `, ${stage.label}` : ''
      setStatus(
        `Поколение ${generationIndex + 1}/${generations}, особь ${individualIndex + 1}/${batchSize} ` +
          `(rollouts ${rolloutCount}, steps ${steps}${stageTag})`
      )
      const rewards: number[] = []
      for (let rollout = 0; rollout < rolloutCount; rollout++) {
        rewards.push(
          await runFastEpisode(genome, rollout, rolloutCount, steps, minProgress)
        )
      }
      return median(rewards)
    }

    try {
      const afterGeneration = async (summary: {
        generationIndex: number
        generationCount: number
        bestRewardSoFar: number
        bestGenomeThisGen: Genome
      }) => {
        rewardHistory.push(summary.bestRewardSoFar)
        drawRewardChart()
        previewGenome = summary.bestGenomeThisGen
        setStatus(`Поколение ${summary.generationIndex + 1}/${summary.generationCount}: показ элитной особи`)
        await runVisibleEpisode(summary.bestGenomeThisGen, maxSteps, dt, VIZ_SPEED)
        resetSimulationForEpisode(summary.bestGenomeThisGen)
      }

      let best: { bestGenome: Genome; bestReward: number }
      if (algo === 'cmaes') {
        const trainer = new TrainerCmaEs({ generations, lambda: populationSize })
        best = await trainer.run(nAct, evaluateOne, () => {}, async (summary) => {
          await afterGeneration(summary)
        })
      } else if (algo === 'de') {
        const trainer = new TrainerDe({
          generations,
          populationSize
        })
        best = await trainer.run(nAct, evaluateOne, () => {}, async (summary) => {
          await afterGeneration(summary)
        })
      } else if (algo === 'pso') {
        const trainer = new TrainerPso({
          generations,
          populationSize
        })
        best = await trainer.run(nAct, evaluateOne, () => {}, async (summary) => {
          await afterGeneration(summary)
        })
      } else if (algo === 'cem') {
        const trainer = new TrainerCem({
          generations,
          populationSize
        })
        best = await trainer.run(nAct, evaluateOne, () => {}, async (summary) => {
          await afterGeneration(summary)
        })
      } else {
        const trainer = new TrainerEvolution({
          populationSize,
          eliteCount: Math.max(2, Math.floor(populationSize * 0.2)),
          generations
        })
        best = await trainer.run(nAct, evaluateOne, () => {}, async (summary) => {
          await afterGeneration(summary)
        })
      }

      previewGenome = best.bestGenome
      resetSimulationForEpisode(previewGenome)
      isPlaying = false
      updateSimControlButtons()
      const algoLabel =
        algo === 'cmaes' ? 'CMA-ES' : algo === 'de' ? 'DE' : algo === 'pso' ? 'PSO' : algo === 'cem' ? 'CEM' : 'элита + мутация'
      setStatus(
        `Обучение готово (${algoLabel}, reward ≈ ${best.bestReward.toFixed(3)}). Нажмите «Запустить», чтобы проиграть политику.`
      )
    } catch (e) {
      setStatus(`Ошибка обучения: ${(e as Error).message}`)
      isPlaying = false
      updateSimControlButtons()
    }
  }

  const start = async () => {
    initThree()
    setStatus('Загрузка Ammo.js…')

    const ammoGlobal: any = (globalThis as any).Ammo
    if (typeof ammoGlobal !== 'function') {
      throw new Error('Ammo global is not a function (ammo.js not loaded?)')
    }
    ammo = await ammoGlobal()
    currentFigure = (figureTypeEl.value as FigureId) || 'table'
    currentTerrain = (terrainTypeEl.value as TerrainId) || 'flat'
    randomTerrainPerEpisode = terrainRandomEpisodesEl.checked
    await rebuildFigure(currentFigure)
    drawRewardChart()
    animate()

    figureTypeEl.addEventListener('change', async () => {
      const next = (figureTypeEl.value as FigureId) || 'table'
      await rebuildFigure(next)
    })

    terrainTypeEl.addEventListener('change', () => {
      const next = (terrainTypeEl.value as TerrainId) || 'flat'
      currentTerrain = next
      applyTerrainById(next)
      if (ammoWorld && tableRigidBody && tableResetPool && actuatorSystem && previewGenome) {
        resetSimulationForEpisode(previewGenome)
      }
      setStatus(`Terrain Lab: ${terrainLabel(currentTerrain)}.`)
    })

    terrainRandomEpisodesEl.addEventListener('change', () => {
      randomTerrainPerEpisode = terrainRandomEpisodesEl.checked
      setStatus(
        randomTerrainPerEpisode
          ? 'Terrain Lab: случайный выбор покрытия на каждом эпизоде включен.'
          : `Terrain Lab: фиксированное покрытие (${terrainLabel(currentTerrain)}).`
      )
    })

    robotBackendEl.addEventListener('change', () => {
      const backend = (robotBackendEl.value as RobotBackendId) || 'sim'
      const endpoint = robotWsUrlEl.value.trim() || 'ws://localhost:8765'
      robotIoLayer?.setBackend(backend, endpoint)
      setStatus(
        backend === 'sim'
          ? 'Robot I/O: sim backend активен.'
          : `Robot I/O: hardware backend (WebSocket) активен, endpoint=${endpoint}.`
      )
    })

    robotWsUrlEl.addEventListener('change', () => {
      const backend = (robotBackendEl.value as RobotBackendId) || 'sim'
      const endpoint = robotWsUrlEl.value.trim() || 'ws://localhost:8765'
      if (backend === 'hardware-websocket') {
        robotIoLayer?.setBackend(backend, endpoint)
        setStatus(`Robot I/O: подключение hardware backend к ${endpoint}.`)
      }
    })

    identifyJointBtn.addEventListener('click', () => {
      if (!tableRigidBody || !robotIoLayer) {
        setStatus('Identification недоступен: физика/Robot I/O не готовы.')
        return
      }
      const idx = Number.parseInt(calibJointEl.value || '0', 10)
      identificationJointIdx = Number.isFinite(idx) ? Math.max(0, Math.min(footPivotsLocal.length - 1, idx)) : 0
      identificationStartTime = simTimeSeconds
      identificationActive = !identificationActive
      if (identificationActive) {
        isPlaying = false
        updateSimControlButtons()
      }
      setStatus(
        identificationActive
          ? `Identification: auto-режим leg-${identificationJointIdx} активен.`
          : 'Identification остановлен.'
      )
    })

    calibStepZeroBtn.addEventListener('click', () => {
      const idx = Number.parseInt(calibJointEl.value || '0', 10)
      const samples = calibMeasHistory.slice(-50)
      const mean = samples.length ? samples.reduce((a, b) => a + b, 0) / samples.length : 0
      if (idx >= 0 && idx < calibrationYamlObject.joints.length) {
        calibrationYamlObject.joints[idx]!.position_offset = -mean
      }
      setStatus(`Calibration step 1 (zero): leg-${idx}, offset=${(-mean).toFixed(4)}.`)
    })

    calibStepDirectionBtn.addEventListener('click', () => {
      const idx = Number.parseInt(calibJointEl.value || '0', 10)
      let corr = 0
      for (let i = 1; i < Math.min(calibCmdHistory.length, calibMeasHistory.length); i++) {
        corr += (calibCmdHistory[i]! - calibCmdHistory[i - 1]!) * (calibMeasHistory[i]! - calibMeasHistory[i - 1]!)
      }
      const sign = corr >= 0 ? 1 : -1
      if (idx >= 0 && idx < calibrationYamlObject.joints.length) {
        calibrationYamlObject.joints[idx]!.sign = sign
      }
      setStatus(`Calibration step 2 (direction): leg-${idx}, sign=${sign}.`)
    })

    calibStepLimitsBtn.addEventListener('click', () => {
      const idx = Number.parseInt(calibJointEl.value || '0', 10)
      const samples = calibMeasHistory.slice(-120)
      const min = samples.length ? Math.min(...samples) : -1
      const max = samples.length ? Math.max(...samples) : 1
      if (idx >= 0 && idx < calibrationYamlObject.joints.length) {
        calibrationYamlObject.joints[idx]!.limits.position_min = min - 0.04
        calibrationYamlObject.joints[idx]!.limits.position_max = max + 0.04
      }
      setStatus(`Calibration step 3 (limits): leg-${idx}, [${(min - 0.04).toFixed(3)}, ${(max + 0.04).toFixed(3)}].`)
    })

    calibSaveYamlBtn.addEventListener('click', () => {
      calibrationYamlObject.generatedAt = new Date().toISOString()
      calibrationYamlObject.robotBackend = (robotBackendEl.value as RobotBackendId) || 'sim'
      calibrationYamlObject.robotWsUrl = robotWsUrlEl.value.trim() || 'ws://localhost:8765'
      triggerCalibratedYamlDownload()
      setStatus('Calibration YAML сохранен.')
    })

    trainAlgorithmEl.addEventListener('change', () => {
      const persisted = loadPersistedGenomeForAlgorithm(selectedTrainingAlgorithm(), footPivotsLocal.length)
      if (!persisted) return
      previewGenome = persisted
      if (ammoWorld && tableRigidBody && tableResetPool && actuatorSystem) {
        resetSimulationForEpisode(previewGenome)
      }
    })

    trainStartBtn.addEventListener('click', async () => {
      if (!ammo || !meshGeometry) return
      identificationActive = false
      trainStartBtn.disabled = true
      trainGenerationsEl.disabled = true
      trainAlgorithmEl.disabled = true
      figureTypeEl.disabled = true
      terrainTypeEl.disabled = true
      terrainRandomEpisodesEl.disabled = true
      robotBackendEl.disabled = true
      robotWsUrlEl.disabled = true
      try {
        await runTraining()
      } finally {
        figureTypeEl.disabled = false
        trainGenerationsEl.disabled = false
        trainAlgorithmEl.disabled = false
        trainStartBtn.disabled = false
        terrainTypeEl.disabled = false
        terrainRandomEpisodesEl.disabled = false
        robotBackendEl.disabled = false
        robotWsUrlEl.disabled = false
      }
    })

    policySaveBtn.addEventListener('click', () => {
      const snapshot = buildPolicySnapshot()
      if (!snapshot) {
        setStatus('Не удалось сохранить: политика еще не инициализирована.')
        return
      }
      triggerPolicyDownload(snapshot)
      setStatus(
        `Политика сохранена в JSON (${figureLabel(snapshot.figure)}, ${snapshot.algorithm}, приводов: ${snapshot.genome.amplitudes.length}).`
      )
    })

    policyLoadBtn.addEventListener('click', () => {
      policyFileInput.value = ''
      policyFileInput.click()
    })

    policyFileInput.addEventListener('change', async () => {
      const file = policyFileInput.files?.[0]
      if (!file) return
      try {
        const text = await file.text()
        const snapshot = parsePolicySnapshot(text)
        if (snapshot.version > POLICY_FILE_VERSION) {
          throw new Error(`Версия policy v${snapshot.version} не поддерживается этой сборкой`)
        }

        if (figureTypeEl.value !== snapshot.figure) {
          figureTypeEl.value = snapshot.figure
          await rebuildFigure(snapshot.figure)
        } else if (!ammoWorld || !tableRigidBody || !actuatorSystem) {
          await rebuildFigure(snapshot.figure)
        }

        const expectedActuators = footPivotsLocal.length
        const genome = sanitizeGenomeLengths(snapshot.genome, expectedActuators)
        if (!genome) {
          throw new Error(
            `Genome из файла не подходит: в файле ${snapshot.genome.amplitudes.length} приводов, у фигуры ${expectedActuators}`
          )
        }

        trainAlgorithmEl.value = snapshot.algorithm
        trainGenerationsEl.value = String(
          Math.min(TRAINING_GENERATIONS_MAX, Math.max(TRAINING_GENERATIONS_MIN, Math.round(snapshot.trainingGenerations)))
        )

        const flexMax = Number.parseFloat(flexGainEl.max)
        const flexMin = Number.parseFloat(flexGainEl.min)
        elasticitySlider = Math.min(flexMax, Math.max(flexMin, snapshot.elasticitySlider))
        flexGainEl.value = elasticitySlider.toFixed(3)
        flexGainValEl.textContent = elasticitySlider.toFixed(3)

        showActuatorSpheres = snapshot.showActuatorSpheres
        sphereVizEl.value = showActuatorSpheres ? '1' : '0'
        sphereVizValEl.textContent = showActuatorSpheres ? 'вкл' : 'выкл'

        currentTerrain = snapshot.terrain
        randomTerrainPerEpisode = snapshot.randomTerrainPerEpisode
        terrainTypeEl.value = currentTerrain
        terrainRandomEpisodesEl.checked = randomTerrainPerEpisode
        applyTerrainById(currentTerrain)

        robotBackendEl.value = snapshot.robotBackend
        robotWsUrlEl.value = snapshot.robotWsUrl
        robotIoLayer?.setBackend(snapshot.robotBackend, snapshot.robotWsUrl)
        calibrationYamlObject.robotBackend = snapshot.robotBackend
        calibrationYamlObject.robotWsUrl = snapshot.robotWsUrl

        previewGenome = genome
        if (ammoWorld && tableRigidBody && tableResetPool && actuatorSystem) {
          resetSimulationForEpisode(previewGenome)
        }
        syncElasticityFromSlider()
        syncActuatorSphereVisibility()
        setStatus(
          `Политика загружена из JSON (${figureLabel(snapshot.figure)}, ${snapshot.algorithm}, приводов: ${genome.amplitudes.length}).`
        )
      } catch (e) {
        setStatus(`Ошибка загрузки policy JSON: ${(e as Error).message}`)
      }
    })
  }

  void start()
}
