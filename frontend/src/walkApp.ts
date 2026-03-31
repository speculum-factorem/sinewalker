import * as THREE from 'three'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js'
import { mergeGeometries, mergeVertices } from 'three/addons/utils/BufferGeometryUtils.js'
import spiderObjRaw from './assets/models/spider3dmodel/TRANTULA/TRANTULA.OBJ?raw'
import spiderObjUrl from './assets/models/spider3dmodel/TRANTULA/TRANTULA.OBJ?url'

import { AmmoWorld } from './physics/AmmoWorld'
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
const GROUND_PLANE_Y = 0
/** Ползунок и визуальный изгиб, и физика приводов; это значение = множитель 1.0 к базовой жёсткости. */
const DEFAULT_ELASTICITY_SLIDER = 0.086
const BASE_ACTUATOR_SPRING_GAIN = 38
const BASE_ACTUATOR_MAX_FORCE = 46
const ELASTICITY_MULT_MIN = 0.25
const ELASTICITY_MULT_MAX = 2.2

const TABLE_RIGID_MASS = 14
type FigureId = 'table' | 'stool' | 'bench' | 'spider' | 'turtle' | 'beetle' | 'crab' | 'rocket'

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
      <div class="row">
        <button type="button" id="trainStart">Обучить политику</button>
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

  const initialTablePos = new THREE.Vector3()
  const initialTableQuat = new THREE.Quaternion()
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

  let meshGeometry: THREE.BufferGeometry | null = null
  let footPivotsLocal: THREE.Vector3[] = []
  let meshTransform = new THREE.Matrix4().makeTranslation(0, 0.55, 0)

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

  const disposeCurrentPhysics = () => {
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
    ammoWorld.createGround(20, 20, 0)
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
      `Симуляция выключена — «Запустить». ${figureLabel(currentFigure)}, приводов: ${footPivotsLocal.length}.`
    )
    if (currentFigure === 'spider' && !spiderObjGeometryTemplate) {
      const why = spiderObjLastError ? ` (${spiderObjLastError})` : ''
      setStatus(`Паук: OBJ не загружен, используется fallback-геометрия${why}.`)
    } else if (currentFigure === 'spider') {
      setStatus(`Паук: загружен OBJ из бандла, приводов: ${footPivotsLocal.length}.`)
    }
    updateSimControlButtons()
  }

  const resetSimulationForEpisode = (genome?: Genome) => {
    if (!ammoWorld || !tableRigidBody || !tableResetPool || !actuatorSystem) {
      throw new Error('Физика не инициализирована')
    }
    resetRigidBodyToPose(tableRigidBody, initialTablePos, initialTableQuat, tableResetPool)
    if (genome) previewGenome = genome
    if (!previewGenome) previewGenome = randomGenome(footPivotsLocal.length)
    actuatorSystem.setGenome(previewGenome)
    actuatorSystem.applyAtTime(0)
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

    const grid = new THREE.GridHelper(24, 48, 0xcbd5e1, 0xe2e8f0)
    grid.position.y = 0.004
    scene.add(grid)

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
      actuatorSystem?.applyAtTime(simTimeSeconds)
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
    controls.update()
    renderer.render(scene, camera)
  }

  const stepPreviewOnce = (dtSeconds: number) => {
    if (!ammoWorld || !tableRigidBody || !tableVisualMesh) return
    actuatorSystem?.applyAtTime(simTimeSeconds)
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
    resetSimulationForEpisode(genome)
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
      resetSimulationForEpisode(genome)
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
        actuatorSystem?.applyAtTime(simTimeSeconds)
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
    await rebuildFigure(currentFigure)
    drawRewardChart()
    animate()

    figureTypeEl.addEventListener('change', async () => {
      const next = (figureTypeEl.value as FigureId) || 'table'
      await rebuildFigure(next)
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
      trainStartBtn.disabled = true
      trainGenerationsEl.disabled = true
      trainAlgorithmEl.disabled = true
      figureTypeEl.disabled = true
      try {
        await runTraining()
      } finally {
        figureTypeEl.disabled = false
        trainGenerationsEl.disabled = false
        trainAlgorithmEl.disabled = false
        trainStartBtn.disabled = false
      }
    })
  }

  void start()
}
