import * as THREE from 'three'
import { AmmoWorld } from './AmmoWorld'

export interface SoftBodyFromMeshOptions {
  mass?: number
  viterations?: number
  piterations?: number
  margin?: number
  collisionMarginFactor?: number
  generateGraphics?: boolean
  /** Linear stiffness (higher = more rigid, still slightly elastic). */
  linearStiffness?: number
  /** Angular stiffness. */
  angularStiffness?: number
  /** Для замкнутого объёма; на открытой оболочке (стол) не задавать — иначе нестабильность. */
  volumeStiffness?: number
  /** Линейное затухание узлов (kDP); выше — меньше «разлет». */
  nodeDamping?: number
  /** Жёсткая «память формы» (pose matching), 0..1. */
  poseMatching?: number
  /** Бендинг-ограничения между удаленными вершинами. */
  bendingDistance?: number
}

export interface SoftBodyHandle {
  softBody: any
  mesh: THREE.Mesh | null
  geometry: THREE.BufferGeometry
  nodeCount: number
  vertexNormals: Float32Array
  dispose: () => void
}

function toFlatArray(typed: ArrayLike<number>): number[] {
  // Ammo.js bindings accept plain `number[]` for CreateFromTriMesh.
  // We keep this conversion explicit to avoid surprises with TypedArrays.
  return Array.from(typed)
}

export function createSoftBodyFromMesh(
  world: AmmoWorld,
  inputGeometry: THREE.BufferGeometry,
  transform: THREE.Matrix4,
  options: SoftBodyFromMeshOptions = {}
): SoftBodyHandle {
  const {
    mass = 1.0,
    viterations = 10,
    piterations = 10,
    generateGraphics = true,
    linearStiffness = 0.9,
    angularStiffness = 0.82,
    volumeStiffness,
    nodeDamping = 0.05,
    poseMatching = 0.22,
    bendingDistance = 2
  } = options

  // Clone and transform so that physics coordinates == render coordinates.
  const geometry = inputGeometry.clone()
  geometry.applyMatrix4(transform)
  geometry.computeVertexNormals()

  const positionAttr = geometry.getAttribute('position')
  const normalAttr = geometry.getAttribute('normal')
  if (!positionAttr) throw new Error('Geometry has no position attribute')
  if (!normalAttr) throw new Error('Geometry has no normal attribute after computeVertexNormals()')

  const positions = positionAttr.array as Float32Array
  const normals = normalAttr.array as Float32Array

  const vertices = toFlatArray(positions)

  const indexAttr = geometry.index
  const triangles = indexAttr
    ? toFlatArray(indexAttr.array as ArrayLike<number>)
    : (() => {
        const vertCount = positions.length / 3
        const triVertCount = Math.floor(vertCount / 3) * 3
        return toFlatArray(Array.from({ length: triVertCount }, (_, i) => i))
      })()

  const ntriangles = Math.floor(triangles.length / 3)
  if (ntriangles <= 0) throw new Error('No triangles found in geometry')

  const softBodyHelpers = new world.ammo.btSoftBodyHelpers()
  const softBody = softBodyHelpers.CreateFromTriMesh(
    world.softBodyWorldInfo,
    vertices,
    triangles,
    ntriangles,
    false
  )

  const cfg = softBody.get_m_cfg()
  cfg.set_viterations(viterations)
  cfg.set_piterations(piterations)

  // Try to improve ground contact/friction/stability (bindings may vary).
  try {
    cfg.set_kDF(0.94)
  } catch {
    // ignore
  }
  try {
    cfg.set_kDP(nodeDamping)
  } catch {
    // ignore
  }
  try {
    cfg.set_kPR(0.0)
  } catch {
    // ignore
  }
  try {
    cfg.set_kVC(0.9)
  } catch {
    // ignore
  }
  try {
    cfg.set_kVCF(1.0)
  } catch {
    // ignore
  }
  try {
    cfg.set_kMT(poseMatching)
  } catch {
    // ignore
  }
  try {
    cfg.set_kCHR(1.0)
  } catch {
    // ignore
  }
  try {
    cfg.set_kKHR(0.8)
  } catch {
    // ignore
  }
  try {
    cfg.set_kSHR(0.78)
  } catch {
    // ignore
  }
  try {
    cfg.set_diterations(18)
  } catch {
    // ignore
  }
  try {
    cfg.set_citerations(10)
  } catch {
    // ignore
  }
  try {
    cfg.set_kSRHR_CL(1.0)
  } catch {
    // ignore
  }
  try {
    cfg.set_kSKHR_CL(1.0)
  } catch {
    // ignore
  }

  try {
    const materials = softBody.get_m_materials()
    for (let i = 0; i < materials.size(); i++) {
      const m = materials.at(i)
      m.set_m_kLST(linearStiffness)
      m.set_m_kAST(angularStiffness)
      if (volumeStiffness !== undefined && volumeStiffness > 0) {
        try {
          m.set_m_kVST(volumeStiffness)
        } catch {
          // ignore
        }
      }
    }
    try {
      if (materials.size() > 0) softBody.generateBendingConstraints(bendingDistance, materials.at(0))
    } catch {
      // ignore
    }
  } catch {
    // ignore
  }

  // Только rigid/soft. VF_SS на сшитом столе даёт внутренние отталкивания → сетка «разлетается».
  try {
    cfg.set_collisions(0x01 | 0x02)
  } catch {}

  softBody.setTotalMass(mass, false)
  try {
    // Keep overall table shape while allowing local flex.
    softBody.setPose(true, true)
  } catch {
    // ignore
  }
  softBody.setActivationState(4) // disable deactivation

  try {
    const collisionObject = world.ammo.castObject(softBody, world.ammo.btCollisionObject)
    try {
      collisionObject.setFriction(1.0)
    } catch {
      // ignore
    }
  } catch {
    // ignore
  }
  try {
    softBody.setFriction(1.0)
  } catch {
    // ignore
  }
  try {
    softBody.setRollingFriction(0.2)
  } catch {
    // ignore
  }
  try {
    softBody.setSpinningFriction(0.2)
  } catch {
    // ignore
  }

  // Кластеры под CL_RS; больше кластеров — ровнее контакт с коробкой пола.
  try {
    softBody.generateClusters(26)
  } catch {
    // ignore
  }

  world.addSoftBody(softBody, 1, -1)
  try {
    world.ammo.destroy(softBodyHelpers)
  } catch {
    // ignore
  }

  const nodeCount = softBody.get_m_nodes().size()

  const mesh = generateGraphics
    ? new THREE.Mesh(
        geometry,
        new THREE.MeshStandardMaterial({
          color: 0x9ad7ff,
          metalness: 0.05,
          roughness: 0.75
        })
      )
    : null

  const dispose = () => {
    try {
      world.physicsWorld.removeSoftBody(softBody)
    } catch {
      // ignore
    }
    try {
      world.ammo.destroy(softBody)
    } catch {
      // ignore
    }
  }

  return { softBody, mesh, geometry, nodeCount, vertexNormals: normals, dispose }
}

