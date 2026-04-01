import * as THREE from 'three'

export type AmmoType = any

export interface AmmoWorldOptions {
  gravityY?: number
  solverIterations?: number
}

export interface RigidBodyHandle {
  body: any
  shape: any
}

export interface StaticBoxOptions {
  friction?: number
  restitution?: number
}

export class AmmoWorld {
  readonly ammo: AmmoType
  readonly physicsWorld: any
  readonly softBodyWorldInfo: any

  private readonly rigidBodies: any[] = []
  private readonly softBodies: any[] = []
  private readonly constraints: any[] = []
  private readonly owned: any[] = []
  private readonly margin = 0.05
  private readonly collisionConfiguration: any
  private readonly dispatcher: any
  private readonly broadphase: any
  private readonly solver: any
  private readonly softBodySolver: any

  constructor(ammo: AmmoType, options: AmmoWorldOptions = {}) {
    this.ammo = ammo

    const gravityY = options.gravityY ?? -9.8
    const solverIterations = options.solverIterations ?? 10

    this.collisionConfiguration = new ammo.btSoftBodyRigidBodyCollisionConfiguration()
    this.dispatcher = new ammo.btCollisionDispatcher(this.collisionConfiguration)
    this.broadphase = new ammo.btDbvtBroadphase()
    this.solver = new ammo.btSequentialImpulseConstraintSolver()
    this.softBodySolver = new ammo.btDefaultSoftBodySolver()

    this.physicsWorld = new ammo.btSoftRigidDynamicsWorld(
      this.dispatcher,
      this.broadphase,
      this.solver,
      this.collisionConfiguration,
      this.softBodySolver
    )

    const g1 = new ammo.btVector3(0, gravityY, 0)
    const g2 = new ammo.btVector3(0, gravityY, 0)
    this.owned.push(g1, g2)
    this.physicsWorld.setGravity(g1)
    this.physicsWorld.getWorldInfo().set_m_gravity(g2)

    this.physicsWorld.getSolverInfo().set_m_numIterations(solverIterations)

    this.softBodyWorldInfo = this.physicsWorld.getWorldInfo()
    // Без этого soft body не стыкуется с диспетчером/фазой мира → «проваливается» сквозь пол.
    this.softBodyWorldInfo.set_m_dispatcher(this.dispatcher)
    this.softBodyWorldInfo.set_m_broadphase(this.physicsWorld.getBroadphase())
  }

  private own<T>(obj: T): T {
    this.owned.push(obj)
    return obj
  }

  /**
   * Толстый статический ящик: верхняя грань ровно y = 0. Плоскость в Bullet+Ammo для soft body
   * часто даёт артефакты; коробка стабильнее и всё ещё «платформа».
   */
  createGround(sizeX = 20, sizeZ = 20, _y = 0): RigidBodyHandle {
    const { ammo } = this
    const halfH = 1.5
    const halfExtents = this.own(new ammo.btVector3(sizeX * 0.5, halfH, sizeZ * 0.5))
    const shape = this.own(new ammo.btBoxShape(halfExtents))
    shape.setMargin(0.01)

    const transform = this.own(new ammo.btTransform())
    transform.setIdentity()
    transform.setOrigin(this.own(new ammo.btVector3(0, -halfH, 0)))

    const motionState = this.own(new ammo.btDefaultMotionState(transform))
    const localInertia = this.own(new ammo.btVector3(0, 0, 0))

    const rbInfo = this.own(new ammo.btRigidBodyConstructionInfo(0, motionState, shape, localInertia))
    const body = this.own(new ammo.btRigidBody(rbInfo))
    body.setFriction(1.0)
    try {
      body.setRestitution(0)
    } catch {
      // ignore
    }

    const ALL = -1
    const DEFAULT_GROUP = 1
    this.physicsWorld.addRigidBody(body, DEFAULT_GROUP, ALL)
    this.rigidBodies.push(body)
    return { body, shape }
  }

  createStaticBox(
    halfExtentsVec: THREE.Vector3,
    position: THREE.Vector3,
    rotation = new THREE.Quaternion(),
    options: StaticBoxOptions = {}
  ): RigidBodyHandle {
    const { ammo } = this
    const halfExtents = this.own(new ammo.btVector3(halfExtentsVec.x, halfExtentsVec.y, halfExtentsVec.z))
    const shape = this.own(new ammo.btBoxShape(halfExtents))
    shape.setMargin(0.01)

    const transform = this.own(new ammo.btTransform())
    transform.setIdentity()
    transform.setOrigin(this.own(new ammo.btVector3(position.x, position.y, position.z)))
    transform.setRotation(this.own(new ammo.btQuaternion(rotation.x, rotation.y, rotation.z, rotation.w)))

    const motionState = this.own(new ammo.btDefaultMotionState(transform))
    const localInertia = this.own(new ammo.btVector3(0, 0, 0))
    const rbInfo = this.own(new ammo.btRigidBodyConstructionInfo(0, motionState, shape, localInertia))
    const body = this.own(new ammo.btRigidBody(rbInfo))
    body.setFriction(options.friction ?? 1.0)
    try {
      body.setRestitution(options.restitution ?? 0.0)
    } catch {
      // ignore
    }

    const ALL = -1
    const DEFAULT_GROUP = 1
    this.physicsWorld.addRigidBody(body, DEFAULT_GROUP, ALL)
    this.rigidBodies.push(body)
    return { body, shape }
  }

  createKinematicBodySphere(radius: number, pos: THREE.Vector3): RigidBodyHandle {
    const { ammo } = this
    const shape = this.own(new ammo.btSphereShape(radius))

    const transform = this.own(new ammo.btTransform())
    transform.setIdentity()
    transform.setOrigin(this.own(new ammo.btVector3(pos.x, pos.y, pos.z)))

    const motionState = this.own(new ammo.btDefaultMotionState(transform))
    const localInertia = this.own(new ammo.btVector3(0, 0, 0))

    // mass = 0 => kinematic
    const rbInfo = this.own(new ammo.btRigidBodyConstructionInfo(0, motionState, shape, localInertia))
    const body = this.own(new ammo.btRigidBody(rbInfo))
    body.setActivationState(4) // disable deactivation
    try {
      body.setFriction(1.0)
    } catch {
      // ignore
    }

    // CF_KINEMATIC_OBJECT = 2, CF_NO_CONTACT_RESPONSE = 4
    // Anchor spheres should drive nodes only via appendAnchor, not collide physically.
    const flags = body.getCollisionFlags()
    body.setCollisionFlags(flags | 2 | 4)

    try {
      body.setCcdMotionThreshold(1e-4)
      body.setCcdSweptSphereRadius(Math.max(1e-4, radius * 0.95))
    } catch {
      // ignore
    }

    const DEFAULT_GROUP = 1
    this.physicsWorld.addRigidBody(body, DEFAULT_GROUP, 0)
    this.rigidBodies.push(body)
    return { body, shape }
  }

  unregisterRigidBody(body: any) {
    try {
      this.physicsWorld.removeRigidBody(body)
    } catch {
      // ignore
    }
    const i = this.rigidBodies.indexOf(body)
    if (i >= 0) this.rigidBodies.splice(i, 1)
  }

  addSoftBody(softBody: any, collisionGroup = 1, collisionMask = -1) {
    this.physicsWorld.addSoftBody(softBody, collisionGroup, collisionMask)
    this.softBodies.push(softBody)
  }

  addConstraint(constraint: any, disableCollisionsBetweenLinkedBodies = true) {
    this.physicsWorld.addConstraint(constraint, disableCollisionsBetweenLinkedBodies)
    this.constraints.push(constraint)
  }

  removeConstraint(constraint: any) {
    try {
      this.physicsWorld.removeConstraint(constraint)
    } catch {
      // ignore
    }
    const i = this.constraints.indexOf(constraint)
    if (i >= 0) this.constraints.splice(i, 1)
    try {
      this.ammo.destroy(constraint)
    } catch {
      // ignore
    }
  }

  /**
   * Выпуклая оболочка вершин (локальные координаты), динамическое тело.
   */
  createDynamicConvexHull(mass: number, flatVertices: number[], worldMatrix: THREE.Matrix4): any {
    const { ammo } = this
    const shape = new ammo.btConvexHullShape()
    const nv = Math.floor(flatVertices.length / 3)
    for (let i = 0; i < nv; i++) {
      const ix = i * 3
      const recalc = i === nv - 1
      const p = new ammo.btVector3(flatVertices[ix], flatVertices[ix + 1], flatVertices[ix + 2])
      shape.addPoint(p, recalc)
      try {
        ammo.destroy(p)
      } catch {
        // ignore
      }
    }
    try {
      shape.setMargin(0.02)
      shape.recalcLocalAabb()
    } catch {
      // ignore
    }

    const localInertia = this.own(new ammo.btVector3(0, 0, 0))
    shape.calculateLocalInertia(mass, localInertia)

    const wt = this.own(new ammo.btTransform())
    wt.setIdentity()
    const pos = new THREE.Vector3()
    const quat = new THREE.Quaternion()
    const scl = new THREE.Vector3()
    worldMatrix.decompose(pos, quat, scl)
    wt.setOrigin(this.own(new ammo.btVector3(pos.x, pos.y, pos.z)))
    wt.setRotation(this.own(new ammo.btQuaternion(quat.x, quat.y, quat.z, quat.w)))

    const motionState = this.own(new ammo.btDefaultMotionState(wt))
    const rbInfo = this.own(new ammo.btRigidBodyConstructionInfo(mass, motionState, shape, localInertia))
    const body = this.own(new ammo.btRigidBody(rbInfo))
    body.setFriction(0.95)
    try {
      body.setRestitution(0)
    } catch {
      // ignore
    }
    body.setDamping(0.42, 0.32)
    body.setActivationState(4)

    try {
      body.setCcdMotionThreshold(1e-4)
      let maxExt = 0
      for (let i = 0; i < flatVertices.length; i += 3) {
        const d = Math.hypot(flatVertices[i]!, flatVertices[i + 1]!, flatVertices[i + 2]!)
        if (d > maxExt) maxExt = d
      }
      body.setCcdSweptSphereRadius(Math.max(0.02, maxExt * 0.12))
    } catch {
      // ignore
    }

    this.owned.push(shape)

    const ALL = -1
    const DEFAULT_GROUP = 1
    this.physicsWorld.addRigidBody(body, DEFAULT_GROUP, ALL)
    this.rigidBodies.push(body)
    return body
  }

  step(dtSeconds: number, maxSubSteps = 10) {
    this.physicsWorld.stepSimulation(dtSeconds, maxSubSteps)
  }

  dispose() {
    const { ammo } = this
    for (const c of this.constraints) {
      try {
        this.physicsWorld.removeConstraint(c)
      } catch {
        // ignore
      }
      try {
        ammo.destroy(c)
      } catch {
        // ignore
      }
    }
    this.constraints.length = 0

    for (const rb of this.rigidBodies) {
      this.physicsWorld.removeRigidBody(rb)
    }
    this.rigidBodies.length = 0

    for (const sb of this.softBodies) {
      this.physicsWorld.removeSoftBody(sb)
    }
    this.softBodies.length = 0

    // Destroy owned Bullet objects to release WASM heap memory.
    // Order matters less for Ammo.js, but we remove from world first.
    for (let i = this.owned.length - 1; i >= 0; i--) {
      try {
        ammo.destroy(this.owned[i])
      } catch {
        // ignore
      }
    }
    this.owned.length = 0

    try { ammo.destroy(this.physicsWorld) } catch {}
    try { ammo.destroy(this.softBodySolver) } catch {}
    try { ammo.destroy(this.solver) } catch {}
    try { ammo.destroy(this.broadphase) } catch {}
    try { ammo.destroy(this.dispatcher) } catch {}
    try { ammo.destroy(this.collisionConfiguration) } catch {}
  }

  getDefaultMargin() {
    return this.margin
  }
}

