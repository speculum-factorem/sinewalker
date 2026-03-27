import * as THREE from 'three'

/** Переиспользуемые объекты Ammo для сброса rigid body без аллокаций на каждый эпизод. */
export type RigidBodyResetPool = {
  transform: any
  origin: any
  rotation: any
  zero: any
}

export function createRigidBodyResetPool(ammo: any): RigidBodyResetPool {
  return {
    transform: new ammo.btTransform(),
    origin: new ammo.btVector3(0, 0, 0),
    rotation: new ammo.btQuaternion(0, 0, 0, 1),
    zero: new ammo.btVector3(0, 0, 0)
  }
}

/** Вернуть тело в заданную позу и обнулить скорости (между эпизодами обучения). */
export function resetRigidBodyToPose(
  body: any,
  pos: THREE.Vector3,
  quat: THREE.Quaternion,
  pool: RigidBodyResetPool
) {
  pool.origin.setValue(pos.x, pos.y, pos.z)
  pool.rotation.setValue(quat.x, quat.y, quat.z, quat.w)
  pool.transform.setIdentity()
  pool.transform.setOrigin(pool.origin)
  pool.transform.setRotation(pool.rotation)
  body.setWorldTransform(pool.transform)
  const ms = body.getMotionState?.()
  if (ms) ms.setWorldTransform(pool.transform)
  body.setLinearVelocity(pool.zero)
  body.setAngularVelocity(pool.zero)
  try {
    body.setActivationState(4)
  } catch {
    // ignore
  }
  try {
    body.clearForces()
  } catch {
    // ignore
  }
}

export function comWorldFromRigidBody(body: any): THREE.Vector3 {
  const t = body.getCenterOfMassTransform()
  const o = t.getOrigin()
  return new THREE.Vector3(o.x(), o.y(), o.z())
}

/** Базовые значения как в `AmmoWorld.createDynamicConvexHull`; при k=1 поведение как при обучении. */
const TABLE_REF_LINEAR_DAMPING = 0.42
const TABLE_REF_ANGULAR_DAMPING = 0.32

/**
 * «Упругость» жёсткого тела как материал: отскок при контакте и демпфирование движения.
 * `elasticityMultiplier` — тот же множитель, что и у ползунка (1 = референс).
 */
export function applyRigidBodyElasticityMaterial(body: any, elasticityMultiplier: number) {
  const delta = elasticityMultiplier - 1
  const restitution = Math.max(0, Math.min(0.52, delta * 0.32))
  const lin = Math.max(0.1, Math.min(0.68, TABLE_REF_LINEAR_DAMPING - delta * 0.2))
  const ang = Math.max(0.08, Math.min(0.58, TABLE_REF_ANGULAR_DAMPING - delta * 0.15))
  try {
    body.setRestitution(restitution)
  } catch {
    /* ignore */
  }
  try {
    body.setDamping(lin, ang)
  } catch {
    /* ignore */
  }
}

export function syncMeshFromRigidBody(mesh: THREE.Object3D, body: any) {
  const t = body.getCenterOfMassTransform()
  const o = t.getOrigin()
  const r = t.getRotation()
  mesh.position.set(o.x(), o.y(), o.z())
  mesh.quaternion.set(r.x(), r.y(), r.z(), r.w())
}
