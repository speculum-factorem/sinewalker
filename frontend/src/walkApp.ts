import * as THREE from 'three'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import { mergeGeometries, mergeVertices } from 'three/addons/utils/BufferGeometryUtils.js'

import { AmmoWorld } from './physics/AmmoWorld'
import { computeFootPivotsLocal, geometryPositionsFlat } from './physics/geometryFeet'
import {
  applyRigidBodyElasticityMaterial,
  createRigidBodyResetPool,
  resetRigidBodyToPose,
  syncMeshFromRigidBody,
  type RigidBodyResetPool
} from './physics/rigidBodyThree'
import type { Genome } from './actuators/ActuatorSystem'
import { ActuatorSystemRigid } from './actuators/ActuatorSystemRigid'
import { RewardCalculator } from './training/RewardCalculator'
import { TrainerEvolution } from './training/TrainerEvolution'
import {
  applyLegVertexFlex,
  footOutwardDirsXZ,
  legRegionYMax,
  snapshotGeometryPositions
} from './render/tableLegVisualFlex'

/** Fixed evolution / actuator layout (no UI). */
const MAX_ACTUATORS = 4
const TRAINING_POPULATION = 12
const DEFAULT_TRAINING_GENERATIONS = 22
const TRAINING_GENERATIONS_MIN = 1
const TRAINING_GENERATIONS_MAX = 200
const EPISODE_STEPS = 230
const DT_SECONDS = 0.016
const VIZ_EVERY_INDIVIDUAL = 2
const VIZ_SPEED = 6
const GROUND_PLANE_Y = 0
/** Ползунок и визуальный изгиб, и физика приводов; это значение = множитель 1.0 к базовой жёсткости. */
const DEFAULT_ELASTICITY_SLIDER = 0.086
const BASE_ACTUATOR_SPRING_GAIN = 38
const BASE_ACTUATOR_MAX_FORCE = 46
const ELASTICITY_MULT_MIN = 0.25
const ELASTICITY_MULT_MAX = 2.2

const TABLE_RIGID_MASS = 14

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

  const merged = mergeGeometries(pieces, false)
  if (!merged) throw new Error('Failed to merge table parts')
  for (const p of pieces) p.dispose()

  const welded = mergeVertices(merged, 0.06)
  merged.dispose()
  welded.computeVertexNormals()
  return normalizeGeometry(welded)
}

export function createWalkApp(root: HTMLDivElement) {
  root.innerHTML = `
    <div id="viewport"></div>
    <div id="hud">
      <h2>Стол на плоскости</h2>
      <p class="hud-desc">Жёсткий convex hull на полу. Ползунок: приводы, визуальный изгиб и материал всего стола (отскок, демпфирование).</p>
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
  const simStartBtn = root.querySelector<HTMLButtonElement>('#simStart')
  const simStopBtn = root.querySelector<HTMLButtonElement>('#simStop')
  const trainStartBtn = root.querySelector<HTMLButtonElement>('#trainStart')
  const trainGenerationsEl = root.querySelector<HTMLInputElement>('#trainGenerations')

  if (
    !viewport ||
    !statusEl ||
    !rewardChart ||
    !flexGainEl ||
    !flexGainValEl ||
    !sphereVizEl ||
    !sphereVizValEl ||
    !simStartBtn ||
    !simStopBtn ||
    !trainStartBtn ||
    !trainGenerationsEl
  ) {
    throw new Error('HUD elements missing')
  }

  const readTrainingGenerations = (): number => {
    const raw = parseInt(trainGenerationsEl.value, 10)
    if (!Number.isFinite(raw)) return DEFAULT_TRAINING_GENERATIONS
    return Math.min(TRAINING_GENERATIONS_MAX, Math.max(TRAINING_GENERATIONS_MIN, Math.round(raw)))
  }

  let showActuatorSpheres = true

  let simTimeSeconds = 0
  let isPlaying = false
  let externalStepping = false

  const updateSimControlButtons = () => {
    simStartBtn.disabled = isPlaying
    simStopBtn.disabled = !isPlaying
  }

  simStartBtn.addEventListener('click', () => {
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

  let scene: THREE.Scene
  let camera: THREE.PerspectiveCamera
  let renderer: THREE.WebGLRenderer
  let controls: any

  let meshGeometry: THREE.BufferGeometry | null = null
  let footPivotsLocal: THREE.Vector3[] = []
  let meshTransform = new THREE.Matrix4().makeTranslation(0, 0.55, 0)

  const setStatus = (line: string) => {
    statusEl.textContent = line
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
    const sys = new ActuatorSystemRigid(
      ammoWorld,
      true,
      GROUND_PLANE_Y,
      BASE_ACTUATOR_SPRING_GAIN,
      BASE_ACTUATOR_MAX_FORCE
    )
    sys.createForFeet(tableRigidBody, footPivotsLocal, { scene, tableSpawnMatrix: meshTransform })
    if (!previewGenome || previewGenome.amplitudes.length !== footPivotsLocal.length) {
      previewGenome = randomGenome(footPivotsLocal.length)
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
      `Симуляция выключена — «Запустить». Стол, приводов: ${footPivotsLocal.length}. Можно «Обучить политику».`
    )
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
  }

  const initThree = () => {
    scene = new THREE.Scene()
    scene.background = new THREE.Color(0x0b0c10)

    scene.add(new THREE.AmbientLight(0xffffff, 0.65))
    const dir = new THREE.DirectionalLight(0xffffff, 1.05)
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
        color: 0x1c2535,
        metalness: 0.04,
        roughness: 0.94
      })
    )
    floor.rotation.x = -Math.PI / 2
    floor.position.y = floorY
    scene.add(floor)

    const grid = new THREE.GridHelper(24, 48, 0x334155, 0x1e2836)
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

    const reward = new RewardCalculator({ fallMinY: 0.12, fallPenalty: 5, comAxis: 'x' })
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

    const trainer = new TrainerEvolution({
      populationSize,
      eliteCount: Math.max(2, Math.floor(populationSize * 0.2)),
      generations
    })

    try {
      const best = await trainer.run(nAct, async (genome, individualIndex, generationIndex) => {
        const show = (generationIndex * populationSize + individualIndex) % VIZ_EVERY_INDIVIDUAL === 0
        if (show) {
          setStatus(`Поколение ${generationIndex + 1}/${generations}, особь ${individualIndex + 1}/${populationSize}`)
          return await runVisibleEpisode(genome as Genome, maxSteps, dt, VIZ_SPEED)
        }

        resetSimulationForEpisode(genome as Genome)
        const reward = new RewardCalculator({ fallMinY: 0.12, fallPenalty: 5, comAxis: 'x' })
        reward.startRigid(tableRigidBody!)

        for (let step = 0; step < maxSteps; step++) {
          actuatorSystem?.applyAtTime(simTimeSeconds)
          ammoWorld!.step(dt, 32)
          if (step % 10 === 0) reward.observeRigidStep(tableRigidBody!)
          simTimeSeconds += dt
          if (step % 80 === 0) await yieldToUI()
        }
        return reward.finishRigid(tableRigidBody!).reward
      }, () => {}, (summary) => {
        rewardHistory.push(summary.bestRewardSoFar)
        drawRewardChart()
        const maybeBest = trainer.loadBestGenome()
        if (maybeBest) {
          previewGenome = maybeBest
          resetSimulationForEpisode(previewGenome)
        }
      })

      previewGenome = best.bestGenome
      resetSimulationForEpisode(previewGenome)
      isPlaying = false
      updateSimControlButtons()
      setStatus(
        `Обучение готово (reward ≈ ${best.bestReward.toFixed(3)}). Нажмите «Запустить», чтобы проиграть политику.`
      )
    } catch (e) {
      setStatus(`Ошибка обучения: ${(e as Error).message}`)
      isPlaying = false
      updateSimControlButtons()
    }
  }

  const start = async () => {
    initThree()
    meshGeometry = createTableGeometry()
    meshGeometry.computeBoundingBox()
    const bbox = meshGeometry.boundingBox
    const lift = bbox ? Math.max(0.02, -bbox.min.y + 0.025) : 0.55
    meshTransform = new THREE.Matrix4().makeTranslation(0, lift, 0)
    footPivotsLocal = computeFootPivotsLocal(meshGeometry, MAX_ACTUATORS)

    setStatus('Загрузка Ammo.js…')

    const ammoGlobal: any = (globalThis as any).Ammo
    if (typeof ammoGlobal !== 'function') {
      throw new Error('Ammo global is not a function (ammo.js not loaded?)')
    }
    ammo = await ammoGlobal()

    initPhysicsWorldOnce()
    drawRewardChart()
    animate()

    trainStartBtn.addEventListener('click', async () => {
      if (!ammo || !meshGeometry) return
      trainStartBtn.disabled = true
      trainGenerationsEl.disabled = true
      try {
        await runTraining()
      } finally {
        trainGenerationsEl.disabled = false
        trainStartBtn.disabled = false
      }
    })
  }

  void start()
}
