import type { Genome } from '../actuators/ActuatorSystem'
import type { GenerationSummary, TrainerResult } from './TrainerEvolution'

export interface CmaEsConfig {
  /** Число итераций CMA-ES (каждая оценивает λ особей). */
  generations?: number
  /** Размер потомства λ; по умолчанию max(4, 4 + ⌊3 ln n⌋). */
  lambda?: number
  /** Начальный шаг σ в пространстве кодирования. */
  sigma0?: number
}

function clamp(x: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, x))
}

function encodeGenomeToVector(g: Genome): Float64Array {
  const n = g.amplitudes.length
  const x = new Float64Array(3 * n)
  const eps = 1e-9
  for (let i = 0; i < n; i++) {
    const a = clamp(g.amplitudes[i], eps, 0.22 - eps)
    const ra = (a / 0.22) * 2 - 1
    x[3 * i] = 0.5 * Math.log((1 + ra) / (1 - ra))

    const o = clamp(g.omegas[i], 0.2 + eps, 8.5 - eps)
    const ro = ((o - 0.2) / 8.3) * 2 - 1
    x[3 * i + 1] = 0.5 * Math.log((1 + ro) / (1 - ro))

    x[3 * i + 2] = g.phases[i]
  }
  return x
}

function decodeVectorToGenome(x: Float64Array, actuatorCount: number): Genome {
  const amplitudes: number[] = []
  const omegas: number[] = []
  const phases: number[] = []
  for (let i = 0; i < actuatorCount; i++) {
    const za = x[3 * i]
    const zo = x[3 * i + 1]
    const zp = x[3 * i + 2]
    amplitudes.push((0.22 * (Math.tanh(za) + 1)) / 2)
    omegas.push(0.2 + (8.3 * (Math.tanh(zo) + 1)) / 2)
    phases.push(zp)
  }
  return { amplitudes, omegas, phases }
}

function randomGenome(actuatorCount: number): Genome {
  const amplitudes: number[] = []
  const omegas: number[] = []
  const phases: number[] = []
  for (let i = 0; i < actuatorCount; i++) {
    amplitudes.push(0.05 + Math.random() * 0.11)
    omegas.push(0.8 + Math.random() * 4.6)
    phases.push((i % 2 === 0 ? 0 : Math.PI) + (Math.random() - 0.5) * 0.5)
  }
  return { amplitudes, omegas, phases }
}

export function cmaDefaultLambdaFromDim(dim: number): number {
  return Math.max(4, 4 + Math.floor(3 * Math.log(dim)))
}

export function cmaDefaultLambda(actuatorCount: number): number {
  return cmaDefaultLambdaFromDim(actuatorCount * 3)
}

/** Нижняя треугольная L: C = L L^T (диагональ положительна). */
function cholesky(C: Float64Array, n: number): Float64Array | null {
  const L = new Float64Array(n * n)
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = C[i * n + j]
      for (let k = 0; k < j; k++) s -= L[i * n + k] * L[j * n + k]
      if (i > j) {
        const d = L[j * n + j]
        if (d <= 0) return null
        L[i * n + j] = s / d
      } else {
        if (s <= 0) return null
        L[i * n + i] = Math.sqrt(s)
      }
    }
  }
  return L
}

function choleskyWithJitter(C: Float64Array, n: number): Float64Array {
  for (let r = 0; r < 13; r++) {
    const jitter = 1e-12 * Math.pow(10, r * 0.25)
    const Cj = new Float64Array(n * n)
    Cj.set(C)
    for (let d = 0; d < n; d++) Cj[d * n + d] += jitter
    const L = cholesky(Cj, n)
    if (L) return L
  }
  throw new Error('CMA-ES: Cholesky decomposition failed')
}

function mulLowerL(L: Float64Array, n: number, z: Float64Array, y: Float64Array) {
  for (let i = 0; i < n; i++) {
    let s = 0
    for (let j = 0; j <= i; j++) s += L[i * n + j] * z[j]
    y[i] = s
  }
}

/** Решить L x = b, L — нижнетреугольная. */
function solveLower(L: Float64Array, n: number, b: Float64Array, x: Float64Array) {
  for (let i = 0; i < n; i++) {
    let s = b[i]
    for (let j = 0; j < i; j++) s -= L[i * n + j] * x[j]
    const d = L[i * n + i]
    x[i] = d !== 0 ? s / d : 0
  }
}

function gaussianVector(out: Float64Array) {
  for (let i = 0; i < out.length; i += 2) {
    let u = 0
    let v = 0
    while (u === 0) u = Math.random()
    while (v === 0) v = Math.random()
    const r = Math.sqrt(-2 * Math.log(u))
    out[i] = r * Math.cos(2 * Math.PI * v)
    if (i + 1 < out.length) out[i + 1] = r * Math.sin(2 * Math.PI * v)
  }
}

function chiN(nn: number): number {
  return Math.sqrt(nn) * (1 - 1 / (4 * nn) + 1 / (21 * nn * nn))
}

export class TrainerCmaEs {
  private readonly config: Required<CmaEsConfig>
  private readonly storageKey = 'walking_softbody_best_genome_cma_v1'

  constructor(config: CmaEsConfig = {}) {
    this.config = {
      generations: config.generations ?? 20,
      lambda: config.lambda ?? 0,
      sigma0: config.sigma0 ?? 0.35
    }
  }

  loadBestGenome(actuatorCount: number): Genome | null {
    try {
      const raw = localStorage.getItem(this.storageKey)
      if (!raw) return null
      const parsed = JSON.parse(raw) as Genome
      if (!parsed?.amplitudes || parsed.amplitudes.length !== actuatorCount) return null
      return parsed
    } catch {
      return null
    }
  }

  saveBestGenome(genome: Genome) {
    localStorage.setItem(this.storageKey, JSON.stringify(genome))
  }

  async run(
    actuatorCount: number,
    evaluateGenome: (genome: Genome, individualIndex: number, generationIndex: number) => Promise<number>,
    onLog?: (line: string) => void,
    onGenerationEnd?: (summary: GenerationSummary) => void | Promise<void>
  ): Promise<TrainerResult> {
    const nDim = actuatorCount * 3
    const λ = this.config.lambda > 0 ? this.config.lambda : cmaDefaultLambdaFromDim(nDim)
    const μ = Math.floor(λ / 2)
    if (μ < 1) throw new Error('CMA-ES: λ must be >= 2')

    const weightsRaw = new Float64Array(μ)
    let ws = 0
    for (let i = 0; i < μ; i++) {
      weightsRaw[i] = Math.log(μ + 0.5) - Math.log(i + 1)
      ws += weightsRaw[i]
    }
    const w = new Float64Array(μ)
    for (let i = 0; i < μ; i++) w[i] = weightsRaw[i] / ws

    let μeff = 0
    for (let i = 0; i < μ; i++) μeff += w[i] * w[i]
    μeff = 1 / μeff

    const n = nDim
    const cσ = (μeff + 2) / (n + μeff + 5)
    const dσ = 1 + (2 * Math.max(0, Math.sqrt((μeff - 1) / (n + 1)) - 1)) + cσ
    const cc = (4 + μeff / n) / (n + 4 + (2 * μeff) / n)
    const c1 = 2 / ((n + 1.3) ** 2 + μeff)
    const cμ = Math.min(1 - c1, (2 * (μeff - 2 + 1 / μeff)) / ((n + 2) ** 2 + μeff))
    const χn = chiN(n)

    let stored = this.loadBestGenome(actuatorCount)
    const m = stored ? encodeGenomeToVector(stored) : encodeGenomeToVector(randomGenome(actuatorCount))
    let σ = this.config.sigma0
    const C = new Float64Array(n * n)
    for (let d = 0; d < n; d++) C[d * n + d] = 1
    const pσ = new Float64Array(n)
    const pc = new Float64Array(n)

    let bestReward = Number.NEGATIVE_INFINITY
    let bestGenome: Genome | null = stored

    const gens = this.config.generations
    const zBuf = new Float64Array(n)
    const yBuf = new Float64Array(n)
    const xBuf = new Float64Array(n)
    const invsqrtY = new Float64Array(n)
    const yW = new Float64Array(n)
    const ys: Float64Array[] = Array.from({ length: λ }, () => new Float64Array(n))

    for (let g = 0; g < gens; g++) {
      onLog?.(`Generation ${g + 1}/${gens} (CMA-ES, λ=${λ})`)

      const population: Array<{ y: Float64Array; reward: number; genome: Genome }> = []

      const L = choleskyWithJitter(C, n)

      for (let k = 0; k < λ; k++) {
        gaussianVector(zBuf)
        mulLowerL(L, n, zBuf, yBuf)
        for (let i = 0; i < n; i++) xBuf[i] = m[i] + σ * yBuf[i]
        const genome = decodeVectorToGenome(xBuf, actuatorCount)
        const reward = await evaluateGenome(genome, k, g)
        const yCopy = ys[k]
        yCopy.set(yBuf)
        population.push({ y: yCopy, reward, genome })

        if (reward > bestReward) {
          bestReward = reward
          bestGenome = genome
          this.saveBestGenome(genome)
        }
      }

      population.sort((a, b) => b.reward - a.reward)

      const topRewardThisGen = population[0].reward

      onLog?.(`  bestReward=${bestReward.toFixed(3)} topThisGen=${topRewardThisGen.toFixed(3)}`)
      await onGenerationEnd?.({
        generationIndex: g,
        generationCount: gens,
        bestRewardSoFar: bestReward,
        topRewardThisGen,
        bestGenomeThisGen: population[0].genome
      })

      yW.fill(0)
      for (let i = 0; i < μ; i++) {
        const yi = population[i].y
        const wi = w[i]
        for (let j = 0; j < n; j++) yW[j] += wi * yi[j]
      }

      for (let j = 0; j < n; j++) m[j] += σ * yW[j]

      solveLower(L, n, yW, invsqrtY)

      for (let j = 0; j < n; j++) {
        pσ[j] = (1 - cσ) * pσ[j] + Math.sqrt(cσ * (2 - cσ) * μeff) * invsqrtY[j]
      }

      let pσNorm = 0
      for (let j = 0; j < n; j++) pσNorm += pσ[j] * pσ[j]
      pσNorm = Math.sqrt(pσNorm)

      σ *= Math.exp((cσ / dσ) * (pσNorm / χn - 1))
      σ = clamp(σ, 1e-20, 1e10)

      const denom = Math.sqrt(1 - (1 - cσ) ** (2 * (g + 1)))
      const hs = denom > 0 && pσNorm / denom < (1.4 + 2 / (n + 1)) * χn ? 1 : 0

      for (let j = 0; j < n; j++) {
        pc[j] = (1 - cc) * pc[j] + hs * Math.sqrt(cc * (2 - cc) * μeff) * yW[j]
      }

      const scaleC = 1 - c1 - cμ
      for (let i = 0; i < n * n; i++) C[i] *= scaleC

      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          C[i * n + j] += c1 * pc[i] * pc[j]
        }
      }

      for (let k = 0; k < μ; k++) {
        const yk = population[k].y
        const wk = w[k]
        for (let i = 0; i < n; i++) {
          for (let j = 0; j < n; j++) {
            C[i * n + j] += cμ * wk * yk[i] * yk[j]
          }
        }
      }

      for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
          const v = 0.5 * (C[i * n + j] + C[j * n + i])
          C[i * n + j] = v
          C[j * n + i] = v
        }
      }
    }

    if (!bestGenome) throw new Error('CMA-ES: no genome')
    return { bestGenome, bestReward }
  }
}
