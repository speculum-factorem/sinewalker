import type { Genome } from '../actuators/ActuatorSystem'
import type { GenerationSummary, TrainerResult } from './TrainerEvolution'

export interface PsoConfig {
  generations?: number
  populationSize?: number
  inertiaWeight?: number
  cognitiveWeight?: number
  socialWeight?: number
}

function clamp(x: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, x))
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

function genomeToVector(genome: Genome): Float64Array {
  const n = genome.amplitudes.length
  const x = new Float64Array(3 * n)
  for (let i = 0; i < n; i++) {
    x[3 * i] = genome.amplitudes[i]
    x[3 * i + 1] = genome.omegas[i]
    x[3 * i + 2] = genome.phases[i]
  }
  return x
}

function vectorToGenome(vec: Float64Array, actuatorCount: number): Genome {
  const amplitudes: number[] = []
  const omegas: number[] = []
  const phases: number[] = []
  for (let i = 0; i < actuatorCount; i++) {
    amplitudes.push(clamp(vec[3 * i], 0.0, 0.22))
    omegas.push(clamp(vec[3 * i + 1], 0.2, 8.5))
    phases.push(vec[3 * i + 2])
  }
  return { amplitudes, omegas, phases }
}

function cloneVec(x: Float64Array): Float64Array {
  const c = new Float64Array(x.length)
  c.set(x)
  return c
}

export class TrainerPso {
  private readonly config: Required<PsoConfig>
  private readonly storageKey = 'walking_softbody_best_genome_pso_v1'

  constructor(config: PsoConfig = {}) {
    this.config = {
      generations: config.generations ?? 20,
      populationSize: Math.max(4, config.populationSize ?? 14),
      inertiaWeight: config.inertiaWeight ?? 0.72,
      cognitiveWeight: config.cognitiveWeight ?? 1.49,
      socialWeight: config.socialWeight ?? 1.49
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
    const gens = this.config.generations
    const NP = this.config.populationSize
    const w = this.config.inertiaWeight
    const c1 = this.config.cognitiveWeight
    const c2 = this.config.socialWeight
    const dim = actuatorCount * 3

    const x: Float64Array[] = Array.from({ length: NP }, () => genomeToVector(randomGenome(actuatorCount)))
    const v: Float64Array[] = Array.from({ length: NP }, () => new Float64Array(dim))
    const pBest: Float64Array[] = Array.from({ length: NP }, () => new Float64Array(dim))
    const pBestReward = new Float64Array(NP)
    pBestReward.fill(Number.NEGATIVE_INFINITY)

    const stored = this.loadBestGenome(actuatorCount)
    if (stored) x[0] = genomeToVector(stored)

    let gBest: Float64Array | null = null
    let gBestReward = Number.NEGATIVE_INFINITY

    const velocityLimit = new Float64Array(dim)
    for (let i = 0; i < actuatorCount; i++) {
      velocityLimit[3 * i] = 0.05
      velocityLimit[3 * i + 1] = 0.8
      velocityLimit[3 * i + 2] = Math.PI * 0.35
    }

    for (let g = 0; g < gens; g++) {
      onLog?.(`Generation ${g + 1}/${gens} (PSO, swarm=${NP})`)

      let topRewardThisGen = Number.NEGATIVE_INFINITY
      let bestGenomeThisGen: Genome | null = null
      for (let i = 0; i < NP; i++) {
        const genome = vectorToGenome(x[i], actuatorCount)
        const reward = await evaluateGenome(genome, i, g)
        if (reward >= topRewardThisGen) {
          topRewardThisGen = reward
          bestGenomeThisGen = genome
        }

        if (reward > pBestReward[i]) {
          pBestReward[i] = reward
          pBest[i] = cloneVec(x[i])
        }
        if (reward > gBestReward) {
          gBestReward = reward
          gBest = cloneVec(x[i])
          this.saveBestGenome(genome)
        }
      }

      if (!gBest) throw new Error('PSO: no global best vector')

      onLog?.(`  bestReward=${gBestReward.toFixed(3)} topThisGen=${topRewardThisGen.toFixed(3)}`)
      await onGenerationEnd?.({
        generationIndex: g,
        generationCount: gens,
        bestRewardSoFar: gBestReward,
        topRewardThisGen,
        bestGenomeThisGen: bestGenomeThisGen ?? vectorToGenome(gBest, actuatorCount)
      })

      for (let i = 0; i < NP; i++) {
        const xi = x[i]
        const vi = v[i]
        const pi = pBest[i]
        for (let d = 0; d < dim; d++) {
          const r1 = Math.random()
          const r2 = Math.random()
          const vd = w * vi[d] + c1 * r1 * (pi[d] - xi[d]) + c2 * r2 * (gBest[d] - xi[d])
          vi[d] = clamp(vd, -velocityLimit[d], velocityLimit[d])
          xi[d] += vi[d]
        }
      }
    }

    const bestGenome = gBest ? vectorToGenome(gBest, actuatorCount) : null
    if (!bestGenome) throw new Error('PSO: no genome')
    return { bestGenome, bestReward: gBestReward }
  }
}
