import type { Genome } from '../actuators/ActuatorSystem'
import type { GenerationSummary, TrainerResult } from './TrainerEvolution'

export interface CemConfig {
  generations?: number
  populationSize?: number
  eliteCount?: number
  smoothing?: number
  sigmaAmplitude0?: number
  sigmaOmega0?: number
  sigmaPhase0?: number
}

function clamp(x: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, x))
}

function gaussianRandom(): number {
  let u = 0
  let v = 0
  while (u === 0) u = Math.random()
  while (v === 0) v = Math.random()
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v)
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

function vectorToGenome(x: Float64Array, actuatorCount: number): Genome {
  const amplitudes: number[] = []
  const omegas: number[] = []
  const phases: number[] = []
  for (let i = 0; i < actuatorCount; i++) {
    amplitudes.push(clamp(x[3 * i], 0.0, 0.22))
    omegas.push(clamp(x[3 * i + 1], 0.2, 8.5))
    phases.push(x[3 * i + 2])
  }
  return { amplitudes, omegas, phases }
}

function cloneVec(x: Float64Array): Float64Array {
  const c = new Float64Array(x.length)
  c.set(x)
  return c
}

export class TrainerCem {
  private readonly config: Required<CemConfig>
  private readonly storageKey = 'walking_softbody_best_genome_cem_v1'

  constructor(config: CemConfig = {}) {
    const pop = Math.max(4, config.populationSize ?? 14)
    this.config = {
      generations: config.generations ?? 20,
      populationSize: pop,
      eliteCount: Math.min(pop - 1, Math.max(2, config.eliteCount ?? Math.floor(pop * 0.3))),
      smoothing: clamp(config.smoothing ?? 0.7, 0.05, 1.0),
      sigmaAmplitude0: clamp(config.sigmaAmplitude0 ?? 0.04, 0.005, 0.2),
      sigmaOmega0: clamp(config.sigmaOmega0 ?? 0.65, 0.05, 2.5),
      sigmaPhase0: clamp(config.sigmaPhase0 ?? 0.55, 0.05, 2.5)
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
    const eliteCount = this.config.eliteCount
    const alpha = this.config.smoothing
    const dim = actuatorCount * 3

    const stored = this.loadBestGenome(actuatorCount) ?? randomGenome(actuatorCount)
    let mean = genomeToVector(stored)
    let sigma = new Float64Array(dim)
    for (let i = 0; i < actuatorCount; i++) {
      sigma[3 * i] = this.config.sigmaAmplitude0
      sigma[3 * i + 1] = this.config.sigmaOmega0
      sigma[3 * i + 2] = this.config.sigmaPhase0
    }

    const sigmaFloor = new Float64Array(dim)
    for (let i = 0; i < actuatorCount; i++) {
      sigmaFloor[3 * i] = 0.004
      sigmaFloor[3 * i + 1] = 0.08
      sigmaFloor[3 * i + 2] = 0.09
    }

    let bestReward = Number.NEGATIVE_INFINITY
    let bestGenome: Genome | null = null

    for (let g = 0; g < gens; g++) {
      onLog?.(`Generation ${g + 1}/${gens} (CEM, N=${NP}, elite=${eliteCount})`)

      const population: Array<{ x: Float64Array; reward: number; genome: Genome }> = []
      for (let i = 0; i < NP; i++) {
        const x = new Float64Array(dim)
        for (let d = 0; d < dim; d++) {
          x[d] = mean[d] + sigma[d] * gaussianRandom()
        }
        const genome = vectorToGenome(x, actuatorCount)
        const reward = await evaluateGenome(genome, i, g)
        population.push({ x: cloneVec(x), reward, genome })

        if (reward > bestReward) {
          bestReward = reward
          bestGenome = genome
          this.saveBestGenome(genome)
        }
      }

      population.sort((a, b) => b.reward - a.reward)
      const elites = population.slice(0, eliteCount)
      const topRewardThisGen = population[0].reward

      const eliteMean = new Float64Array(dim)
      for (const e of elites) {
        for (let d = 0; d < dim; d++) eliteMean[d] += e.x[d]
      }
      for (let d = 0; d < dim; d++) eliteMean[d] /= eliteCount

      const eliteStd = new Float64Array(dim)
      for (const e of elites) {
        for (let d = 0; d < dim; d++) {
          const diff = e.x[d] - eliteMean[d]
          eliteStd[d] += diff * diff
        }
      }
      for (let d = 0; d < dim; d++) eliteStd[d] = Math.sqrt(eliteStd[d] / eliteCount)

      for (let d = 0; d < dim; d++) {
        mean[d] = (1 - alpha) * mean[d] + alpha * eliteMean[d]
        sigma[d] = Math.max(sigmaFloor[d], (1 - alpha) * sigma[d] + alpha * eliteStd[d])
      }

      onLog?.(`  bestReward=${bestReward.toFixed(3)} topThisGen=${topRewardThisGen.toFixed(3)}`)
      await onGenerationEnd?.({
        generationIndex: g,
        generationCount: gens,
        bestRewardSoFar: bestReward,
        topRewardThisGen,
        bestGenomeThisGen: population[0].genome
      })
    }

    if (!bestGenome) throw new Error('CEM: no genome')
    return { bestGenome, bestReward }
  }
}
