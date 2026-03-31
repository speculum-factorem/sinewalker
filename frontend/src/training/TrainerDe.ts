import type { Genome } from '../actuators/ActuatorSystem'
import type { GenerationSummary, TrainerResult } from './TrainerEvolution'

export interface DeConfig {
  generations?: number
  populationSize?: number
  differentialWeight?: number
  crossoverRate?: number
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

export class TrainerDe {
  private readonly config: Required<DeConfig>
  private readonly storageKey = 'walking_softbody_best_genome_de_v1'

  constructor(config: DeConfig = {}) {
    this.config = {
      generations: config.generations ?? 20,
      populationSize: Math.max(4, config.populationSize ?? 14),
      differentialWeight: clamp(config.differentialWeight ?? 0.75, 0.05, 1.8),
      crossoverRate: clamp(config.crossoverRate ?? 0.9, 0.0, 1.0)
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
    const generations = this.config.generations
    const NP = Math.max(4, this.config.populationSize)
    const F = this.config.differentialWeight
    const CR = this.config.crossoverRate
    const dim = actuatorCount * 3

    const stored = this.loadBestGenome(actuatorCount)
    const population: Genome[] = Array.from({ length: NP }, () => randomGenome(actuatorCount))
    if (stored) population[0] = stored

    const fitness = new Float64Array(NP)
    let bestReward = Number.NEGATIVE_INFINITY
    let bestGenome: Genome | null = null

    for (let i = 0; i < NP; i++) {
      const reward = await evaluateGenome(population[i], i, -1)
      fitness[i] = reward
      if (reward > bestReward) {
        bestReward = reward
        bestGenome = population[i]
        this.saveBestGenome(bestGenome)
      }
    }

    for (let g = 0; g < generations; g++) {
      onLog?.(`Generation ${g + 1}/${generations} (DE/rand/1/bin, NP=${NP})`)

      for (let i = 0; i < NP; i++) {
        let a = i
        let b = i
        let c = i
        while (a === i) a = Math.floor(Math.random() * NP)
        while (b === i || b === a) b = Math.floor(Math.random() * NP)
        while (c === i || c === a || c === b) c = Math.floor(Math.random() * NP)

        const xa = genomeToVector(population[a])
        const xb = genomeToVector(population[b])
        const xc = genomeToVector(population[c])
        const xt = genomeToVector(population[i])
        const v = new Float64Array(dim)
        for (let d = 0; d < dim; d++) v[d] = xa[d] + F * (xb[d] - xc[d])

        const jRand = Math.floor(Math.random() * dim)
        const u = new Float64Array(dim)
        for (let d = 0; d < dim; d++) {
          const useDonor = d === jRand || Math.random() < CR
          u[d] = useDonor ? v[d] : xt[d]
        }

        const trial = vectorToGenome(u, actuatorCount)
        const trialReward = await evaluateGenome(trial, i, g)
        if (trialReward >= fitness[i]) {
          population[i] = trial
          fitness[i] = trialReward
          if (trialReward > bestReward) {
            bestReward = trialReward
            bestGenome = trial
            this.saveBestGenome(bestGenome)
          }
        }
      }

      let topRewardThisGen = Number.NEGATIVE_INFINITY
      let bestGenomeThisGen = population[0]
      for (let i = 0; i < NP; i++) topRewardThisGen = Math.max(topRewardThisGen, fitness[i])
      for (let i = 0; i < NP; i++) {
        if (fitness[i] >= topRewardThisGen) bestGenomeThisGen = population[i]
      }
      onLog?.(`  bestReward=${bestReward.toFixed(3)} topThisGen=${topRewardThisGen.toFixed(3)}`)
      await onGenerationEnd?.({
        generationIndex: g,
        generationCount: generations,
        bestRewardSoFar: bestReward,
        topRewardThisGen,
        bestGenomeThisGen
      })
    }

    if (!bestGenome) throw new Error('DE: no genome')
    return { bestGenome, bestReward }
  }
}
