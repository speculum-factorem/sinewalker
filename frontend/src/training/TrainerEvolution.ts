import type { Genome } from '../actuators/ActuatorSystem'

export interface EvolutionConfig {
  populationSize?: number
  eliteCount?: number
  generations?: number
  mutationSigmaAmplitude?: number
  mutationSigmaOmega?: number
  mutationSigmaPhase?: number
}

export interface TrainerResult {
  bestGenome: Genome
  bestReward: number
}

export interface GenerationSummary {
  generationIndex: number
  generationCount: number
  bestRewardSoFar: number
  topRewardThisGen: number
}

function clamp(x: number, min: number, max: number) {
  return Math.max(min, Math.min(max, x))
}

function gaussianRandom(): number {
  // Box-Muller transform.
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

export class TrainerEvolution {
  private readonly config: Required<EvolutionConfig>
  // v2: old genomes may contain aggressive amplitudes from unstable experiments.
  private readonly storageKey = 'walking_softbody_best_genome_v2'

  constructor(config: EvolutionConfig = {}) {
    this.config = {
      populationSize: config.populationSize ?? 14,
      eliteCount: config.eliteCount ?? 3,
      generations: config.generations ?? 20,
      mutationSigmaAmplitude: config.mutationSigmaAmplitude ?? 0.03,
      mutationSigmaOmega: config.mutationSigmaOmega ?? 0.4,
      mutationSigmaPhase: config.mutationSigmaPhase ?? 0.6
    }
  }

  loadBestGenome(): Genome | null {
    try {
      const raw = localStorage.getItem(this.storageKey)
      if (!raw) return null
      const parsed = JSON.parse(raw) as Genome
      if (!parsed || !parsed.amplitudes || !parsed.omegas || !parsed.phases) return null
      return this.sanitizeGenome(parsed)
    } catch {
      return null
    }
  }

  saveBestGenome(genome: Genome) {
    localStorage.setItem(this.storageKey, JSON.stringify(this.sanitizeGenome(genome)))
  }

  private sanitizeGenome(genome: Genome): Genome {
    return {
      amplitudes: genome.amplitudes.map((a) => clamp(a, 0.0, 0.22)),
      omegas: genome.omegas.map((o) => clamp(o, 0.2, 8.5)),
      phases: genome.phases.slice()
    }
  }

  private mutate(parent: Genome): Genome {
    const { mutationSigmaAmplitude, mutationSigmaOmega, mutationSigmaPhase } = this.config

    const amplitudes = parent.amplitudes.map((a) =>
      clamp(a + gaussianRandom() * mutationSigmaAmplitude, 0.0, 0.22)
    )
    const omegas = parent.omegas.map((o) =>
      clamp(o + gaussianRandom() * mutationSigmaOmega, 0.2, 8.5)
    )
    const phases = parent.phases.map((p) => p + gaussianRandom() * mutationSigmaPhase)

    return this.sanitizeGenome({ amplitudes, omegas, phases })
  }

  async run(
    actuatorCount: number,
    evaluateGenome: (genome: Genome, individualIndex: number, generationIndex: number) => Promise<number>,
    onLog?: (line: string) => void,
    onGenerationEnd?: (summary: GenerationSummary) => void
  ): Promise<TrainerResult> {
    const { populationSize, eliteCount, generations } = this.config

    let bestGenome: Genome | null = this.loadBestGenome()
    let bestReward = Number.NEGATIVE_INFINITY

    // Ensure we have a genome sized correctly.
    if (!bestGenome || bestGenome.amplitudes.length !== actuatorCount) {
      bestGenome = randomGenome(actuatorCount)
    } else {
      bestGenome = this.sanitizeGenome(bestGenome)
    }

    let population: Genome[] = Array.from({ length: populationSize }, () => randomGenome(actuatorCount))
    // Seed population with best so far.
    population[0] = bestGenome

    for (let g = 0; g < generations; g++) {
      const scored: Array<{ genome: Genome; reward: number }> = []

      onLog?.(`Generation ${g + 1}/${generations}`)

      for (let i = 0; i < population.length; i++) {
        const genome = this.sanitizeGenome(population[i])
        const reward = await evaluateGenome(genome, i, g)
        scored.push({ genome, reward })

        if (reward > bestReward) {
          bestReward = reward
          bestGenome = genome
          this.saveBestGenome(bestGenome)
        }
      }

      scored.sort((a, b) => b.reward - a.reward)

      const elites = scored.slice(0, eliteCount).map((s) => s.genome)
      onLog?.(`  bestReward=${bestReward.toFixed(3)} topElite=${scored[0].reward.toFixed(3)}`)
      onGenerationEnd?.({
        generationIndex: g,
        generationCount: generations,
        bestRewardSoFar: bestReward,
        topRewardThisGen: scored[0].reward
      })

      // Create next population: keep elites + mutate them.
      population = []
      for (let i = 0; i < eliteCount; i++) population.push(elites[i])
      while (population.length < populationSize) {
        const parent = elites[Math.floor(Math.random() * elites.length)]
        population.push(this.mutate(parent))
      }
    }

    if (!bestGenome) throw new Error('Failed to find best genome')
    return { bestGenome, bestReward }
  }
}

