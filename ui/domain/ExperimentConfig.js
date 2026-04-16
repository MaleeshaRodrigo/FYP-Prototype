import { ClinicalMetrics } from './ClinicalMetrics.js';

/**
 * Immutable value object representing AT hyperparameters for a single experiment version.
 */
export class ExperimentConfig {
  static PHASE_TRANSITION_THRESHOLD = 0.225;

  /** @type {string} */ #version;
  /** @type {number} */ #advLossWeight;
  /** @type {number} */ #epsilon;
  /** @type {number} */ #pgdSteps;
  /** @type {number} */ #pgdAlpha;
  /** @type {number} */ #learningRate;
  /** @type {number} */ #epochs;
  /** @type {ClinicalMetrics} */ #cleanMetrics;
  /** @type {ClinicalMetrics|null} */ #advMetrics;
  /** @type {string} */ #status;

  /**
   * @param {{ version: string, advLossWeight: number, epsilon: number, pgdSteps: number, pgdAlpha: number, learningRate: number, epochs: number, cleanMetrics: ClinicalMetrics, advMetrics: ClinicalMetrics|null, status?: string }} params
   */
  constructor({ version, advLossWeight, epsilon, pgdSteps, pgdAlpha, learningRate, epochs, cleanMetrics, advMetrics, status }) {
    this.#version = version;
    this.#advLossWeight = advLossWeight;
    this.#epsilon = epsilon;
    this.#pgdSteps = pgdSteps;
    this.#pgdAlpha = pgdAlpha;
    this.#learningRate = learningRate;
    this.#epochs = epochs;
    this.#cleanMetrics = cleanMetrics;
    this.#advMetrics = advMetrics ?? null;
    this.#status = status ?? 'partial';
    Object.freeze(this);
  }

  get version() { return this.#version; }
  get advLossWeight() { return this.#advLossWeight; }
  get epsilon() { return this.#epsilon; }
  get pgdSteps() { return this.#pgdSteps; }
  get pgdAlpha() { return this.#pgdAlpha; }
  get learningRate() { return this.#learningRate; }
  get epochs() { return this.#epochs; }
  get cleanMetrics() { return this.#cleanMetrics; }
  get advMetrics() { return this.#advMetrics; }
  get status() { return this.#status; }

  get isAbovePhaseTransition() {
    return this.#advLossWeight >= ExperimentConfig.PHASE_TRANSITION_THRESHOLD;
  }

  /** @returns {'safe'|'transition'|'forgetting'} */
  get riskLevel() {
    if (this.#advLossWeight < 0.20) return 'safe';
    if (this.#advLossWeight <= 0.25) return 'transition';
    return 'forgetting';
  }

  /** CSS class for colour-coded experiment table rows. */
  get rowCSSClass() {
    const map = { safe: 'experiment-row--safe', transition: 'experiment-row--transition', forgetting: 'experiment-row--forgetting' };
    return map[this.riskLevel];
  }

  /**
   * @param {object} data - API/fixture row (snake_case)
   * @returns {ExperimentConfig}
   */
  static fromApiResponse(data) {
    const cleanMetrics = new ClinicalMetrics({
      auc: data.best_auc,
      balancedAccuracy: data.best_bal_acc,
      melanomaSensitivity: data.best_sens_mel,
      nonMelSpecificity: null,
      modelVersion: data.version,
      evaluationType: 'clean'
    });

    const advMetrics = data.adv_bal_acc != null
      ? new ClinicalMetrics({
          auc: null,
          balancedAccuracy: data.adv_bal_acc,
          melanomaSensitivity: null,
          nonMelSpecificity: null,
          modelVersion: data.version,
          evaluationType: 'adversarial'
        })
      : null;

    return new ExperimentConfig({
      version: data.version,
      advLossWeight: data.adv_loss_weight,
      epsilon: data.epsilon,
      pgdSteps: data.pgd_steps,
      pgdAlpha: data.pgd_alpha,
      learningRate: data.lr,
      epochs: data.epochs,
      cleanMetrics,
      advMetrics,
      status: data.status
    });
  }
}
