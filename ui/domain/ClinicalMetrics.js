import { MetricsFormatter } from './MetricsFormatter.js';

/**
 * Immutable value object encapsulating the four clinical performance metrics.
 */
export class ClinicalMetrics {
  static TARGETS = { auc: 0.80, balAcc: 0.65, sens: 0.40, spec: 0.82 };

  /** @type {number|null} */ #auc;
  /** @type {number} */ #balancedAccuracy;
  /** @type {number|null} */ #melanomaSensitivity;
  /** @type {number|null} */ #nonMelSpecificity;
  /** @type {string} */ #modelVersion;
  /** @type {'clean'|'adversarial'} */ #evaluationType;

  /**
   * @param {{ auc: number|null, balancedAccuracy: number, melanomaSensitivity: number|null, nonMelSpecificity: number|null, modelVersion: string, evaluationType: 'clean'|'adversarial' }} params
   */
  constructor({ auc, balancedAccuracy, melanomaSensitivity, nonMelSpecificity, modelVersion, evaluationType }) {
    this.#auc = auc;
    this.#balancedAccuracy = balancedAccuracy;
    this.#melanomaSensitivity = melanomaSensitivity;
    this.#nonMelSpecificity = nonMelSpecificity;
    this.#modelVersion = modelVersion;
    this.#evaluationType = evaluationType;
    Object.freeze(this);
  }

  get auc() { return this.#auc; }
  get balancedAccuracy() { return this.#balancedAccuracy; }
  get melanomaSensitivity() { return this.#melanomaSensitivity; }
  get nonMelSpecificity() { return this.#nonMelSpecificity; }
  get modelVersion() { return this.#modelVersion; }
  get evaluationType() { return this.#evaluationType; }

  get allTargetsMet() {
    return this.failedTargets.length === 0;
  }

  /** @returns {string[]} */
  get failedTargets() {
    const failed = [];
    if (this.#auc !== null && this.#auc < ClinicalMetrics.TARGETS.auc) failed.push('auc');
    if (this.#balancedAccuracy < ClinicalMetrics.TARGETS.balAcc) failed.push('balancedAccuracy');
    if (this.#melanomaSensitivity !== null && this.#melanomaSensitivity < ClinicalMetrics.TARGETS.sens) failed.push('melanomaSensitivity');
    if (this.#nonMelSpecificity !== null && this.#nonMelSpecificity < ClinicalMetrics.TARGETS.spec) failed.push('nonMelSpecificity');
    return failed;
  }

  /**
   * @param {string} metricName
   * @returns {boolean}
   */
  meetsTarget(metricName) {
    const map = {
      auc: [this.#auc, ClinicalMetrics.TARGETS.auc],
      balancedAccuracy: [this.#balancedAccuracy, ClinicalMetrics.TARGETS.balAcc],
      melanomaSensitivity: [this.#melanomaSensitivity, ClinicalMetrics.TARGETS.sens],
      nonMelSpecificity: [this.#nonMelSpecificity, ClinicalMetrics.TARGETS.spec]
    };
    const [value, target] = map[metricName] ?? [null, 0];
    return value !== null && value >= target;
  }

  /**
   * Computes signed deltas from another ClinicalMetrics instance.
   * @param {ClinicalMetrics} other
   * @returns {object}
   */
  deltaFrom(other) {
    return {
      auc: this.#auc !== null && other.auc !== null ? this.#auc - other.auc : null,
      balancedAccuracy: this.#balancedAccuracy - other.balancedAccuracy,
      melanomaSensitivity: this.#melanomaSensitivity !== null && other.melanomaSensitivity !== null
        ? this.#melanomaSensitivity - other.melanomaSensitivity : null,
      nonMelSpecificity: this.#nonMelSpecificity !== null && other.nonMelSpecificity !== null
        ? this.#nonMelSpecificity - other.nonMelSpecificity : null
    };
  }

  /** @returns {object} Formatted display strings via MetricsFormatter. */
  toDisplayFormat() {
    return {
      auc: MetricsFormatter.formatAUC(this.#auc),
      balancedAccuracy: MetricsFormatter.formatPercent(this.#balancedAccuracy),
      melanomaSensitivity: MetricsFormatter.formatPercent(this.#melanomaSensitivity),
      nonMelSpecificity: MetricsFormatter.formatPercent(this.#nonMelSpecificity),
      modelVersion: this.#modelVersion,
      evaluationType: this.#evaluationType
    };
  }

  /**
   * @param {object} data - API response
   * @returns {ClinicalMetrics}
   */
  static fromApiResponse(data) {
    return new ClinicalMetrics({
      auc: data.auc,
      balancedAccuracy: data.balancedAccuracy ?? data.balanced_accuracy,
      melanomaSensitivity: data.melanomaSensitivity ?? data.melanoma_sensitivity,
      nonMelSpecificity: data.nonMelSpecificity ?? data.non_mel_specificity,
      modelVersion: data.modelVersion ?? data.model_version,
      evaluationType: data.evaluationType ?? data.evaluation_type
    });
  }
}
