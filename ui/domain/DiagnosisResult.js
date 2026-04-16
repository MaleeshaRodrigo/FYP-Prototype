import { MetricsFormatter } from './MetricsFormatter.js';
import { ModelVersion } from './ModelVersion.js';
import { RobustnessTier } from './RobustnessTier.js';

/**
 * Immutable value object wrapping a single HARE prediction.
 * Enforces clinical safety invariants (borderline detection, GradCAM requirement).
 */
export class DiagnosisResult {
  /** @type {string} */ #imageId;
  /** @type {'MEL'|'NON_MEL'} */ #prediction;
  /** @type {number} */ #confidence;
  /** @type {number} */ #threshold;
  /** @type {ModelVersion} */ #modelVersion;
  /** @type {Date} */ #timestamp;

  /**
   * @param {{ imageId: string, prediction: 'MEL'|'NON_MEL', confidence: number, threshold: number, modelVersion: ModelVersion, timestamp: Date }} params
   */
  constructor({ imageId, prediction, confidence, threshold, modelVersion, timestamp }) {
    this.#imageId = imageId;
    this.#prediction = prediction;
    this.#confidence = confidence;
    this.#threshold = threshold;
    this.#modelVersion = modelVersion;
    this.#timestamp = timestamp;
    Object.freeze(this);
  }

  get imageId() { return this.#imageId; }
  get prediction() { return this.#prediction; }
  get confidence() { return this.#confidence; }
  get threshold() { return this.#threshold; }
  get modelVersion() { return this.#modelVersion; }
  get timestamp() { return this.#timestamp; }

  /** @returns {boolean} */
  get isMelanoma() { return this.#prediction === 'MEL'; }

  /** Clinical safety: flags predictions near the decision boundary. */
  get isBorderline() { return Math.abs(this.#confidence - this.#threshold) < 0.05; }

  /** GradCAM is mandatory for all MEL-positive diagnoses. */
  get requiresGradCAM() { return this.isMelanoma; }

  /** @returns {'high'|'borderline'|'low'} */
  get riskLevel() {
    if (this.isBorderline) return 'borderline';
    return this.isMelanoma ? 'high' : 'low';
  }

  get formattedConfidence() {
    return MetricsFormatter.formatConfidence(this.#confidence);
  }

  toJSON() {
    return {
      imageId: this.#imageId,
      prediction: this.#prediction,
      confidence: this.#confidence,
      threshold: this.#threshold,
      modelVersion: this.#modelVersion.toJSON(),
      timestamp: this.#timestamp.toISOString()
    };
  }

  /**
   * @param {object} data - API response (snake_case)
   * @returns {DiagnosisResult}
   */
  static fromApiResponse(data) {
    const tier = RobustnessTier.fromModelVersion(data.model_version);
    const modelVersion = new ModelVersion({
      id: data.model_version,
      label: data.model_version,
      checkpoint: '',
      stage: data.model_version.includes('trades') ? 3 : 2,
      robustnessTier: tier,
      isActive: true,
      isPending: false
    });

    return new DiagnosisResult({
      imageId: data.image_id,
      prediction: data.prediction,
      confidence: data.confidence,
      threshold: data.threshold,
      modelVersion,
      timestamp: new Date(data.timestamp)
    });
  }
}
