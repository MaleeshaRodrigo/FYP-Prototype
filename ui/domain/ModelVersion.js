import { RobustnessTier } from './RobustnessTier.js';

/**
 * Immutable value object representing a single HARE model checkpoint.
 */
export class ModelVersion {
  /** @type {string} */ #id;
  /** @type {string} */ #label;
  /** @type {string} */ #checkpoint;
  /** @type {1|2|3} */ #stage;
  /** @type {RobustnessTier} */ #robustnessTier;
  /** @type {boolean} */ #isActive;
  /** @type {boolean} */ #isPending;

  /**
   * @param {{ id: string, label: string, checkpoint: string, stage: 1|2|3, robustnessTier: RobustnessTier, isActive: boolean, isPending: boolean }} params
   */
  constructor({ id, label, checkpoint, stage, robustnessTier, isActive, isPending }) {
    this.#id = id;
    this.#label = label;
    this.#checkpoint = checkpoint;
    this.#stage = stage;
    this.#robustnessTier = robustnessTier;
    this.#isActive = isActive;
    this.#isPending = isPending;
    Object.freeze(this);
  }

  get id() { return this.#id; }
  get label() { return this.#label; }
  get checkpoint() { return this.#checkpoint; }
  get stage() { return this.#stage; }
  get robustnessTier() { return this.#robustnessTier; }
  get isActive() { return this.#isActive; }
  get isPending() { return this.#isPending; }

  get isTradesCertified() {
    return this.#robustnessTier === RobustnessTier.TRADES;
  }

  get displayLabel() {
    return `${this.#label}${this.#isPending ? ' (pending)' : ''}`;
  }

  /** @returns {'active'|'deprecated'|'pending'} */
  get statusTag() {
    if (this.#isPending) return 'pending';
    if (this.#isActive) return 'active';
    return 'deprecated';
  }

  toJSON() {
    return {
      id: this.#id,
      label: this.#label,
      checkpoint: this.#checkpoint,
      stage: this.#stage,
      robustnessTier: this.#robustnessTier.code,
      isActive: this.#isActive,
      isPending: this.#isPending
    };
  }

  /**
   * @param {object} data - raw API/fixture data
   * @returns {ModelVersion}
   */
  static fromApiResponse(data) {
    return new ModelVersion({
      id: data.id,
      label: data.label,
      checkpoint: data.checkpoint,
      stage: data.stage,
      robustnessTier: RobustnessTier.fromModelVersion(data.id),
      isActive: data.isActive ?? data.status === 'active',
      isPending: data.isPending ?? data.status === 'pending'
    });
  }
}
