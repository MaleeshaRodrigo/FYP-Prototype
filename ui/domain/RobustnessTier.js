/**
 * Enum-like class representing adversarial robustness certification levels.
 * Eliminates magic strings for tier comparisons across the platform.
 */
export class RobustnessTier {
  /** @type {string} */ #code;
  /** @type {string} */ #label;
  /** @type {string} */ #icon;
  /** @type {string} */ #cssClass;
  /** @type {number} */ #level;

  /**
   * @param {string} code
   * @param {string} label
   * @param {string} icon
   * @param {string} cssClass
   * @param {number} level
   */
  constructor(code, label, icon, cssClass, level) {
    this.#code = code;
    this.#label = label;
    this.#icon = icon;
    this.#cssClass = cssClass;
    this.#level = level;
    Object.freeze(this);
  }

  get code() { return this.#code; }
  get label() { return this.#label; }
  get icon() { return this.#icon; }
  get cssClass() { return this.#cssClass; }
  get level() { return this.#level; }

  static BASELINE = new RobustnessTier('BASELINE', 'Baseline', '🟡', 'warning', 0);
  static PARTIAL = new RobustnessTier('PARTIAL', 'Partial Robustness', '🟠', 'partial', 1);
  static TRADES = new RobustnessTier('TRADES', 'TRADES-Certified', '🟢', 'success', 2);

  /**
   * Determines tier from adversarial loss weight used during training.
   * @param {number} w - adversarial loss weight
   * @param {boolean} [isTradesLoss=false] - whether TRADES loss function was used
   * @returns {RobustnessTier}
   */
  static fromAdvLossWeight(w, isTradesLoss = false) {
    if (isTradesLoss && w > 0.20) return RobustnessTier.TRADES;
    if (w <= 0.05) return RobustnessTier.BASELINE;
    if (w <= 0.20) return RobustnessTier.PARTIAL;
    return RobustnessTier.PARTIAL;
  }

  /**
   * Determines tier from model version identifier.
   * @param {string} versionId
   * @returns {RobustnessTier}
   */
  static fromModelVersion(versionId) {
    if (versionId.includes('trades')) return RobustnessTier.TRADES;
    if (versionId === 'stage1') return RobustnessTier.BASELINE;
    return RobustnessTier.PARTIAL;
  }

  /**
   * @param {RobustnessTier} tier
   * @returns {boolean}
   */
  isAtLeast(tier) {
    return this.#level >= tier.level;
  }

  toString() {
    return `${this.#icon} ${this.#label}`;
  }
}
