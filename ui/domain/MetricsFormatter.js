/**
 * Pure static utility for formatting clinical and model performance metrics.
 * No constructor, no state, no side effects.
 */
export class MetricsFormatter {
  /** @param {number} value @returns {string} */
  static formatAUC(value) {
    if (value === null || value === undefined) return '—';
    return value.toFixed(4);
  }

  /** @param {number} value @returns {string} */
  static formatPercent(value) {
    if (value === null || value === undefined) return '—';
    return `${(value * 100).toFixed(1)}%`;
  }

  /** @param {number} a @param {number} b @returns {string} */
  static formatDelta(a, b) {
    const delta = a - b;
    const sign = delta >= 0 ? '+' : '−';
    return `${sign}${Math.abs(delta).toFixed(3)}`;
  }

  /** @param {number} value @returns {string} */
  static formatConfidence(value) {
    if (value === null || value === undefined) return '—';
    return `${(value * 100).toFixed(1)}%`;
  }

  /** @param {number} w @returns {string} */
  static formatAdvLossWeight(w) {
    return `w_adv = ${w.toFixed(2)}`;
  }

  /** @param {number} e @returns {string} */
  static formatEpsilon(e) {
    return `ε = ${e.toFixed(2)} (L∞)`;
  }

  /**
   * Maps internal metric key to human-readable label.
   * @param {string} name
   * @returns {string}
   */
  static metricLabel(name) {
    const labels = {
      auc: 'AUC-ROC',
      balancedAccuracy: 'Balanced Accuracy',
      melanomaSensitivity: 'MEL Sensitivity',
      nonMelSpecificity: 'Non-MEL Specificity',
      advBalancedAccuracy: 'Adversarial Bal. Acc.'
    };
    return labels[name] ?? name;
  }
}
