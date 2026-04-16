/**
 * Pure static utility for clinical threshold validation.
 * Enforces hard clinical safety constraints.
 */
export class ThresholdValidator {
  static CLINICAL_TARGETS = {
    auc: 0.80,
    balancedAccuracy: 0.65,
    melanomaSensitivity: 0.40,
    nonMelSpecificity: 0.82
  };

  static BORDERLINE_MARGIN = 0.05;

  /**
   * Validates all four clinical metrics against targets.
   * @param {import('./ClinicalMetrics.js').ClinicalMetrics} metrics
   * @returns {{ passed: boolean, failures: Array<{ metric: string, value: number, target: number }> }}
   */
  static validateMetrics(metrics) {
    const failures = [];
    const checks = [
      ['auc', metrics.auc],
      ['balancedAccuracy', metrics.balancedAccuracy],
      ['melanomaSensitivity', metrics.melanomaSensitivity],
      ['nonMelSpecificity', metrics.nonMelSpecificity]
    ];

    for (const [metric, value] of checks) {
      const target = ThresholdValidator.CLINICAL_TARGETS[metric];
      if (value !== null && value < target) {
        failures.push({ metric, value, target });
      }
    }

    return { passed: failures.length === 0, failures };
  }

  /**
   * Checks if a confidence score falls within the borderline zone around theta.
   * @param {number} confidence
   * @param {number} theta - decision threshold
   * @returns {boolean}
   */
  static isBorderline(confidence, theta) {
    return Math.abs(confidence - theta) < ThresholdValidator.BORDERLINE_MARGIN;
  }

  /**
   * Validates GA-optimised parameters are within safe ranges.
   * @param {import('./GAParameters.js').GAParameters} params
   * @returns {{ valid: boolean, errors: string[] }}
   */
  static validateGAParameters(params) {
    const errors = [];
    if (params.alpha < 0 || params.alpha > 1) {
      errors.push('α must be in [0, 1]');
    }
    if (params.tau < 0.5 || params.tau > 2.0) {
      errors.push('τ must be in [0.5, 2.0]');
    }
    if (params.theta < 0.3 || params.theta > 0.7) {
      errors.push('θ must be in [0.3, 0.7]');
    }
    if (params.theta > 0.50) {
      errors.push('Warning: θ > 0.50 increases false negatives — clinical risk');
    }
    return { valid: errors.length === 0, errors };
  }

  /**
   * Assesses catastrophic forgetting risk based on adversarial loss weight.
   * @param {number} advLossWeight
   * @returns {'safe'|'transition'|'forgetting'}
   */
  static phaseTransitionRisk(advLossWeight) {
    if (advLossWeight < 0.20) return 'safe';
    if (advLossWeight <= 0.25) return 'transition';
    return 'forgetting';
  }
}
