/**
 * Mutable value object for GA-optimised ensemble parameters.
 * Edited by System Admin through the parameter editor.
 */
export class GAParameters {
  /** @type {number} CNN fusion weight, range [0, 1] */
  alpha;
  /** @type {number} Temperature scaling, range [0.5, 2.0] */
  tau;
  /** @type {number} Decision threshold, range [0.3, 0.7] */
  theta;

  /**
   * @param {{ alpha: number, tau: number, theta: number }} params
   */
  constructor({ alpha, tau, theta }) {
    this.alpha = alpha;
    this.tau = tau;
    this.theta = theta;
  }

  /**
   * Validates parameter ranges and clinical safety.
   * @returns {{ valid: boolean, errors: string[] }}
   */
  validate() {
    const errors = [];
    if (this.alpha < 0 || this.alpha > 1) {
      errors.push('α (alpha) must be in [0, 1]');
    }
    if (this.tau < 0.5 || this.tau > 2.0) {
      errors.push('τ (tau) must be in [0.5, 2.0]');
    }
    if (this.theta < 0.3 || this.theta > 0.7) {
      errors.push('θ (theta) must be in [0.3, 0.7]');
    }
    if (this.theta > 0.50) {
      errors.push('Warning: θ > 0.50 increases false negatives — clinical risk');
    }
    return { valid: errors.filter(e => !e.startsWith('Warning')).length === 0, errors };
  }

  /** Serialises for API PUT request. */
  toPayload() {
    return { alpha: this.alpha, tau: this.tau, theta: this.theta };
  }

  /** @returns {GAParameters} */
  static defaults() {
    return new GAParameters({ alpha: 0.5467, tau: 0.7671, theta: 0.3985 });
  }

  /**
   * @param {object} data - API/fixture response
   * @returns {GAParameters}
   */
  static fromApiResponse(data) {
    return new GAParameters({
      alpha: data.alpha,
      tau: data.tau,
      theta: data.theta
    });
  }
}
