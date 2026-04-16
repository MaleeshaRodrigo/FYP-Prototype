import { BaseComponent } from './BaseComponent.js';
import { MetricsFormatter } from '../../domain/MetricsFormatter.js';

/**
 * Atomic component: animated probability bar with threshold marker.
 */
export class ConfidenceBar extends BaseComponent {
  #confidence;
  #threshold;

  /**
   * @param {HTMLElement} container
   * @param {{ confidence: number, threshold: number }} config
   */
  constructor(container, { confidence, threshold }) {
    super(container);
    this.#confidence = confidence;
    this.#threshold = threshold;
  }

  render() {
    const pct = (this.#confidence * 100).toFixed(1);
    const threshPct = (this.#threshold * 100).toFixed(1);
    const colorClass = this._getColorClass();

    return `
      <div class="confidence-bar" role="meter" aria-valuenow="${pct}" aria-valuemin="0" aria-valuemax="100" aria-label="Confidence: ${pct}%">
        <div class="confidence-bar__track">
          <div class="confidence-bar__fill confidence-bar__fill--${colorClass}" style="width: 0%" data-target-width="${pct}%"></div>
          <div class="confidence-bar__threshold" style="left: ${threshPct}%" data-tooltip="θ = ${this.#threshold}">
            <span class="confidence-bar__threshold-label">θ</span>
          </div>
        </div>
        <div class="confidence-bar__labels">
          <span class="confidence-bar__value metric-value">${MetricsFormatter.formatConfidence(this.#confidence)}</span>
          <span class="confidence-bar__threshold-value metric-value">θ = ${this.#threshold}</span>
        </div>
      </div>
    `;
  }

  mount() {
    super.mount();
    const fill = this._container.querySelector('.confidence-bar__fill');
    if (fill) {
      requestAnimationFrame(() => {
        fill.style.width = fill.dataset.targetWidth;
      });
    }
  }

  _getColorClass() {
    const margin = Math.abs(this.#confidence - this.#threshold);
    if (margin < 0.05) return 'borderline';
    return this.#confidence > this.#threshold ? 'positive' : 'negative';
  }
}
