import { BaseComponent } from './BaseComponent.js';
import { RobustnessBadge } from './RobustnessBadge.js';
import { ConfidenceBar } from './ConfidenceBar.js';
import { MetricsFormatter } from '../../domain/MetricsFormatter.js';
import { EventBus } from '../../application/EventBus.js';

/**
 * Molecule component displaying a complete HARE diagnosis result.
 * Enforces clinical safety constraints: borderline banner, model version visibility, GradCAM trigger.
 */
export class DiagnosisResultCard extends BaseComponent {
  /** @type {import('../../domain/DiagnosisResult.js').DiagnosisResult} */
  #result;

  /**
   * @param {HTMLElement} container
   * @param {import('../../domain/DiagnosisResult.js').DiagnosisResult} result
   * @param {import('../../application/StateStore.js').StateStore} store
   */
  constructor(container, result, store) {
    super(container, store);
    this.#result = result;
  }

  render() {
    const r = this.#result;
    const predClass = r.isMelanoma ? 'prediction--mel' : 'prediction--non-mel';
    const predLabel = r.isMelanoma ? 'Melanoma (MEL)' : 'Non-Melanoma';

    return `
      <div class="diagnosis-result-card ${predClass}" role="region" aria-label="Diagnosis Result">
        ${this._renderBorderlineBanner()}
        <div class="diagnosis-result-card__header">
          <h3 class="diagnosis-result-card__prediction">${predLabel}</h3>
          <div class="diagnosis-result-card__badges">
            <span class="diagnosis-result-card__version-chip">${r.modelVersion.displayLabel}</span>
            <div id="robustness-badge-slot"></div>
          </div>
        </div>

        <div class="diagnosis-result-card__confidence" id="confidence-bar-slot"></div>

        <div class="diagnosis-result-card__meta">
          <span class="diagnosis-result-card__timestamp">${r.timestamp.toLocaleString()}</span>
          <span class="diagnosis-result-card__inference metric-value">${r.formattedConfidence} confidence</span>
        </div>

        <div class="diagnosis-result-card__actions">
          ${r.requiresGradCAM ? '<button class="btn btn--primary" data-action="gradcam">View GradCAM Heatmap</button>' : ''}
          <button class="btn btn--outline" data-action="export">Export Case</button>
          <button class="btn btn--outline" data-action="referral">Flag for Referral</button>
        </div>
      </div>
    `;
  }

  _renderBorderlineBanner() {
    if (!this.#result.isBorderline) return '';
    return `
      <div class="diagnosis-result-card__borderline-banner" role="alert">
        <strong>⚠ Borderline — Recommend Specialist Review</strong>
        <p>Confidence is within ±5% of the decision threshold (θ = ${this.#result.threshold}).</p>
      </div>
    `;
  }

  mount() {
    super.mount();
    this._mountChildComponents();
  }

  _mountChildComponents() {
    const badgeSlot = this._container.querySelector('#robustness-badge-slot');
    if (badgeSlot) {
      new RobustnessBadge(badgeSlot, this.#result.modelVersion.robustnessTier).mount();
    }
    const barSlot = this._container.querySelector('#confidence-bar-slot');
    if (barSlot) {
      new ConfidenceBar(barSlot, {
        confidence: this.#result.confidence,
        threshold: this.#result.threshold
      }).mount();
    }
  }

  _bindEvents() {
    const signal = this._abortController.signal;
    this._container.addEventListener('click', (e) => {
      const btn = e.target.closest('[data-action]');
      if (!btn) return;
      const action = btn.dataset.action;
      if (action === 'gradcam') {
        EventBus.emit('gradcam:requested', { imageId: this.#result.imageId });
      } else if (action === 'export') {
        EventBus.emit('export:requested', { imageId: this.#result.imageId });
      } else if (action === 'referral') {
        EventBus.emit('referral:flagged', { result: this.#result });
      }
    }, { signal });
  }
}
