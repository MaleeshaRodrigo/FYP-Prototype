import { BaseComponent } from './BaseComponent.js';
import { GAParameters } from '../../domain/GAParameters.js';

/**
 * Molecule component: three sliders for GA-optimised parameters (α, τ, θ)
 * with real-time validation and clinical safety warnings.
 */
export class GAParameterEditor extends BaseComponent {
  /** @type {GAParameters} */ #params;
  /** @type {Function} */ #onSave;

  /**
   * @param {HTMLElement} container
   * @param {GAParameters} params
   * @param {Function} onSave - callback receiving updated GAParameters
   */
  constructor(container, params, onSave) {
    super(container);
    this.#params = params;
    this.#onSave = onSave;
  }

  render() {
    const p = this.#params;
    return `
      <div class="ga-editor" role="form" aria-label="GA Parameter Editor">
        <h4 class="ga-editor__title">GA-Optimised Parameters</h4>

        <div class="ga-editor__field">
          <label for="ga-alpha">α (CNN Fusion Weight)</label>
          <div class="ga-editor__slider-row">
            <input type="range" id="ga-alpha" min="0" max="1" step="0.001" value="${p.alpha}" />
            <span class="ga-editor__value metric-value" data-for="alpha">${p.alpha.toFixed(4)}</span>
          </div>
          <small class="ga-editor__hint">Controls CNN vs ViT weight in ensemble fusion</small>
        </div>

        <div class="ga-editor__field">
          <label for="ga-tau">τ (Temperature Scaling)</label>
          <div class="ga-editor__slider-row">
            <input type="range" id="ga-tau" min="0.5" max="2.0" step="0.001" value="${p.tau}" />
            <span class="ga-editor__value metric-value" data-for="tau">${p.tau.toFixed(4)}</span>
          </div>
          <small class="ga-editor__hint">Calibrates softmax confidence distribution</small>
        </div>

        <div class="ga-editor__field">
          <label for="ga-theta">θ (Decision Threshold)</label>
          <div class="ga-editor__slider-row">
            <input type="range" id="ga-theta" min="0.3" max="0.7" step="0.001" value="${p.theta}" />
            <span class="ga-editor__value metric-value" data-for="theta">${p.theta.toFixed(4)}</span>
          </div>
          <small class="ga-editor__hint">MEL/Non-MEL decision boundary</small>
        </div>

        <div class="ga-editor__warnings" id="ga-warnings" role="alert"></div>

        <div class="ga-editor__actions">
          <button class="btn btn--outline" data-action="reset">Reset to Defaults</button>
          <button class="btn btn--primary" data-action="save" id="ga-save-btn">Save Parameters</button>
        </div>
      </div>
    `;
  }

  _bindEvents() {
    const signal = this._abortController.signal;
    const sliders = this._container.querySelectorAll('input[type="range"]');
    sliders.forEach(slider => {
      slider.addEventListener('input', () => this._onSliderChange(), { signal });
    });

    this._container.addEventListener('click', (e) => {
      const btn = e.target.closest('[data-action]');
      if (!btn) return;
      if (btn.dataset.action === 'save') this._handleSave();
      if (btn.dataset.action === 'reset') this._handleReset();
    }, { signal });
  }

  _onSliderChange() {
    const alpha = parseFloat(this._container.querySelector('#ga-alpha').value);
    const tau = parseFloat(this._container.querySelector('#ga-tau').value);
    const theta = parseFloat(this._container.querySelector('#ga-theta').value);

    this._updateDisplay('alpha', alpha.toFixed(4));
    this._updateDisplay('tau', tau.toFixed(4));
    this._updateDisplay('theta', theta.toFixed(4));

    this.#params = new GAParameters({ alpha, tau, theta });
    this._validateAndShowWarnings();
  }

  _updateDisplay(field, value) {
    const el = this._container.querySelector(`[data-for="${field}"]`);
    if (el) el.textContent = value;
  }

  _validateAndShowWarnings() {
    const { errors } = this.#params.validate();
    const warningsEl = this._container.querySelector('#ga-warnings');
    const saveBtn = this._container.querySelector('#ga-save-btn');
    const hasBlockingErrors = errors.filter(e => !e.startsWith('Warning')).length > 0;

    if (warningsEl) {
      warningsEl.innerHTML = errors
        .map(e => `<div class="ga-editor__warning ${e.startsWith('Warning') ? 'warning' : 'error'}">${e}</div>`)
        .join('');
    }
    if (saveBtn) saveBtn.disabled = hasBlockingErrors;
  }

  _handleSave() {
    const { valid } = this.#params.validate();
    if (valid && this.#onSave) {
      this.#onSave(this.#params);
    }
  }

  _handleReset() {
    this.#params = GAParameters.defaults();
    this.update();
  }
}
