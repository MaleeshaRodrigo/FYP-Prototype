import { BaseComponent } from './BaseComponent.js';
import { MetricsFormatter } from '../../domain/MetricsFormatter.js';

/**
 * Atomic component rendering a single metric KPI card.
 */
export class MetricsKPICard extends BaseComponent {
  #label;
  #value;
  #target;
  #delta;
  #unit;

  /**
   * @param {HTMLElement} container
   * @param {{ label: string, value: number|null, target: number, delta?: number|null, unit?: string }} config
   */
  constructor(container, { label, value, target, delta, unit }) {
    super(container);
    this.#label = label;
    this.#value = value;
    this.#target = target;
    this.#delta = delta ?? null;
    this.#unit = unit ?? '';
  }

  render() {
    const met = this.#value !== null && this.#value >= this.#target;
    const statusIcon = met ? '✅' : '❌';
    const formatted = this.#unit === '%'
      ? MetricsFormatter.formatPercent(this.#value)
      : MetricsFormatter.formatAUC(this.#value);
    const targetFormatted = this.#unit === '%'
      ? MetricsFormatter.formatPercent(this.#target)
      : MetricsFormatter.formatAUC(this.#target);
    const deltaHtml = this._renderDelta();

    return `
      <div class="kpi-card ${met ? 'kpi-card--met' : 'kpi-card--unmet'}" role="group" aria-label="${this.#label}">
        <div class="kpi-card__label">${this.#label}</div>
        <div class="kpi-card__value metric-value">${formatted}</div>
        <div class="kpi-card__target">
          <span class="kpi-card__target-label">Target: ${targetFormatted}</span>
          <span class="kpi-card__status">${statusIcon}</span>
        </div>
        ${deltaHtml}
      </div>
    `;
  }

  _renderDelta() {
    if (this.#delta === null || this.#delta === undefined) return '';
    const sign = this.#delta >= 0 ? '+' : '−';
    const cls = this.#delta >= 0 ? 'delta--positive' : 'delta--negative';
    return `<div class="kpi-card__delta ${cls} metric-value">${sign}${Math.abs(this.#delta).toFixed(3)}</div>`;
  }
}
