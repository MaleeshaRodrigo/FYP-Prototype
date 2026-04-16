import { BaseComponent } from './BaseComponent.js';
import { MetricsFormatter } from '../../domain/MetricsFormatter.js';

/**
 * Molecule component rendering a single row in the experiment version history table.
 */
export class ExperimentTableRow extends BaseComponent {
  /** @type {import('../../domain/ExperimentConfig.js').ExperimentConfig} */
  #config;

  /**
   * @param {HTMLElement} container
   * @param {import('../../domain/ExperimentConfig.js').ExperimentConfig} config
   */
  constructor(container, config) {
    super(container);
    this.#config = config;
  }

  render() {
    const c = this.#config;
    const cm = c.cleanMetrics;
    const advBal = c.advMetrics?.balancedAccuracy;
    const statusClass = `status--${c.status}`;

    return `
      <tr class="${c.rowCSSClass}">
        <td class="metric-value">${c.version}</td>
        <td class="metric-value">${c.advLossWeight.toFixed(2)}</td>
        <td class="metric-value">${MetricsFormatter.formatEpsilon(c.epsilon)}</td>
        <td class="metric-value">${c.pgdSteps}</td>
        <td class="metric-value">${MetricsFormatter.formatAUC(cm.auc)}</td>
        <td class="metric-value">${MetricsFormatter.formatPercent(cm.balancedAccuracy)}</td>
        <td class="metric-value">${advBal != null ? MetricsFormatter.formatPercent(advBal) : '—'}</td>
        <td><span class="badge ${statusClass}">${c.status}</span></td>
      </tr>
    `;
  }
}
