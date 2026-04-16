import { BaseComponent } from './BaseComponent.js';

/**
 * Molecule component: Chart.js scatter plot of w_adv vs AUC + adv_bal_acc.
 * Visualises the adversarial trade-off with the phase-transition boundary.
 */
export class TradeoffChart extends BaseComponent {
  #experiments;
  #chart = null;

  /**
   * @param {HTMLElement} container
   * @param {Array<import('../../domain/ExperimentConfig.js').ExperimentConfig>} experiments
   */
  constructor(container, experiments) {
    super(container);
    this.#experiments = experiments;
  }

  render() {
    return `
      <div class="tradeoff-chart">
        <h4 class="tradeoff-chart__title">Adversarial Trade-off: w_adv vs Performance</h4>
        <canvas id="tradeoff-canvas" width="600" height="350"></canvas>
      </div>
    `;
  }

  mount() {
    super.mount();
    this._initChart();
  }

  unmount() {
    if (this.#chart) {
      this.#chart.destroy();
      this.#chart = null;
    }
    super.unmount();
  }

  _initChart() {
    const canvas = this._container.querySelector('#tradeoff-canvas');
    if (!canvas || typeof Chart === 'undefined') return;

    const cleanData = this.#experiments.map(e => ({
      x: e.advLossWeight,
      y: e.cleanMetrics.auc
    })).filter(d => d.y !== null);

    const advData = this.#experiments
      .filter(e => e.advMetrics?.balancedAccuracy != null)
      .map(e => ({
        x: e.advLossWeight,
        y: e.advMetrics.balancedAccuracy
      }));

    this.#chart = new Chart(canvas, {
      type: 'scatter',
      data: {
        datasets: [
          {
            label: 'Clean AUC',
            data: cleanData,
            backgroundColor: 'rgba(1, 105, 111, 0.8)',
            borderColor: '#01696f',
            pointRadius: 6,
            showLine: true
          },
          {
            label: 'Adversarial Bal. Acc.',
            data: advData,
            backgroundColor: 'rgba(161, 44, 123, 0.8)',
            borderColor: '#a12c7b',
            pointRadius: 6,
            pointStyle: 'triangle',
            showLine: true
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: {
            title: { display: true, text: 'Adversarial Loss Weight (w_adv)' },
            min: 0, max: 0.45
          },
          y: {
            title: { display: true, text: 'Metric Value' },
            min: 0, max: 1.0
          }
        },
        plugins: {
          annotation: {
            annotations: {
              phaseTransition: {
                type: 'line',
                xMin: 0.225, xMax: 0.225,
                borderColor: 'rgba(150, 66, 25, 0.7)',
                borderDash: [6, 4],
                borderWidth: 2,
                label: {
                  display: true,
                  content: 'Phase Transition',
                  position: 'start'
                }
              }
            }
          }
        }
      }
    });
  }
}
