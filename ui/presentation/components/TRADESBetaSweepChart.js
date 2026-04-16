import { BaseComponent } from './BaseComponent.js';

/**
 * Molecule component: Chart.js line chart for TRADES beta sweep.
 * Shows expected Clean AUC + adv_bal_acc across beta values.
 */
export class TRADESBetaSweepChart extends BaseComponent {
  #betaResults;
  #chart = null;

  static DEFAULT_DATA = [
    { beta: 1.0, cleanAUC: 0.865, advBalAcc: 0.425 },
    { beta: 2.0, cleanAUC: 0.850, advBalAcc: 0.550 },
    { beta: 3.0, cleanAUC: 0.835, advBalAcc: 0.650 },
    { beta: 6.0, cleanAUC: 0.800, advBalAcc: 0.715 }
  ];

  /**
   * @param {HTMLElement} container
   * @param {Array} [betaResults]
   */
  constructor(container, betaResults) {
    super(container);
    this.#betaResults = betaResults?.length ? betaResults : TRADESBetaSweepChart.DEFAULT_DATA;
  }

  render() {
    return `
      <div class="trades-chart">
        <div class="trades-chart__header">
          <h4 class="trades-chart__title">TRADES β Sweep — Expected Performance</h4>
          <span class="badge badge--pending">v9-PENDING</span>
        </div>
        <canvas id="trades-canvas" width="600" height="350"></canvas>
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
    const canvas = this._container.querySelector('#trades-canvas');
    if (!canvas || typeof Chart === 'undefined') return;

    const labels = this.#betaResults.map(d => d.beta.toString());

    this.#chart = new Chart(canvas, {
      type: 'line',
      data: {
        labels,
        datasets: [
          {
            label: 'Clean AUC',
            data: this.#betaResults.map(d => d.cleanAUC),
            borderColor: '#01696f',
            backgroundColor: 'rgba(1, 105, 111, 0.1)',
            tension: 0.3,
            fill: true
          },
          {
            label: 'Adversarial Bal. Acc.',
            data: this.#betaResults.map(d => d.advBalAcc),
            borderColor: '#a12c7b',
            backgroundColor: 'rgba(161, 44, 123, 0.1)',
            tension: 0.3,
            fill: true
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { title: { display: true, text: 'β (TRADES regularisation)' } },
          y: { title: { display: true, text: 'Metric Value' }, min: 0, max: 1.0 }
        },
        plugins: {
          annotation: {
            annotations: {
              targetLine: {
                type: 'line',
                yMin: 0.65, yMax: 0.65,
                borderColor: 'rgba(150, 66, 25, 0.6)',
                borderDash: [6, 4],
                borderWidth: 2,
                label: {
                  display: true,
                  content: 'adv_bal_acc target = 0.65',
                  position: 'end'
                }
              }
            }
          }
        }
      }
    });
  }
}
