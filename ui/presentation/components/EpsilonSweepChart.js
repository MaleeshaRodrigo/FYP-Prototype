import { BaseComponent } from './BaseComponent.js';

/**
 * Epsilon robustness sweep chart for thesis reporting.
 */
export class EpsilonSweepChart extends BaseComponent {
  #sweep;
  #chart = null;

  constructor(container, sweep) {
    super(container);
    this.#sweep = sweep ?? {};
  }

  render() {
    return `
      <div class="tradeoff-chart">
        <h4 class="tradeoff-chart__title">Stage 2 + GA Robustness Sweep</h4>
        <canvas id="epsilon-sweep-canvas" width="600" height="350"></canvas>
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
    const canvas = this._container.querySelector('#epsilon-sweep-canvas');
    if (!canvas || typeof Chart === 'undefined') return;

    const points = Object.entries(this.#sweep)
      .map(([eps, metrics]) => ({ eps: parseFloat(eps), ...metrics }))
      .filter(item => Number.isFinite(item.eps))
      .sort((a, b) => a.eps - b.eps);

    const labels = points.map(p => p.eps.toFixed(2));

    this.#chart = new Chart(canvas, {
      type: 'line',
      data: {
        labels,
        datasets: [
          {
            label: 'Balanced Accuracy',
            data: points.map(p => p.bal_acc),
            borderColor: '#01696f',
            backgroundColor: 'rgba(1, 105, 111, 0.12)',
            tension: 0.25,
            fill: false
          },
          {
            label: 'MEL Sensitivity',
            data: points.map(p => p.sens_mel),
            borderColor: '#a12c7b',
            backgroundColor: 'rgba(161, 44, 123, 0.12)',
            tension: 0.25,
            fill: false
          },
          {
            label: 'Non-MEL Specificity',
            data: points.map(p => p.spec_nonmel),
            borderColor: '#437a22',
            backgroundColor: 'rgba(67, 122, 34, 0.12)',
            tension: 0.25,
            fill: false
          },
          {
            label: 'AUC',
            data: points.map(p => p.auc),
            borderColor: '#964219',
            backgroundColor: 'rgba(150, 66, 25, 0.12)',
            tension: 0.25,
            borderDash: [5, 4],
            fill: false
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: true,
        aspectRatio: 1.9,
        scales: {
          x: { title: { display: true, text: 'Epsilon (Linf)' } },
          y: { title: { display: true, text: 'Metric Value' }, min: 0, max: 1 }
        },
        plugins: {
          annotation: {
            annotations: {
              trainEpsilon: {
                type: 'line',
                xMin: '0.03',
                xMax: '0.03',
                borderColor: 'rgba(40, 37, 29, 0.5)',
                borderDash: [6, 4],
                borderWidth: 2,
                label: {
                  display: true,
                  content: 'Training epsilon = 0.03',
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