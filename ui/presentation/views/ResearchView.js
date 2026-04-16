import { BaseView } from './BaseView.js';
import { MetricsKPICard } from '../components/MetricsKPICard.js';
import { ExperimentTableRow } from '../components/ExperimentTableRow.js';
import { TradeoffChart } from '../components/TradeoffChart.js';
import { TRADESBetaSweepChart } from '../components/TRADESBetaSweepChart.js';
import { MetricsFormatter } from '../../domain/MetricsFormatter.js';
import { ClinicalMetrics } from '../../domain/ClinicalMetrics.js';
import { EventBus } from '../../application/EventBus.js';

/**
 * Research dashboard view: KPIs, experiment table, trade-off chart, TRADES panel, attack sim.
 */
export class ResearchView extends BaseView {
  render() {
    return `
      <div class="research-view">
        <header class="view-header">
          <h2 class="view-header__title">Research Dashboard</h2>
          <p class="view-header__subtitle">Adversarial Training Experiment Analysis</p>
          <div class="thesis-view__actions">
            <button class="btn btn--outline btn--sm" data-action="open-thesis">Thesis Hub</button>
            <button class="btn btn--outline btn--sm" data-action="logout">Sign Out</button>
          </div>
        </header>

        <section class="research-view__kpis" id="kpi-row" aria-label="Key Performance Indicators"></section>

        <section class="research-view__experiments" aria-label="Version History">
          <h3>Experiment Version History</h3>
          <div class="table-wrapper">
            <table class="experiment-table">
              <thead>
                <tr>
                  <th>Version</th><th>w_adv</th><th>ε</th><th>PGD Steps</th>
                  <th>Clean AUC</th><th>Bal. Acc.</th><th>Adv. Bal. Acc.</th><th>Status</th>
                </tr>
              </thead>
              <tbody id="experiment-tbody"></tbody>
            </table>
          </div>
        </section>

        <div class="research-view__charts-row">
          <section class="research-view__tradeoff" id="tradeoff-slot" aria-label="Trade-off Chart"></section>
          <section class="research-view__trades" id="trades-slot" aria-label="TRADES Beta Sweep"></section>
        </div>

        <section class="research-view__attack" aria-label="PGD Attack Simulator">
          <h3>PGD Attack Simulator</h3>
          <form class="attack-form" id="attack-form">
            <div class="attack-form__field attack-form__field--file">
              <label for="atk-image">Dermoscopic Image</label>
              <input type="file" id="atk-image" accept="image/*" class="input" required />
            </div>
            <div class="attack-form__field">
              <label for="atk-epsilon">ε (L∞ budget)</label>
              <input type="number" id="atk-epsilon" value="0.01" min="0.001" max="0.1" step="0.001" class="input" />
            </div>
            <div class="attack-form__field">
              <label for="atk-steps">PGD Steps</label>
              <input type="number" id="atk-steps" value="10" min="1" max="100" step="1" class="input" />
            </div>
            <div class="attack-form__field">
              <label for="atk-alpha">Step Size (α)</label>
              <input type="number" id="atk-alpha" value="0.003" min="0.001" max="0.01" step="0.001" class="input" />
            </div>
            <button type="submit" class="btn btn--primary">Run Attack</button>
          </form>
          <div id="attack-result" class="attack-result"></div>
        </section>
      </div>
    `;
  }

  _bindEvents() {
    const signal = this._abortController.signal;
    const form = this._container.querySelector('#attack-form');
    if (form) {
      form.addEventListener('submit', (e) => {
        e.preventDefault();
        const imageInput = this._container.querySelector('#atk-image');
        const imageFile = imageInput?.files?.[0] ?? null;
        const config = {
          epsilon: parseFloat(this._container.querySelector('#atk-epsilon').value),
          pgd_steps: parseInt(this._container.querySelector('#atk-steps').value, 10),
          pgd_alpha: parseFloat(this._container.querySelector('#atk-alpha').value)
        };
        this._controller?.runAttackSimulation(config, imageFile);
      }, { signal });
    }

    this._container.addEventListener('click', (e) => {
      if (e.target.closest('[data-action="open-thesis"]')) {
        EventBus.emit('navigate', { path: '#/thesis' });
        return;
      }
      if (e.target.closest('[data-action="logout"]')) {
        this._store.setState({ auth: { user: null, role: null, isAuthenticated: false } });
        EventBus.emit('auth:logout');
        EventBus.emit('navigate', { path: '#/login' });
      }
    }, { signal });
  }

  _subscribeToStore() {
    super._subscribeToStore();
    EventBus.on('attack:complete', ({ result }) => this._renderAttackResult(result));
  }

  _onStoreChange(state) {
    const { research } = state;
    if (research.metrics?.v8) {
      this._renderKPIs(research.metrics.v8);
    }
    if (research.versionHistory?.length) {
      this._renderExperimentTable(research.versionHistory);
      this._renderCharts(research.versionHistory, research.tradesBetaSweep);
    }
  }

  _renderKPIs(metrics) {
    const row = this._container.querySelector('#kpi-row');
    if (!row) return;
    row.innerHTML = '';

    const kpiData = [
      { label: MetricsFormatter.metricLabel('auc'), value: metrics.auc, target: ClinicalMetrics.TARGETS.auc },
      { label: MetricsFormatter.metricLabel('balancedAccuracy'), value: metrics.balancedAccuracy, target: ClinicalMetrics.TARGETS.balAcc, unit: '%' },
      { label: MetricsFormatter.metricLabel('melanomaSensitivity'), value: metrics.melanomaSensitivity, target: ClinicalMetrics.TARGETS.sens, unit: '%' },
      { label: MetricsFormatter.metricLabel('nonMelSpecificity'), value: metrics.nonMelSpecificity, target: ClinicalMetrics.TARGETS.spec, unit: '%' }
    ];

    for (const data of kpiData) {
      const slot = document.createElement('div');
      slot.className = 'kpi-slot';
      row.appendChild(slot);
      this._mountChild(new MetricsKPICard(slot, data));
    }
  }

  _renderExperimentTable(configs) {
    const tbody = this._container.querySelector('#experiment-tbody');
    if (!tbody) return;
    tbody.innerHTML = '';
    for (const config of configs) {
      const tr = document.createElement('tr');
      tbody.appendChild(tr);
      const row = new ExperimentTableRow(tr, config);
      row.mount();
      tbody.replaceChild(tr.querySelector('tr') ?? tr, tr);
    }
  }

  _renderCharts(configs, tradesBetaSweep) {
    const tradeoffSlot = this._container.querySelector('#tradeoff-slot');
    if (tradeoffSlot && !tradeoffSlot.querySelector('canvas')) {
      this._mountChild(new TradeoffChart(tradeoffSlot, configs));
    }
    const tradesSlot = this._container.querySelector('#trades-slot');
    if (tradesSlot && !tradesSlot.querySelector('canvas')) {
      this._mountChild(new TRADESBetaSweepChart(tradesSlot, tradesBetaSweep));
    }
  }

  _renderAttackResult(result) {
    const area = this._container.querySelector('#attack-result');
    if (!area) return;
    const success = result.attack_success ? 'Successful' : 'Failed';
    area.innerHTML = `
      <div class="attack-result__card">
        <h4>Attack Result: ${success}</h4>
        <div class="attack-result__grid">
          <div><span class="label">Original:</span> <span class="metric-value">${result.original_prediction} (${(result.original_confidence * 100).toFixed(1)}%)</span></div>
          <div><span class="label">Adversarial:</span> <span class="metric-value">${result.adversarial_prediction} (${(result.adversarial_confidence * 100).toFixed(1)}%)</span></div>
          <div><span class="label">ε:</span> <span class="metric-value">${result.epsilon}</span></div>
          <div><span class="label">Steps:</span> <span class="metric-value">${result.pgd_steps}</span></div>
        </div>
      </div>
    `;
  }
}
