import { BaseView } from './BaseView.js';
import { TradeoffChart } from '../components/TradeoffChart.js';
import { EpsilonSweepChart } from '../components/EpsilonSweepChart.js';
import { TRADESBetaSweepChart } from '../components/TRADESBetaSweepChart.js';
import { MetricsFormatter } from '../../domain/MetricsFormatter.js';
import { EventBus } from '../../application/EventBus.js';

/**
 * Thesis-focused route to present canonical results and trade-offs.
 */
export class ThesisView extends BaseView {
  render() {
    return `
      <div class="thesis-view">
        <header class="view-header thesis-view__header">
          <h2 class="view-header__title">Thesis Results Hub</h2>
          <p class="view-header__subtitle">Stage 1 to Stage 2 progression and safety-focused robustness summary</p>
          <div class="thesis-view__actions">
            <button class="btn btn--outline btn--sm" data-action="open-research">Research Dashboard</button>
            <button class="btn btn--outline btn--sm" data-action="logout">Sign Out</button>
          </div>
        </header>

        <section class="thesis-view__summary" aria-label="Canonical Thesis Summary">
          <h3>Canonical Evaluation Blocks</h3>
          <div class="table-wrapper">
            <table class="experiment-table">
              <thead>
                <tr>
                  <th>Block</th><th>Threshold</th><th>AUC</th><th>Bal. Acc.</th><th>MEL Sens.</th><th>Non-MEL Spec.</th>
                </tr>
              </thead>
              <tbody id="thesis-summary-tbody"></tbody>
            </table>
          </div>
        </section>

        <section class="thesis-view__delta" aria-label="Stage Delta" id="thesis-delta"></section>

        <section class="thesis-view__delta" aria-label="Attack Delta" id="thesis-attack-delta"></section>

        <section class="thesis-view__exports" aria-label="Thesis Export">
          <h3>Thesis Export</h3>
          <div class="thesis-view__actions">
            <button class="btn btn--primary btn--sm" data-action="export-json">Download JSON</button>
            <button class="btn btn--outline btn--sm" data-action="export-csv">Download CSV</button>
          </div>
        </section>

        <div class="thesis-view__charts-row">
          <section class="thesis-view__chart-slot" id="thesis-tradeoff-slot" aria-label="Trade-off"></section>
          <section class="thesis-view__chart-slot" id="thesis-sweep-slot" aria-label="Epsilon Sweep"></section>
        </div>

        <div class="thesis-view__charts-row thesis-view__charts-row--single">
          <section class="thesis-view__chart-slot" id="thesis-trades-beta-slot" aria-label="TRADES Beta Sweep"></section>
        </div>
      </div>
    `;
  }

  _bindEvents() {
    const signal = this._abortController.signal;
    this._container.addEventListener('click', (e) => {
      if (e.target.closest('[data-action="logout"]')) {
        this._store.setState({ auth: { user: null, role: null, isAuthenticated: false } });
        EventBus.emit('auth:logout');
        EventBus.emit('navigate', { path: '#/login' });
        return;
      }
      if (e.target.closest('[data-action="open-research"]')) {
        EventBus.emit('navigate', { path: '#/research' });
        return;
      }
      if (e.target.closest('[data-action="export-json"]')) {
        this._exportJson();
        return;
      }
      if (e.target.closest('[data-action="export-csv"]')) {
        this._exportCsv();
      }
    }, { signal });
  }

  _onStoreChange(state) {
    const { research } = state;
    if (research.thesisSummary) {
      this._renderSummaryTable(research.thesisSummary);
    }
    if (research.comparison) {
      this._renderDelta(research.comparison);
    }
    if (research.attackComparison) {
      this._renderAttackDelta(research.attackComparison);
    }
    if (research.versionHistory?.length) {
      this._renderTradeoff(research.versionHistory);
    }
    if (research.thesisSweep) {
      this._renderSweep(research.thesisSweep);
    }
    if (research.tradesBetaSweep?.length) {
      this._renderTradesBetaSweep(research.tradesBetaSweep);
    }
  }

  _renderSummaryTable(summary) {
    const tbody = this._container.querySelector('#thesis-summary-tbody');
    if (!tbody) return;

    const rows = [
      ['Stage 1 Clean', 'Default (0.5)', summary.stage1_clean_default],
      ['Stage 2 Clean', 'Default (0.5)', summary.stage2_clean_default],
      ['Stage 2 Clean', 'GA (theta)', summary.stage2_clean_ga],
      ['Stage 2 Adversarial', 'GA (theta) - Primary', summary.stage2_adv_ga]
    ];

    tbody.innerHTML = rows.map(([label, threshold, m]) => {
      const auc = MetricsFormatter.formatAUC(m?.auc ?? null);
      const bal = MetricsFormatter.formatPercent(m?.balancedAccuracy ?? null);
      const sens = MetricsFormatter.formatPercent(m?.melanomaSensitivity ?? null);
      const spec = MetricsFormatter.formatPercent(m?.nonMelSpecificity ?? null);
      return `
        <tr>
          <td>${label}</td>
          <td>${threshold}</td>
          <td class="metric-value">${auc}</td>
          <td class="metric-value">${bal}</td>
          <td class="metric-value">${sens}</td>
          <td class="metric-value">${spec}</td>
        </tr>
      `;
    }).join('');
  }

  _renderDelta(comparison) {
    const slot = this._container.querySelector('#thesis-delta');
    if (!slot) return;

    const delta = comparison.delta ?? {};
    const formatDelta = (v) => {
      if (v === null || v === undefined) return '—';
      const num = Number(v);
      const sign = num >= 0 ? '+' : '−';
      return `${sign}${Math.abs(num).toFixed(4)}`;
    };

    slot.innerHTML = `
      <h3>Stage Delta: ${comparison.baselineVersion} -> ${comparison.candidateVersion}</h3>
      <div class="stats-grid">
        <div class="stat-card">
          <div class="stat-card__value metric-value">${formatDelta(delta.auc)}</div>
          <div class="stat-card__label">AUC Delta</div>
        </div>
        <div class="stat-card">
          <div class="stat-card__value metric-value">${formatDelta(delta.balancedAccuracy)}</div>
          <div class="stat-card__label">Bal. Acc. Delta</div>
        </div>
        <div class="stat-card">
          <div class="stat-card__value metric-value">${formatDelta(delta.melanomaSensitivity)}</div>
          <div class="stat-card__label">MEL Sens. Delta</div>
        </div>
        <div class="stat-card">
          <div class="stat-card__value metric-value">${formatDelta(delta.nonMelSpecificity)}</div>
          <div class="stat-card__label">Non-MEL Spec. Delta</div>
        </div>
      </div>
    `;
  }

  _renderAttackDelta(comparison) {
    const slot = this._container.querySelector('#thesis-attack-delta');
    if (!slot) return;

    const delta = comparison.delta ?? {};
    const formatDelta = (v) => {
      if (v === null || v === undefined) return '—';
      const num = Number(v);
      const sign = num >= 0 ? '+' : '−';
      return `${sign}${Math.abs(num).toFixed(4)}`;
    };

    slot.innerHTML = `
      <h3>PGD vs Clean Delta: ${comparison.baselineVersion} -> ${comparison.candidateVersion}</h3>
      <div class="stats-grid">
        <div class="stat-card">
          <div class="stat-card__value metric-value">${formatDelta(delta.auc)}</div>
          <div class="stat-card__label">AUC Shift</div>
        </div>
        <div class="stat-card">
          <div class="stat-card__value metric-value">${formatDelta(delta.balancedAccuracy)}</div>
          <div class="stat-card__label">Bal. Acc. Shift</div>
        </div>
        <div class="stat-card">
          <div class="stat-card__value metric-value">${formatDelta(delta.melanomaSensitivity)}</div>
          <div class="stat-card__label">MEL Sens. Shift</div>
        </div>
        <div class="stat-card">
          <div class="stat-card__value metric-value">${formatDelta(delta.nonMelSpecificity)}</div>
          <div class="stat-card__label">Spec. Shift</div>
        </div>
      </div>
    `;
  }

  _renderTradeoff(configs) {
    const slot = this._container.querySelector('#thesis-tradeoff-slot');
    if (slot && !slot.querySelector('canvas')) {
      this._mountChild(new TradeoffChart(slot, configs));
    }
  }

  _renderSweep(sweep) {
    const slot = this._container.querySelector('#thesis-sweep-slot');
    if (slot && !slot.querySelector('canvas')) {
      this._mountChild(new EpsilonSweepChart(slot, sweep));
    }
  }

  _renderTradesBetaSweep(tradesBetaSweep) {
    const slot = this._container.querySelector('#thesis-trades-beta-slot');
    if (slot && !slot.querySelector('canvas')) {
      this._mountChild(new TRADESBetaSweepChart(slot, tradesBetaSweep));
    }
  }

  async _exportJson() {
    try {
      const payload = await this._controller?.getThesisExportJson();
      if (!payload) return;
      const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
      this._downloadBlob(blob, 'hare-thesis-export.json');
      EventBus.emit('notification', { type: 'success', message: 'JSON export downloaded' });
    } catch (err) {
      this._controller?.handleError(err);
    }
  }

  async _exportCsv() {
    try {
      const csv = await this._controller?.getThesisExportCsv();
      if (!csv) return;
      const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' });
      this._downloadBlob(blob, 'hare-thesis-summary.csv');
      EventBus.emit('notification', { type: 'success', message: 'CSV export downloaded' });
    } catch (err) {
      this._controller?.handleError(err);
    }
  }

  _downloadBlob(blob, filename) {
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = filename;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    URL.revokeObjectURL(url);
  }
}