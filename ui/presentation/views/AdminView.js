import { BaseView } from './BaseView.js';
import { AuditLogRow } from '../components/AuditLogRow.js';
import { EventBus } from '../../application/EventBus.js';

/**
 * Hospital admin view: usage stats, model health, audit log.
 */
export class AdminView extends BaseView {
  render() {
    return `
      <div class="admin-view">
        <header class="view-header">
          <h2 class="view-header__title">Hospital Admin Dashboard</h2>
          <p class="view-header__subtitle">Usage Analytics & Audit</p>
          <button class="btn btn--outline btn--sm" data-action="logout">Sign Out</button>
        </header>

        <section class="admin-view__stats" id="usage-stats-area" aria-label="Usage Statistics">
          <h3>Usage Statistics</h3>
          <div class="stats-grid" id="stats-grid"></div>
        </section>

        <section class="admin-view__audit" aria-label="Audit Log">
          <h3>Audit Log</h3>
          <div class="audit-controls">
            <select id="audit-filter" class="input input--select">
              <option value="">All Actions</option>
              <option value="model:activated">Model Activated</option>
              <option value="parameters:updated">Parameters Updated</option>
              <option value="scan:completed">Scan Completed</option>
            </select>
          </div>
          <div class="table-wrapper">
            <table class="audit-table">
              <thead>
                <tr>
                  <th>Timestamp</th><th>Action</th><th>User</th><th>Target</th><th>Details</th>
                </tr>
              </thead>
              <tbody id="audit-tbody"></tbody>
            </table>
          </div>
        </section>
      </div>
    `;
  }

  _bindEvents() {
    const signal = this._abortController.signal;

    this._container.querySelector('#audit-filter')?.addEventListener('change', (e) => {
      const filter = e.target.value;
      this._controller?.loadAuditLog(filter ? { action: filter } : {});
    }, { signal });

    this._container.addEventListener('click', (e) => {
      if (e.target.closest('[data-action="logout"]')) {
        this._store.setState({ auth: { user: null, role: null, isAuthenticated: false } });
        EventBus.emit('auth:logout');
        EventBus.emit('navigate', { path: '#/login' });
      }
    }, { signal });
  }

  _onStoreChange(state) {
    const { admin } = state;
    if (admin.usageStats) this._renderUsageStats(admin.usageStats);
    if (admin.auditLog?.length) this._renderAuditLog(admin.auditLog);
  }

  _renderUsageStats(stats) {
    const grid = this._container.querySelector('#stats-grid');
    if (!grid) return;
    grid.innerHTML = `
      <div class="stat-card">
        <div class="stat-card__value metric-value">${stats.total_scans?.toLocaleString() ?? '—'}</div>
        <div class="stat-card__label">Total Scans</div>
      </div>
      <div class="stat-card">
        <div class="stat-card__value metric-value">${stats.mel_detections ?? '—'}</div>
        <div class="stat-card__label">MEL Detections</div>
      </div>
      <div class="stat-card">
        <div class="stat-card__value metric-value">${stats.mel_rate != null ? (stats.mel_rate * 100).toFixed(1) + '%' : '—'}</div>
        <div class="stat-card__label">MEL Rate</div>
      </div>
      <div class="stat-card">
        <div class="stat-card__value metric-value">${stats.referral_rate != null ? (stats.referral_rate * 100).toFixed(1) + '%' : '—'}</div>
        <div class="stat-card__label">Referral Rate</div>
      </div>
      <div class="stat-card">
        <div class="stat-card__value metric-value">${stats.avg_inference_ms ?? '—'}ms</div>
        <div class="stat-card__label">Avg. Inference Time</div>
      </div>
    `;
  }

  _renderAuditLog(entries) {
    const tbody = this._container.querySelector('#audit-tbody');
    if (!tbody) return;
    tbody.innerHTML = '';
    for (const entry of entries) {
      const rowContainer = document.createElement('tbody');
      const row = new AuditLogRow(rowContainer, entry);
      row.mount();
      const tr = rowContainer.querySelector('tr');
      if (tr) tbody.appendChild(tr);
    }
  }
}
