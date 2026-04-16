import { BaseView } from './BaseView.js';
import { GAParameterEditor } from '../components/GAParameterEditor.js';
import { GAParameters } from '../../domain/GAParameters.js';
import { RobustnessTier } from '../../domain/RobustnessTier.js';
import { EventBus } from '../../application/EventBus.js';

/**
 * System admin view: model registry, GA parameter editor, runtime health.
 */
export class SystemView extends BaseView {
  render() {
    return `
      <div class="system-view">
        <header class="view-header">
          <h2 class="view-header__title">System Administration</h2>
          <p class="view-header__subtitle">Model Management & Configuration</p>
          <button class="btn btn--outline btn--sm" data-action="logout">Sign Out</button>
        </header>

        <section class="system-view__registry" aria-label="Model Registry">
          <h3>Model Registry</h3>
          <div class="model-registry-grid" id="model-registry"></div>
        </section>

        <section class="system-view__params" id="ga-editor-slot" aria-label="GA Parameters"></section>

        <section class="system-view__health" aria-label="Runtime Health">
          <h3>Runtime Health</h3>
          <div class="health-grid" id="health-grid">
            <div class="health-card">
              <div class="health-card__label">GPU Utilisation</div>
              <div class="health-card__value metric-value">—</div>
            </div>
            <div class="health-card">
              <div class="health-card__label">CPU Usage</div>
              <div class="health-card__value metric-value">—</div>
            </div>
            <div class="health-card">
              <div class="health-card__label">Avg Latency</div>
              <div class="health-card__value metric-value">—</div>
            </div>
            <div class="health-card">
              <div class="health-card__label">Queue Depth</div>
              <div class="health-card__value metric-value">—</div>
            </div>
          </div>
        </section>
      </div>
    `;
  }

  _bindEvents() {
    const signal = this._abortController.signal;

    this._container.addEventListener('click', (e) => {
      const activateBtn = e.target.closest('[data-action="activate"]');
      if (activateBtn) {
        this._controller?.activateModel(activateBtn.dataset.version);
        return;
      }
      if (e.target.closest('[data-action="logout"]')) {
        this._store.setState({ auth: { user: null, role: null, isAuthenticated: false } });
        EventBus.emit('auth:logout');
        EventBus.emit('navigate', { path: '#/login' });
      }
    }, { signal });
  }

  _onStoreChange(state) {
    const { system } = state;
    if (system.modelRegistry?.length) {
      this._renderModelRegistry(system.modelRegistry);
    }
    this._ensureGAEditor(state);
  }

  _renderModelRegistry(models) {
    const grid = this._container.querySelector('#model-registry');
    if (!grid) return;
    grid.innerHTML = models.map(m => {
      const tier = RobustnessTier.fromModelVersion(m.id);
      const statusClass = m.isActive ? 'active' : m.isPending ? 'pending' : 'deprecated';
      return `
        <div class="model-card model-card--${statusClass}">
          <div class="model-card__header">
            <h4 class="model-card__id">${m.id}</h4>
            <span class="badge badge--${statusClass}">${statusClass}</span>
          </div>
          <p class="model-card__label">${m.label}</p>
          <p class="model-card__checkpoint metric-value">${m.checkpoint}</p>
          <div class="model-card__tier">${tier.icon} ${tier.label}</div>
          <div class="model-card__actions">
            ${!m.isActive ? `<button class="btn btn--primary btn--sm" data-action="activate" data-version="${m.id}">Activate</button>` : '<span class="model-card__active-label">Currently Active</span>'}
          </div>
        </div>
      `;
    }).join('');
  }

  _ensureGAEditor(state) {
    const slot = this._container.querySelector('#ga-editor-slot');
    if (!slot || slot.querySelector('.ga-editor')) return;

    const params = state.system.gaParameters
      ? GAParameters.fromApiResponse(state.system.gaParameters)
      : GAParameters.defaults();

    const editor = new GAParameterEditor(slot, params, (updated) => {
      this._controller?.updateGAParameters(updated);
    });
    this._mountChild(editor);
  }
}
