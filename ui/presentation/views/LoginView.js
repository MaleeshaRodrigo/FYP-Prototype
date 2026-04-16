import { BaseView } from './BaseView.js';
import { EventBus } from '../../application/EventBus.js';

/**
 * Login/role-selection view for the demo environment.
 */
export class LoginView extends BaseView {
  render() {
    return `
      <div class="login-view">
        <div class="login-view__card">
          <div class="login-view__header">
            <h1 class="login-view__title">HARE Platform</h1>
            <p class="login-view__subtitle">Hybrid Adversarially Robust Ensemble</p>
            <p class="login-view__desc">Adversarial-Robust Skin Cancer Detection</p>
          </div>

          <div class="login-view__roles">
            <h2 class="login-view__roles-title">Select Your Role</h2>
            <div class="login-view__role-grid">
              <button class="login-view__role-btn" data-role="clinician">
                <span class="login-view__role-icon">🔬</span>
                <span class="login-view__role-label">Clinician</span>
                <span class="login-view__role-desc">Upload images, view diagnoses</span>
              </button>
              <button class="login-view__role-btn" data-role="research">
                <span class="login-view__role-icon">📊</span>
                <span class="login-view__role-label">Research Engineer</span>
                <span class="login-view__role-desc">Experiment history, attack simulation</span>
              </button>
              <button class="login-view__role-btn" data-role="admin">
                <span class="login-view__role-icon">🏥</span>
                <span class="login-view__role-label">Hospital Admin</span>
                <span class="login-view__role-desc">Usage stats, audit log</span>
              </button>
              <button class="login-view__role-btn" data-role="system">
                <span class="login-view__role-icon">⚙️</span>
                <span class="login-view__role-label">System Admin</span>
                <span class="login-view__role-desc">Model management, parameters</span>
              </button>
            </div>
          </div>
        </div>
      </div>
    `;
  }

  _bindEvents() {
    const signal = this._abortController.signal;
    this._container.addEventListener('click', (e) => {
      const btn = e.target.closest('[data-role]');
      if (!btn) return;
      const role = btn.dataset.role;
      this._login(role);
    }, { signal });
  }

  _login(role) {
    const { ServiceLocator } = this._getServiceLocator();
    const store = ServiceLocator.resolve('store');
    store.setState({
      auth: { user: `${role}@hare.med`, role, isAuthenticated: true }
    });
    EventBus.emit('auth:login', { user: `${role}@hare.med`, role });

    const routeMap = {
      clinician: '#/clinical',
      research: '#/thesis',
      admin: '#/admin',
      system: '#/system'
    };
    EventBus.emit('navigate', { path: routeMap[role] });
  }

  _getServiceLocator() {
    return { ServiceLocator: window.__ServiceLocator };
  }
}
