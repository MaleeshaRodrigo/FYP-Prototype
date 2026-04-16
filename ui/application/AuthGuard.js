import { EventBus } from './EventBus.js';

/**
 * Route protection. Checks user role before allowing navigation.
 */
export class AuthGuard {
  static ROLES = {
    CLINICIAN: 'clinician',
    RESEARCH: 'research',
    ADMIN: 'admin',
    SYSTEM: 'system'
  };

  /** @type {import('./StateStore.js').StateStore} */
  #store;

  /** @param {import('./StateStore.js').StateStore} store */
  constructor(store) {
    this.#store = store;
  }

  /**
   * @param {string|null} requiredRole - null means no auth required
   * @returns {boolean}
   */
  canActivate(requiredRole) {
    if (!requiredRole) return true;
    const { auth } = this.#store.getState();
    if (!auth.isAuthenticated) return false;
    return auth.role === requiredRole;
  }

  redirectToLogin() {
    EventBus.emit('navigate', { path: '#/login' });
  }
}
