import { EventBus } from './EventBus.js';

/**
 * Hash-based SPA router. Maps #/path to Controller + View pairs.
 */
export class Router {
  /** @type {Map<string, Object>} */ #routes = new Map();
  /** @type {import('./AuthGuard.js').AuthGuard} */ #authGuard;
  /** @type {Object|null} */ #currentRoute = null;
  /** @type {Object|null} */ #currentView = null;
  /** @type {Object|null} */ #currentController = null;

  /**
   * @param {import('./AuthGuard.js').AuthGuard} authGuard
   */
  constructor(authGuard) {
    this.#authGuard = authGuard;
    this._onHashChange = this._onHashChange.bind(this);
  }

  /**
   * @param {string} path - e.g. '#/clinical'
   * @param {Function} ViewClass
   * @param {Function|null} ControllerClass
   * @param {string|null} role - required role, or null for public routes
   */
  register(path, ViewClass, ControllerClass, role) {
    this.#routes.set(path, { ViewClass, ControllerClass, role });
  }

  /** Starts listening for hash changes and navigates to the current hash. */
  init() {
    window.addEventListener('hashchange', this._onHashChange);
    EventBus.on('navigate', ({ path }) => this.navigate(path));
    const initial = window.location.hash || '#/login';
    this.navigate(initial);
  }

  /** @param {string} path */
  navigate(path) {
    if (window.location.hash !== path) {
      window.location.hash = path;
      return;
    }
    this._onHashChange();
  }

  /** @private */
  _onHashChange() {
    const hash = window.location.hash || '#/login';
    const route = this.#routes.get(hash);

    if (!route) {
      this.navigate('#/login');
      return;
    }

    if (!this.#authGuard.canActivate(route.role)) {
      this.#authGuard.redirectToLogin();
      return;
    }

    this._teardownCurrentRoute();
    this._initRoute(hash, route);
  }

  /**
   * @param {string} path
   * @param {Object} route
   */
  async _initRoute(path, route) {
    const container = this._getContainer();
    this.#currentRoute = { path, ...route };

    const { ServiceLocator } = await import('../infrastructure/ServiceLocator.js');
    const store = ServiceLocator.resolve('store');
    const view = new route.ViewClass(container, store, EventBus);
    this.#currentView = view;

    if (route.ControllerClass) {
      const controller = route.ControllerClass.create(ServiceLocator);
      this.#currentController = controller;
      view.setController(controller);
      await controller.init();
    }

    view.mount();
    store.setState({ ui: { activeRoute: path } });
  }

  _teardownCurrentRoute() {
    if (this.#currentView) {
      this.#currentView.unmount();
      this.#currentView = null;
    }
    if (this.#currentController) {
      this.#currentController.destroy();
      this.#currentController = null;
    }
    this.#currentRoute = null;
  }

  /** @returns {HTMLElement} */
  _getContainer() {
    return document.getElementById('app-content');
  }

  destroy() {
    window.removeEventListener('hashchange', this._onHashChange);
    this._teardownCurrentRoute();
  }
}
