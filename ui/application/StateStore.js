/**
 * Observer-pattern central application state.
 * No global variables — instance injected via ServiceLocator.
 */
export class StateStore {
  /** @type {Object} */ #state;
  /** @type {Set<Function>} */ #listeners;

  /** @param {Object} [initialState] */
  constructor(initialState) {
    this.#state = this._freeze(initialState ?? StateStore.defaultState());
    this.#listeners = new Set();
  }

  static defaultState() {
    return {
      auth: { user: null, role: null, isAuthenticated: false },
      clinical: { currentResult: null, caseHistory: [], isLoading: false },
      research: {
        metrics: {},
        versionHistory: [],
        thesisSummary: null,
        thesisSweep: null,
        tradesBetaSweep: null,
        comparison: null,
        attackComparison: null,
        isLoading: false
      },
      admin: { usageStats: null, auditLog: [], modelHealth: null },
      system: { modelRegistry: [], gaParameters: null, runtimeHealth: null },
      ui: { activeRoute: null, error: null, notification: null }
    };
  }

  /** @returns {Object} deep-frozen copy */
  getState() {
    return this.#state;
  }

  /**
   * Shallow-merges partial state then notifies all listeners.
   * @param {Object} partial - top-level keys to merge
   */
  setState(partial) {
    const next = { ...this.#state };
    for (const [key, value] of Object.entries(partial)) {
      if (typeof value === 'object' && value !== null && !Array.isArray(value)) {
        next[key] = { ...this.#state[key], ...value };
      } else {
        next[key] = value;
      }
    }
    this.#state = this._freeze(next);
    this._notify();
  }

  /**
   * @param {Function} listener
   * @returns {Function} unsubscribe
   */
  subscribe(listener) {
    this.#listeners.add(listener);
    return () => this.#listeners.delete(listener);
  }

  /** @private */
  _notify() {
    for (const listener of this.#listeners) {
      try {
        listener(this.#state);
      } catch (err) {
        console.error('StateStore listener error:', err);
      }
    }
  }

  /**
   * Deep-freezes an object to prevent mutation.
   * @param {Object} obj
   * @returns {Object}
   */
  _freeze(obj) {
    if (obj === null || typeof obj !== 'object') return obj;
    Object.freeze(obj);
    for (const value of Object.values(obj)) {
      if (typeof value === 'object' && value !== null) {
        this._freeze(value);
      }
    }
    return obj;
  }
}
