import { EventBus } from './EventBus.js';

/**
 * Abstract controller providing common lifecycle methods.
 * Subclasses override init() and destroy().
 */
export class BaseController {
  /** @type {import('./StateStore.js').StateStore} */ _store;
  /** @type {Function[]} */ _subscriptions = [];

  /** @param {import('./StateStore.js').StateStore} store */
  constructor(store) {
    this._store = store;
  }

  /** Override in subclasses. Called by Router after view is ready. */
  async init() {}

  /**
   * Emits a normalised error event via EventBus.
   * @param {Error} err
   */
  handleError(err) {
    console.error(`[${this.constructor.name}]`, err);
    EventBus.emit('error', {
      code: err.name ?? 'UNKNOWN',
      message: err.message,
      context: err.originalError ?? null
    });
  }

  /**
   * Sets loading state for a given state slice.
   * @param {string} slice
   * @param {boolean} val
   */
  _setLoading(slice, val) {
    this._store.setState({ [slice]: { isLoading: val } });
  }

  /** Cleanup subscriptions. Override in subclasses to add custom teardown. */
  destroy() {
    for (const unsub of this._subscriptions) {
      unsub();
    }
    this._subscriptions = [];
  }
}
