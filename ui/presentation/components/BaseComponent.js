/**
 * Abstract base for all UI components.
 * Implements the Template Method pattern for component lifecycle.
 */
export class BaseComponent {
  /** @type {HTMLElement} */ _container;
  /** @type {import('../../application/StateStore.js').StateStore|null} */ _store;
  /** @type {Function|null} */ _unsubscribe = null;
  /** @type {AbortController|null} */ _abortController = null;

  /**
   * @param {HTMLElement} container
   * @param {import('../../application/StateStore.js').StateStore} [store]
   */
  constructor(container, store) {
    this._container = container;
    this._store = store ?? null;
  }

  mount() {
    this._container.innerHTML = this.render();
    this._abortController = new AbortController();
    this._bindEvents();
    this._subscribeToStore();
  }

  unmount() {
    this._unbindEvents();
    this._unsubscribeFromStore();
    this._container.innerHTML = '';
  }

  /** Override: returns HTML string for the component. */
  render() {
    return '';
  }

  /** Override to attach DOM event listeners. Use this._abortController.signal for cleanup. */
  _bindEvents() {}

  /** Cleans up DOM event listeners via AbortController. */
  _unbindEvents() {
    if (this._abortController) {
      this._abortController.abort();
      this._abortController = null;
    }
  }

  /** Override to subscribe to store changes. */
  _subscribeToStore() {
    if (this._store) {
      this._unsubscribe = this._store.subscribe(state => this._onStoreChange(state));
    }
  }

  _unsubscribeFromStore() {
    if (this._unsubscribe) {
      this._unsubscribe();
      this._unsubscribe = null;
    }
  }

  /** Called when store changes. Override for reactive updates. */
  _onStoreChange(_state) {}

  /** Full re-render cycle. */
  update() {
    this.unmount();
    this.mount();
  }
}
