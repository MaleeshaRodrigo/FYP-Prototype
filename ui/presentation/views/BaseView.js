import { BaseComponent } from '../components/BaseComponent.js';

/**
 * Base class for all page-level views.
 * Adds controller binding and child component lifecycle management.
 */
export class BaseView extends BaseComponent {
  /** @type {import('../../application/BaseController.js').BaseController|null} */
  _controller = null;
  /** @type {import('../../application/EventBus.js').EventBus} */
  _eventBus;
  /** @type {BaseComponent[]} */
  _children = [];

  /**
   * @param {HTMLElement} container
   * @param {import('../../application/StateStore.js').StateStore} store
   * @param {import('../../application/EventBus.js').EventBus} eventBus
   */
  constructor(container, store, eventBus) {
    super(container, store);
    this._eventBus = eventBus;
  }

  /** @param {import('../../application/BaseController.js').BaseController} controller */
  setController(controller) {
    this._controller = controller;
  }

  unmount() {
    for (const child of this._children) {
      child.unmount();
    }
    this._children = [];
    super.unmount();
  }

  /**
   * Helper to mount a child component and track it for cleanup.
   * @param {BaseComponent} component
   */
  _mountChild(component) {
    component.mount();
    this._children.push(component);
  }
}
