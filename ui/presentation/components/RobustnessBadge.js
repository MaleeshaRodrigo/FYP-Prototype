import { BaseComponent } from './BaseComponent.js';

/**
 * Atomic component rendering a robustness tier badge.
 */
export class RobustnessBadge extends BaseComponent {
  /** @type {import('../../domain/RobustnessTier.js').RobustnessTier} */
  #tier;

  /**
   * @param {HTMLElement} container
   * @param {import('../../domain/RobustnessTier.js').RobustnessTier} tier
   */
  constructor(container, tier) {
    super(container);
    this.#tier = tier;
  }

  render() {
    return `
      <span class="robustness-badge robustness-badge--${this.#tier.cssClass}"
            data-tooltip="${this.#tier.label}"
            role="status"
            aria-label="Robustness tier: ${this.#tier.label}">
        ${this.#tier.icon} ${this.#tier.label}
      </span>
    `;
  }
}
