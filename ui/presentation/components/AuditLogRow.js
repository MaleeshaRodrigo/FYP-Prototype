import { BaseComponent } from './BaseComponent.js';

/**
 * Atomic component rendering a single audit log entry as a table row.
 */
export class AuditLogRow extends BaseComponent {
  #entry;

  /**
   * @param {HTMLElement} container
   * @param {{ id: string, action: string, user: string, target: string, timestamp: string, details: string }} entry
   */
  constructor(container, entry) {
    super(container);
    this.#entry = entry;
  }

  render() {
    const ts = new Date(this.#entry.timestamp).toLocaleString();
    const actionClass = this._actionCssClass();

    return `
      <tr class="audit-log-row" data-id="${this.#entry.id}">
        <td class="audit-log-row__timestamp metric-value">${ts}</td>
        <td class="audit-log-row__action"><span class="badge badge--${actionClass}">${this.#entry.action}</span></td>
        <td class="audit-log-row__user">${this.#entry.user}</td>
        <td class="audit-log-row__target">${this.#entry.target}</td>
        <td class="audit-log-row__details">${this.#entry.details}</td>
      </tr>
    `;
  }

  _actionCssClass() {
    if (this.#entry.action.includes('activated')) return 'success';
    if (this.#entry.action.includes('deprecated')) return 'warning';
    if (this.#entry.action.includes('parameters')) return 'info';
    return 'neutral';
  }
}
