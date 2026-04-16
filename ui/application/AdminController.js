import { BaseController } from './BaseController.js';

/**
 * Orchestrates hospital admin dashboard: usage stats, audit log, model health.
 */
export class AdminController extends BaseController {
  #adminService;

  constructor(adminService, store) {
    super(store);
    this.#adminService = adminService;
  }

  static create(locator) {
    return new AdminController(
      locator.resolve('adminService'),
      locator.resolve('store')
    );
  }

  async init() {
    this._setLoading('admin', true);
    try {
      await Promise.all([
        this.loadUsageStats('daily'),
        this.loadAuditLog({})
      ]);
    } finally {
      this._setLoading('admin', false);
    }
  }

  /** @param {'daily'|'weekly'|'monthly'} period */
  async loadUsageStats(period) {
    try {
      const data = await this.#adminService.getUsageStats(period);
      this._store.setState({ admin: { usageStats: data } });
    } catch (err) {
      this.handleError(err);
    }
  }

  /** @param {Object} filters */
  async loadAuditLog(filters) {
    try {
      const data = await this.#adminService.getAuditLog(filters);
      this._store.setState({ admin: { auditLog: data } });
    } catch (err) {
      this.handleError(err);
    }
  }
}
