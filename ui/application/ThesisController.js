import { BaseController } from './BaseController.js';
import { ExperimentConfig } from '../domain/ExperimentConfig.js';

/**
 * Controller for thesis-focused results hub.
 */
export class ThesisController extends BaseController {
  #metricsService;
  #experimentService;

  constructor(metricsService, experimentService, store) {
    super(store);
    this.#metricsService = metricsService;
    this.#experimentService = experimentService;
  }

  static create(locator) {
    return new ThesisController(
      locator.resolve('metricsService'),
      locator.resolve('experimentService'),
      locator.resolve('store')
    );
  }

  async init() {
    this._setLoading('research', true);
    try {
      await Promise.all([
        this.loadThesisSummary(),
        this.loadThesisSweep(),
        this.loadTradesBetaSweep(),
        this.loadVersionHistory(),
        this.loadMetricsComparison('stage1', 'v8_clean'),
        this.loadMetricsComparison('v8_clean', 'v8_adversarial', 'attackComparison')
      ]);
    } finally {
      this._setLoading('research', false);
    }
  }

  async loadThesisSummary() {
    try {
      const summary = await this.#metricsService.getThesisSummary();
      this._store.setState({ research: { thesisSummary: summary } });
    } catch (err) {
      this.handleError(err);
    }
  }

  async loadThesisSweep() {
    try {
      const sweep = await this.#metricsService.getThesisSweep();
      this._store.setState({ research: { thesisSweep: sweep } });
    } catch (err) {
      this.handleError(err);
    }
  }

  async loadVersionHistory() {
    try {
      const data = await this.#experimentService.getVersionHistory();
      const configs = data.map(d => ExperimentConfig.fromApiResponse(d));
      this._store.setState({ research: { versionHistory: configs } });
    } catch (err) {
      this.handleError(err);
    }
  }

  async loadTradesBetaSweep() {
    try {
      const tradesBetaSweep = await this.#metricsService.getTradesBetaSweep();
      this._store.setState({ research: { tradesBetaSweep } });
    } catch (err) {
      this.handleError(err);
    }
  }

  async loadMetricsComparison(baselineVersion, candidateVersion, key = 'comparison') {
    try {
      const comparison = await this.#metricsService.getMetricsComparison(baselineVersion, candidateVersion);
      this._store.setState({ research: { [key]: comparison } });
    } catch (err) {
      this.handleError(err);
    }
  }

  async getThesisExportJson() {
    return this.#metricsService.getThesisExportJson();
  }

  async getThesisExportCsv() {
    return this.#metricsService.getThesisExportCsv();
  }

  async getStageAttackComparison() {
    return this.#metricsService.getMetricsComparison('v8_clean', 'v8_adversarial');
  }

  async getStageProgressionComparison() {
    return this.#metricsService.getMetricsComparison('stage1', 'v8_clean');
  }

  async refreshComparisons() {
    try {
      const [progression, attackComparison] = await Promise.all([
        this.getStageProgressionComparison(),
        this.getStageAttackComparison()
      ]);
      this._store.setState({ research: { comparison: progression, attackComparison } });
    } catch (err) {
      this.handleError(err);
    }
  }
}