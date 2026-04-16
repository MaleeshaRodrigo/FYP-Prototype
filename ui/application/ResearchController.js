import { BaseController } from './BaseController.js';
import { ExperimentConfig } from '../domain/ExperimentConfig.js';
import { ClinicalMetrics } from '../domain/ClinicalMetrics.js';
import { EventBus } from './EventBus.js';

/**
 * Orchestrates research dashboard: metrics, version history, attack simulation.
 */
export class ResearchController extends BaseController {
  #metricsService;
  #experimentService;
  #attackService;

  constructor(metricsService, experimentService, attackService, store) {
    super(store);
    this.#metricsService = metricsService;
    this.#experimentService = experimentService;
    this.#attackService = attackService;
  }

  static create(locator) {
    return new ResearchController(
      locator.resolve('metricsService'),
      locator.resolve('experimentService'),
      locator.resolve('attackService'),
      locator.resolve('store')
    );
  }

  async init() {
    this._setLoading('research', true);
    try {
      await this.loadTradesBetaSweep();
      await Promise.all([
        this.loadModelMetrics('v8'),
        this.loadVersionHistory(),
        this.loadThesisSummary(),
        this.loadThesisSweep(),
        this.loadMetricsComparison('stage1', 'v8_clean')
      ]);
    } finally {
      this._setLoading('research', false);
    }
  }

  /** @param {string} version */
  async loadModelMetrics(version) {
    try {
      const data = await this.#metricsService.getModelMetrics(version);
      const metrics = ClinicalMetrics.fromApiResponse(data);
      const current = this._store.getState().research.metrics;
      this._store.setState({
        research: { metrics: { ...current, [version]: metrics } }
      });
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

  /** @param {Object} config - attack parameters (epsilon, steps, alpha) */
  async runAttackSimulation(config, imageFile) {
    try {
      this._validateAttackConfig(config);
      if (!imageFile) {
        throw new Error('Please select an image before running PGD attack.');
      }
      this._setLoading('research', true);
      EventBus.emit('attack:started', { config });
      const result = await this.#attackService.runPGDAttack(imageFile, config);
      this._store.setState({ research: { attackResult: result, isLoading: false } });
      EventBus.emit('attack:complete', { result });
    } catch (err) {
      this._setLoading('research', false);
      this.handleError(err);
    }
  }

  /** @param {Object} config */
  _validateAttackConfig(config) {
    if (config.epsilon <= 0 || config.epsilon > 0.1) {
      throw new Error('ε must be in (0, 0.1]');
    }
    if (config.pgd_steps < 1 || config.pgd_steps > 100) {
      throw new Error('PGD steps must be in [1, 100]');
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

  async loadMetricsComparison(baselineVersion, candidateVersion) {
    try {
      const comparison = await this.#metricsService.getMetricsComparison(baselineVersion, candidateVersion);
      this._store.setState({ research: { comparison } });
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
}
