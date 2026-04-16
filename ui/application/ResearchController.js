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
      await Promise.all([
        this.loadModelMetrics('v8'),
        this.loadVersionHistory()
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
  async runAttackSimulation(config) {
    try {
      this._validateAttackConfig(config);
      this._setLoading('research', true);
      EventBus.emit('attack:started', { config });
      const result = await this.#attackService.runPGDAttack(null, config);
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
}
