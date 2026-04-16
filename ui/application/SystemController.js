import { BaseController } from './BaseController.js';
import { GAParameters } from '../domain/GAParameters.js';
import { ThresholdValidator } from '../domain/ThresholdValidator.js';
import { EventBus } from './EventBus.js';

/**
 * Orchestrates system admin: model registry, GA parameters, runtime health.
 */
export class SystemController extends BaseController {
  #systemService;

  constructor(systemService, store) {
    super(store);
    this.#systemService = systemService;
  }

  static create(locator) {
    return new SystemController(
      locator.resolve('systemService'),
      locator.resolve('store')
    );
  }

  async init() {
    this._setLoading('system', true);
    try {
      await Promise.all([
        this.loadModelRegistry(),
        this.loadGAParameters()
      ]);
    } finally {
      this._setLoading('system', false);
    }
  }

  async loadModelRegistry() {
    try {
      const data = await this.#systemService.getModelRegistry();
      this._store.setState({ system: { modelRegistry: data } });
    } catch (err) {
      this.handleError(err);
    }
  }

  async loadGAParameters() {
    try {
      const data = await this.#systemService.getGAParameters();
      this._store.setState({ system: { gaParameters: GAParameters.fromApiResponse(data) } });
    } catch (err) {
      this.handleError(err);
    }
  }

  /** @param {string} versionId */
  async activateModel(versionId) {
    try {
      await this.#systemService.activateModel(versionId);
      EventBus.emit('model:activated', { versionId });
      EventBus.emit('notification', { type: 'success', message: `Model ${versionId} activated` });
      await this.loadModelRegistry();
    } catch (err) {
      this.handleError(err);
    }
  }

  /** @param {GAParameters} params */
  async updateGAParameters(params) {
    try {
      const validation = params.validate();
      if (!validation.valid) {
        throw new Error(`Invalid parameters: ${validation.errors.join(', ')}`);
      }

      const safetyCheck = ThresholdValidator.validateGAParameters(params);
      if (!safetyCheck.valid) {
        const warnings = safetyCheck.errors.filter(e => e.startsWith('Warning'));
        const errors = safetyCheck.errors.filter(e => !e.startsWith('Warning'));
        if (errors.length > 0) {
          throw new Error(`Safety validation failed: ${errors.join(', ')}`);
        }
        if (warnings.length > 0) {
          EventBus.emit('notification', { type: 'warning', message: warnings.join('; ') });
        }
      }

      const result = await this.#systemService.updateParameters(params.toPayload());
      this._store.setState({ system: { gaParameters: params } });
      EventBus.emit('parameters:updated', { params });
      EventBus.emit('notification', { type: 'success', message: 'GA parameters updated successfully' });
      return result;
    } catch (err) {
      this.handleError(err);
    }
  }
}
