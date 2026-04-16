import { BaseController } from './BaseController.js';
import { DiagnosisResult } from '../domain/DiagnosisResult.js';
import { ThresholdValidator } from '../domain/ThresholdValidator.js';
import { EventBus } from './EventBus.js';

/**
 * Orchestrates clinical workflow: image submission, diagnosis, GradCAM.
 * Depends on abstractions (IPredictionService, IGradCAMService) via constructor injection.
 */
export class ClinicalController extends BaseController {
  /** @type {import('../infrastructure/BaseApiService.js').BaseApiService} */ #predictionService;
  /** @type {import('../infrastructure/BaseApiService.js').BaseApiService} */ #gradCAMService;

  /**
   * @param {Object} predictionService
   * @param {Object} gradCAMService
   * @param {import('./StateStore.js').StateStore} store
   */
  constructor(predictionService, gradCAMService, store) {
    super(store);
    this.#predictionService = predictionService;
    this.#gradCAMService = gradCAMService;
  }

  /**
   * Factory method for ServiceLocator-based construction.
   * @param {import('../infrastructure/ServiceLocator.js').ServiceLocator} locator
   */
  static create(locator) {
    return new ClinicalController(
      locator.resolve('predictionService'),
      locator.resolve('gradCAMService'),
      locator.resolve('store')
    );
  }

  async init() {
    this._subscriptions.push(
      EventBus.on('image:selected', ({ file }) => this.submitImage(file))
    );
  }

  /**
   * Submits a dermoscopic image for HARE diagnosis.
   * @param {File} file
   */
  async submitImage(file) {
    try {
      this._validateFile(file);
      this._setLoading('clinical', true);
      const response = await this.#predictionService.predict(file);
      const result = DiagnosisResult.fromApiResponse(response);
      this._updateStateWithResult(result);
      EventBus.emit('diagnosis:complete', { result });
      if (result.requiresGradCAM) {
        await this.fetchGradCAM(result.imageId);
      }
    } catch (err) {
      this._setLoading('clinical', false);
      this.handleError(err);
    }
  }

  /** @param {string} imageId */
  async fetchGradCAM(imageId) {
    try {
      const data = await this.#gradCAMService.getHeatmap(imageId);
      EventBus.emit('gradcam:ready', { imageId, heatmapData: data });
    } catch (err) {
      this.handleError(err);
    }
  }

  /** @param {File} file */
  _validateFile(file) {
    if (!file.type.startsWith('image/')) {
      throw new Error('File must be an image');
    }
    const maxSize = 10 * 1024 * 1024;
    if (file.size > maxSize) {
      throw new Error('File size must not exceed 10 MB');
    }
  }

  /** @param {DiagnosisResult} result */
  _updateStateWithResult(result) {
    const { clinical } = this._store.getState();
    const history = [result, ...(clinical.caseHistory ?? [])].slice(0, 10);
    this._store.setState({
      clinical: { currentResult: result, caseHistory: history, isLoading: false }
    });
  }

  destroy() {
    super.destroy();
  }
}
