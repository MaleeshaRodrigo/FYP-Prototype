import { BaseApiService } from './BaseApiService.js';

/**
 * Production API service calling Azure-hosted FastAPI backend.
 */
export class HareApiService extends BaseApiService {
  /** @type {import('./HttpClient.js').HttpClient} */ #http;
  /** @type {string} */ #baseUrl;

  /**
   * @param {import('./HttpClient.js').HttpClient} httpClient
   * @param {string} baseUrl
   */
  constructor(httpClient, baseUrl) {
    super();
    this.#http = httpClient;
    this.#baseUrl = baseUrl;
  }

  /** @param {File} imageData */
  async predict(imageData) {
    const form = new FormData();
    form.append('image', imageData);
    return this.#http.post('/api/predict', form);
  }

  /** @param {string} imageId */
  async getHeatmap(imageId) {
    return this.#http.get(`/api/gradcam/${imageId}`);
  }

  /** @param {string} version */
  async getModelMetrics(version) {
    return this.#http.get(`/api/metrics/${version}`);
  }

  async getVersionHistory() {
    return this.#http.get('/api/experiments');
  }

  /** @param {File} imageData @param {Object} config */
  async runPGDAttack(imageData, config) {
    const form = new FormData();
    if (imageData) form.append('image', imageData);
    Object.entries(config).forEach(([k, v]) => form.append(k, String(v)));
    return this.#http.post('/api/attack/simulate', form);
  }

  /** @param {string} period */
  async getUsageStats(period) {
    return this.#http.get(`/api/admin/usage?period=${encodeURIComponent(period)}`);
  }

  /** @param {Object} filters */
  async getAuditLog(filters) {
    const params = new URLSearchParams(filters).toString();
    return this.#http.get(`/api/admin/audit?${params}`);
  }

  async getModelRegistry() {
    return this.#http.get('/api/system/models');
  }

  /** @param {Object} params */
  async updateParameters(params) {
    return this.#http.put('/api/system/parameters', params);
  }
}
