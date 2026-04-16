import { HttpClient } from './HttpClient.js';
import { HareApiService } from './HareApiService.js';
import { MockApiService } from './MockApiService.js';

/**
 * Static DI container. Single source of truth for all service instances.
 * No instances should be created outside this class (except in tests).
 */
export class ServiceLocator {
  static #registry = new Map();

  /**
   * @param {string} name
   * @param {any} instance
   */
  static register(name, instance) {
    ServiceLocator.#registry.set(name, instance);
  }

  /**
   * @param {string} name
   * @returns {any}
   * @throws {Error} if service is not registered
   */
  static resolve(name) {
    if (!ServiceLocator.#registry.has(name)) {
      throw new Error(`ServiceLocator: '${name}' is not registered`);
    }
    return ServiceLocator.#registry.get(name);
  }

  /**
   * Configures all services for the specified environment.
   * @param {'production'|'mock'} env
   * @param {Object} [options={}] - additional config (e.g. apiBaseUrl)
   */
  static async configure(env, options = {}) {
    ServiceLocator.#registry.clear();

    if (env === 'production') {
      const baseUrl = options.apiBaseUrl ?? '';
      const httpClient = new HttpClient(baseUrl);
      const api = new HareApiService(httpClient, baseUrl);
      ServiceLocator.#registerApiServices(api);
    } else {
      const fixtures = await ServiceLocator.#loadFixtures();
      const api = new MockApiService(fixtures);
      ServiceLocator.#registerApiServices(api);
    }
  }

  /**
   * Registers the API service under each focused interface name.
   * @param {import('./BaseApiService.js').BaseApiService} api
   */
  static #registerApiServices(api) {
    ServiceLocator.register('predictionService', api);
    ServiceLocator.register('gradCAMService', api);
    ServiceLocator.register('metricsService', api);
    ServiceLocator.register('experimentService', api);
    ServiceLocator.register('attackService', api);
    ServiceLocator.register('adminService', api);
    ServiceLocator.register('systemService', api);
  }

  /** Loads all mock fixture JSON files. */
  static async #loadFixtures() {
    const basePath = '../shared/mock-fixtures';
    const [diagnosisResult, versionHistory, metrics, gaParameters] = await Promise.all([
      fetch(`${basePath}/diagnosis-result.json`).then(r => r.json()),
      fetch(`${basePath}/version-history.json`).then(r => r.json()),
      fetch(`${basePath}/metrics.json`).then(r => r.json()),
      fetch(`${basePath}/ga-parameters.json`).then(r => r.json())
    ]);
    return { diagnosisResult, versionHistory, metrics, gaParameters };
  }

  /** Clears the registry (for testing). */
  static reset() {
    ServiceLocator.#registry.clear();
  }
}
