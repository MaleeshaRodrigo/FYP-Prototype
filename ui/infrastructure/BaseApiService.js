/**
 * Abstract API service defining the full contract.
 * All methods throw 'Not implemented' — subclasses must override.
 */
export class BaseApiService {
  /** @param {File} imageData @returns {Promise<Object>} */
  async predict(imageData) {
    throw new Error('Not implemented');
  }

  /** @param {string} imageId @returns {Promise<Object>} */
  async getHeatmap(imageId) {
    throw new Error('Not implemented');
  }

  /** @param {string} version @returns {Promise<Object>} */
  async getModelMetrics(version) {
    throw new Error('Not implemented');
  }

  /** @returns {Promise<Array>} */
  async getVersionHistory() {
    throw new Error('Not implemented');
  }

  /** @param {File} imageData @param {Object} config @returns {Promise<Object>} */
  async runPGDAttack(imageData, config) {
    throw new Error('Not implemented');
  }

  /** @param {string} period @returns {Promise<Object>} */
  async getUsageStats(period) {
    throw new Error('Not implemented');
  }

  /** @param {Object} filters @returns {Promise<Array>} */
  async getAuditLog(filters) {
    throw new Error('Not implemented');
  }

  /** @returns {Promise<Array>} */
  async getModelRegistry() {
    throw new Error('Not implemented');
  }

  /** @param {Object} params @returns {Promise<Object>} */
  async updateParameters(params) {
    throw new Error('Not implemented');
  }

  /** @returns {Promise<Object>} */
  async getGAParameters() {
    throw new Error('Not implemented');
  }

  /** @param {string} versionId @returns {Promise<Object>} */
  async activateModel(versionId) {
    throw new Error('Not implemented');
  }

  /** @returns {Promise<Object>} */
  async getThesisSummary() {
    throw new Error('Not implemented');
  }

  /** @returns {Promise<Object>} */
  async getThesisSweep() {
    throw new Error('Not implemented');
  }

  /** @returns {Promise<Array>} */
  async getTradesBetaSweep() {
    throw new Error('Not implemented');
  }

  /** @returns {Promise<Object>} */
  async getThesisExportJson() {
    throw new Error('Not implemented');
  }

  /** @returns {Promise<string>} */
  async getThesisExportCsv() {
    throw new Error('Not implemented');
  }

  /** @param {string} baselineVersion @param {string} candidateVersion @returns {Promise<Object>} */
  async getMetricsComparison(baselineVersion, candidateVersion) {
    throw new Error('Not implemented');
  }
}
