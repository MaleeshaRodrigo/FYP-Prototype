import { BaseApiService } from './BaseApiService.js';

/**
 * Development/demo API service returning mock fixture data.
 * Drop-in replacement for HareApiService (Liskov Substitution).
 * All responses shaped identically to the production service.
 */
export class MockApiService extends BaseApiService {
  /** @type {Object} */ #fixtures;

  /** @param {Object} fixtures - pre-loaded fixture data */
  constructor(fixtures) {
    super();
    this.#fixtures = fixtures;
  }

  /** Simulates network latency (300–800ms). */
  #delay() {
    const ms = 300 + Math.random() * 500;
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  async predict(_imageData) {
    await this.#delay();
    return { ...this.#fixtures.diagnosisResult };
  }

  async getHeatmap(imageId) {
    await this.#delay();
    const size = 14;
    const heatmap = Array.from({ length: size }, () =>
      Array.from({ length: size }, () => Math.random())
    );
    return {
      image_id: imageId,
      heatmap_data: heatmap,
      original_url: '',
      overlay_url: ''
    };
  }

  async getModelMetrics(version) {
    await this.#delay();
    const key = version.replace('-', '_');
    const metrics = this.#fixtures.metrics[key] ?? this.#fixtures.metrics[`${key}_clean`];
    return metrics ? { ...metrics } : { error: 'Version not found' };
  }

  async getVersionHistory() {
    await this.#delay();
    return [...this.#fixtures.versionHistory];
  }

  async runPGDAttack(_imageData, config) {
    await this.#delay();
    return {
      original_prediction: 'MEL',
      original_confidence: 0.7232,
      adversarial_prediction: 'NON_MEL',
      adversarial_confidence: 0.3891,
      epsilon: config.epsilon ?? 0.01,
      pgd_steps: config.pgd_steps ?? 10,
      attack_success: true,
      perturbation_l_inf: config.epsilon ?? 0.01
    };
  }

  async getUsageStats(_period) {
    await this.#delay();
    return {
      total_scans: 1247,
      mel_detections: 89,
      mel_rate: 0.0713,
      referral_rate: 0.124,
      avg_inference_ms: 138,
      period_start: '2025-04-01T00:00:00Z',
      period_end: '2025-04-15T23:59:59Z',
      daily_counts: Array.from({ length: 15 }, (_, i) => ({
        date: `2025-04-${String(i + 1).padStart(2, '0')}`,
        scans: 60 + Math.floor(Math.random() * 40),
        mel: Math.floor(Math.random() * 12)
      }))
    };
  }

  async getAuditLog(_filters) {
    await this.#delay();
    return [
      { id: 'a001', action: 'model:activated', user: 'admin@hare.med', target: 'v8', timestamp: '2025-04-10T09:15:00Z', details: 'Activated v8-PGD-AT' },
      { id: 'a002', action: 'parameters:updated', user: 'admin@hare.med', target: 'ga-params', timestamp: '2025-04-10T09:20:00Z', details: 'θ=0.3985, τ=0.7671, α=0.5467' },
      { id: 'a003', action: 'scan:completed', user: 'clinician@hare.med', target: 'img_001', timestamp: '2025-04-11T14:30:00Z', details: 'MEL detected, confidence 0.72' },
      { id: 'a004', action: 'model:deprecated', user: 'admin@hare.med', target: 'v7', timestamp: '2025-04-12T08:00:00Z', details: 'Deprecated v7 in favour of v8' },
      { id: 'a005', action: 'scan:completed', user: 'clinician@hare.med', target: 'img_002', timestamp: '2025-04-13T11:15:00Z', details: 'NON_MEL, confidence 0.15' }
    ];
  }

  async getModelRegistry() {
    await this.#delay();
    return [
      { id: 'v8', label: 'Stage 2 v8 — PGD-AT (w_adv=0.05)', checkpoint: 'stage2_v8.pth', stage: 2, status: 'active', isActive: true, isPending: false },
      { id: 'v7', label: 'Stage 2 v7 — PGD-AT (w_adv=0.15)', checkpoint: 'stage2_v7.pth', stage: 2, status: 'deprecated', isActive: false, isPending: false },
      { id: 'v9-trades', label: 'Stage 3 v9 — TRADES', checkpoint: 'stage3_v9_trades.pth', stage: 3, status: 'pending', isActive: false, isPending: true }
    ];
  }

  async updateParameters(params) {
    await this.#delay();
    return {
      success: true,
      applied_bal_acc: 0.7980,
      ...params
    };
  }
}
