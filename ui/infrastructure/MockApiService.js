import { BaseApiService } from './BaseApiService.js';

/**
 * Development/demo API service returning mock fixture data.
 * Drop-in replacement for HareApiService (Liskov Substitution).
 * All responses shaped identically to the production service.
 */
export class MockApiService extends BaseApiService {
  /** @type {Object} */ #fixtures;
  /** @type {Object} */ #runtime;

  /** @param {Object} fixtures - pre-loaded fixture data */
  constructor(fixtures) {
    super();
    this.#fixtures = fixtures;
    this.#runtime = {
      activeModel: 'v8',
      gaParameters: {
        alpha: fixtures.gaParameters?.alpha ?? 0.5467,
        tau: fixtures.gaParameters?.tau ?? 0.7671,
        theta: fixtures.gaParameters?.theta ?? 0.3985
      }
    };
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
    if (!_imageData) {
      throw new Error('Attack image is required for real PGD evaluation');
    }
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
    const active = this.#runtime.activeModel;
    return [
      { id: 'stage1', label: 'Stage 1 — Clean Baseline', checkpoint: 'stage1_baseline.pth', stage: 1, status: active === 'stage1' ? 'active' : 'deprecated', isActive: active === 'stage1', isPending: false },
      { id: 'v8', label: 'Stage 2 v8 — PGD-AT (w_adv=0.05)', checkpoint: 'stage2_v8.pth', stage: 2, status: active === 'v8' ? 'active' : 'deprecated', isActive: active === 'v8', isPending: false },
      { id: 'v9-trades', label: 'Stage 3 v9 — TRADES', checkpoint: 'stage3_v9_trades.pth', stage: 3, status: active === 'v9-trades' ? 'active' : 'pending', isActive: active === 'v9-trades', isPending: active !== 'v9-trades' }
    ];
  }

  async updateParameters(params) {
    await this.#delay();
    this.#runtime.gaParameters = {
      alpha: params.alpha,
      tau: params.tau,
      theta: params.theta
    };
    return {
      success: true,
      applied_bal_acc: 0.7980,
      ...params,
      weight_cnn: params.alpha,
      temperature: params.tau,
      threshold: params.theta
    };
  }

  async getGAParameters() {
    await this.#delay();
    const p = this.#runtime.gaParameters;
    return {
      success: true,
      applied_bal_acc: 0.7980,
      alpha: p.alpha,
      tau: p.tau,
      theta: p.theta,
      weight_cnn: p.alpha,
      temperature: p.tau,
      threshold: p.theta
    };
  }

  async activateModel(versionId) {
    await this.#delay();
    this.#runtime.activeModel = versionId;
    return { success: true, activated: versionId };
  }

  async getThesisSummary() {
    await this.#delay();
    return {
      stage1_clean_default: this.#fixtures.metrics.stage1,
      stage2_clean_default: this.#fixtures.metrics.v8_clean,
      stage2_clean_ga: this.#fixtures.metrics.v8_ga,
      stage2_adv_ga: this.#fixtures.metrics.v8_adversarial
    };
  }

  async getThesisSweep() {
    await this.#delay();
    return {
      '0.0': { bal_acc: 0.7980, sens_mel: 0.7550, spec_nonmel: 0.8416, auc: 0.8711 },
      '0.01': { bal_acc: 0.6320, sens_mel: 0.6110, spec_nonmel: 0.6530, auc: 0.6220 },
      '0.02': { bal_acc: 0.5050, sens_mel: 0.5720, spec_nonmel: 0.4380, auc: 0.3520 },
      '0.03': { bal_acc: 0.3771, sens_mel: 0.7204, spec_nonmel: 0.0339, auc: 0.1032 },
      '0.06': { bal_acc: 0.2920, sens_mel: 0.6830, spec_nonmel: 0.0110, auc: 0.0660 }
    };
  }

  async getTradesBetaSweep() {
    await this.#delay();
    return [
      { beta: 1.0, cleanAUC: 0.8650, advBalAcc: 0.4250 },
      { beta: 2.0, cleanAUC: 0.8500, advBalAcc: 0.5500 },
      { beta: 3.0, cleanAUC: 0.8350, advBalAcc: 0.6500 },
      { beta: 6.0, cleanAUC: 0.8000, advBalAcc: 0.7150 }
    ];
  }

  async getThesisExportJson() {
    await this.#delay();
    return {
      summary: await this.getThesisSummary(),
      robustness_sweep: await this.getThesisSweep(),
      trades_beta_sweep: await this.getTradesBetaSweep(),
      comparisons: {
        stage1_to_stage2_clean: await this.getMetricsComparison('stage1', 'v8_clean'),
        stage2_clean_to_stage2_adv: await this.getMetricsComparison('v8_clean', 'v8_adversarial')
      }
    };
  }

  async getThesisExportCsv() {
    await this.#delay();
    return [
      'Block,Threshold,AUC,BalancedAccuracy,MelanomaSensitivity,NonMelSpecificity',
      'Stage 1 Clean,Default (0.5),0.8741,0.7983,0.7549,0.8416',
      'Stage 2 Clean,Default (0.5),0.8711,0.7515,0.5573,0.9457',
      'Stage 2 Clean,GA (theta),0.8711,0.7980,0.7550,0.8416',
      'Stage 2 Adversarial,GA (theta) - Primary,,0.1432,0.085,'
    ].join('\n');
  }

  async getMetricsComparison(baselineVersion, candidateVersion) {
    await this.#delay();
    const baseline = await this.getModelMetrics(baselineVersion);
    const candidate = await this.getModelMetrics(candidateVersion);
    const fields = ['auc', 'balancedAccuracy', 'melanomaSensitivity', 'nonMelSpecificity'];
    const delta = Object.fromEntries(fields.map((field) => {
      const a = baseline[field];
      const b = candidate[field];
      return [field, typeof a === 'number' && typeof b === 'number' ? Number((b - a).toFixed(4)) : null];
    }));
    return {
      baselineVersion,
      candidateVersion,
      baseline,
      candidate,
      delta
    };
  }
}
