import { BaseView } from './BaseView.js';
import { DiagnosisResultCard } from '../components/DiagnosisResultCard.js';
import { GradCAMViewer } from '../components/GradCAMViewer.js';
import { EventBus } from '../../application/EventBus.js';

/**
 * Clinical portal view: image upload, diagnosis result, GradCAM, case history.
 */
export class ClinicalView extends BaseView {
  #gradcamUnsub = null;

  render() {
    return `
      <div class="clinical-view">
        <header class="view-header">
          <h2 class="view-header__title">Clinical Portal</h2>
          <p class="view-header__subtitle">Dermoscopic Image Analysis</p>
          <button class="btn btn--outline btn--sm" data-action="logout">Sign Out</button>
        </header>

        <section class="clinical-view__upload" aria-label="Image Upload">
          <div class="upload-panel" id="upload-panel">
            <div class="upload-panel__dropzone" id="dropzone">
              <span class="upload-panel__icon">📤</span>
              <p class="upload-panel__text">Drag & drop a dermoscopic image here</p>
              <p class="upload-panel__or">or</p>
              <label class="btn btn--primary upload-panel__btn">
                Choose File
                <input type="file" accept="image/*" id="file-input" hidden />
              </label>
            </div>
          </div>
        </section>

        <section class="clinical-view__result" id="result-area" aria-label="Diagnosis Result"></section>

        <section class="clinical-view__gradcam" id="gradcam-area" aria-label="GradCAM Heatmap"></section>

        <section class="clinical-view__history" aria-label="Case History">
          <h3>Recent Cases</h3>
          <div id="case-history-list" class="case-history-list"></div>
        </section>
      </div>
    `;
  }

  _bindEvents() {
    const signal = this._abortController.signal;

    const fileInput = this._container.querySelector('#file-input');
    if (fileInput) {
      fileInput.addEventListener('change', (e) => {
        const file = e.target.files?.[0];
        if (file) EventBus.emit('image:selected', { file });
      }, { signal });
    }

    const dropzone = this._container.querySelector('#dropzone');
    if (dropzone) {
      dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('upload-panel__dropzone--active');
      }, { signal });
      dropzone.addEventListener('dragleave', () => {
        dropzone.classList.remove('upload-panel__dropzone--active');
      }, { signal });
      dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('upload-panel__dropzone--active');
        const file = e.dataTransfer?.files?.[0];
        if (file) EventBus.emit('image:selected', { file });
      }, { signal });
    }

    this._container.addEventListener('click', (e) => {
      if (e.target.closest('[data-action="logout"]')) {
        this._store.setState({
          auth: { user: null, role: null, isAuthenticated: false }
        });
        EventBus.emit('auth:logout');
        EventBus.emit('navigate', { path: '#/login' });
      }
    }, { signal });
  }

  _subscribeToStore() {
    super._subscribeToStore();
    this.#gradcamUnsub = EventBus.on('gradcam:ready', ({ heatmapData }) => {
      this._renderGradCAM(heatmapData);
    });
  }

  _unsubscribeFromStore() {
    super._unsubscribeFromStore();
    if (this.#gradcamUnsub) this.#gradcamUnsub();
  }

  _onStoreChange(state) {
    const { clinical } = state;
    if (clinical.currentResult) {
      this._renderResult(clinical.currentResult);
    }
    this._renderCaseHistory(clinical.caseHistory ?? []);
  }

  _renderResult(result) {
    const area = this._container.querySelector('#result-area');
    if (!area) return;
    area.innerHTML = '';
    const card = new DiagnosisResultCard(area, result, this._store);
    this._mountChild(card);
  }

  _renderGradCAM(data) {
    const area = this._container.querySelector('#gradcam-area');
    if (!area) return;
    area.innerHTML = '';
    const viewer = new GradCAMViewer(area, {
      imageUrl: data.original_url ?? '',
      heatmapData: data.heatmap_data ?? data
    });
    this._mountChild(viewer);
  }

  _renderCaseHistory(cases) {
    const list = this._container.querySelector('#case-history-list');
    if (!list || cases.length === 0) return;
    list.innerHTML = cases.map(c => `
      <div class="case-history-item ${c.isMelanoma ? 'case-history-item--mel' : ''}">
        <span class="case-history-item__pred">${c.prediction}</span>
        <span class="case-history-item__conf metric-value">${c.formattedConfidence}</span>
        <span class="case-history-item__time">${c.timestamp.toLocaleString()}</span>
      </div>
    `).join('');
  }
}
