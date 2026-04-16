import { BaseComponent } from './BaseComponent.js';

/**
 * Molecule component: canvas overlay for GradCAM heatmap with opacity slider.
 */
export class GradCAMViewer extends BaseComponent {
  #imageUrl;
  #heatmapData;
  /** @type {HTMLCanvasElement|null} */ #canvas = null;

  /**
   * @param {HTMLElement} container
   * @param {{ imageUrl: string, heatmapData: number[][] }} config
   */
  constructor(container, { imageUrl, heatmapData }) {
    super(container);
    this.#imageUrl = imageUrl;
    this.#heatmapData = heatmapData;
  }

  render() {
    return `
      <div class="gradcam-viewer" role="img" aria-label="GradCAM Heatmap Overlay">
        <div class="gradcam-viewer__canvas-wrapper">
          <img class="gradcam-viewer__original" src="${this.#imageUrl}" alt="Original dermoscopic image" />
          <canvas class="gradcam-viewer__overlay" width="224" height="224"></canvas>
        </div>
        <div class="gradcam-viewer__controls">
          <label class="gradcam-viewer__slider-label" for="gradcam-opacity">
            Overlay Opacity
          </label>
          <input type="range" id="gradcam-opacity" class="gradcam-viewer__slider"
                 min="0" max="1" step="0.05" value="0.5" />
          <span class="gradcam-viewer__opacity-value metric-value">50%</span>
        </div>
      </div>
    `;
  }

  mount() {
    super.mount();
    this.#canvas = this._container.querySelector('.gradcam-viewer__overlay');
    this._drawHeatmap(0.5);
  }

  /**
   * Renders the heatmap onto the canvas using a jet colormap.
   * @param {number} opacity - overlay alpha [0, 1]
   */
  _drawHeatmap(opacity) {
    if (!this.#canvas || !this.#heatmapData) return;
    const ctx = this.#canvas.getContext('2d');
    const rows = this.#heatmapData.length;
    const cols = this.#heatmapData[0]?.length ?? 0;
    if (rows === 0 || cols === 0) return;

    const cellW = this.#canvas.width / cols;
    const cellH = this.#canvas.height / rows;

    ctx.clearRect(0, 0, this.#canvas.width, this.#canvas.height);
    ctx.globalAlpha = opacity;

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const val = this.#heatmapData[r][c];
        ctx.fillStyle = this._jetColor(val);
        ctx.fillRect(c * cellW, r * cellH, cellW + 1, cellH + 1);
      }
    }
    ctx.globalAlpha = 1;
  }

  /**
   * Maps a [0,1] value to a jet colormap RGB string.
   * @param {number} v
   * @returns {string}
   */
  _jetColor(v) {
    const r = Math.min(255, Math.max(0, Math.round(255 * Math.min(4 * v - 1.5, -4 * v + 4.5))));
    const g = Math.min(255, Math.max(0, Math.round(255 * Math.min(4 * v - 0.5, -4 * v + 3.5))));
    const b = Math.min(255, Math.max(0, Math.round(255 * Math.min(4 * v + 0.5, -4 * v + 2.5))));
    return `rgb(${r},${g},${b})`;
  }

  _bindEvents() {
    const signal = this._abortController.signal;
    const slider = this._container.querySelector('#gradcam-opacity');
    const display = this._container.querySelector('.gradcam-viewer__opacity-value');
    if (slider) {
      slider.addEventListener('input', (e) => {
        const val = parseFloat(e.target.value);
        if (display) display.textContent = `${Math.round(val * 100)}%`;
        this._drawHeatmap(val);
      }, { signal });
    }
  }
}
