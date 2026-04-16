/**
 * Pub/Sub event bus. Central communication channel between decoupled components.
 *
 * Event catalogue (no undocumented events):
 *   'auth:login'         { user, role }
 *   'auth:logout'        null
 *   'navigate'           { path: string }
 *   'image:selected'     { file: File }
 *   'diagnosis:complete' { result: DiagnosisResult }
 *   'gradcam:requested'  { imageId: string }
 *   'gradcam:ready'      { imageId: string, heatmapData: Array }
 *   'attack:started'     { config: Object }
 *   'attack:complete'    { result: Object }
 *   'parameters:updated' { params: GAParameters }
 *   'model:activated'    { versionId: string }
 *   'error'              { code: string, message: string, context?: any }
 *   'notification'       { type: 'success'|'warning'|'info', message: string }
 */
export class EventBus {
  static #handlers = new Map();

  /**
   * Subscribe to an event.
   * @param {string} event
   * @param {Function} handler
   * @returns {Function} unsubscribe function
   */
  static on(event, handler) {
    if (!EventBus.#handlers.has(event)) {
      EventBus.#handlers.set(event, new Set());
    }
    EventBus.#handlers.get(event).add(handler);
    return () => EventBus.off(event, handler);
  }

  /**
   * @param {string} event
   * @param {Function} handler
   */
  static off(event, handler) {
    EventBus.#handlers.get(event)?.delete(handler);
  }

  /**
   * @param {string} event
   * @param {any} [payload]
   */
  static emit(event, payload) {
    EventBus.#handlers.get(event)?.forEach(handler => {
      try {
        handler(payload);
      } catch (err) {
        console.error(`EventBus handler error on '${event}':`, err);
      }
    });
  }

  /**
   * Subscribe to an event, auto-removing after the first invocation.
   * @param {string} event
   * @param {Function} handler
   */
  static once(event, handler) {
    const wrapper = (payload) => {
      EventBus.off(event, wrapper);
      handler(payload);
    };
    EventBus.on(event, wrapper);
  }

  /** Clears all handlers (for testing). */
  static reset() {
    EventBus.#handlers.clear();
  }
}
