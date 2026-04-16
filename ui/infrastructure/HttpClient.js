/**
 * HTTP transport layer. Single responsibility: make HTTP requests.
 * No business logic, no domain imports.
 */
export class HttpClient {
  /** @type {string} */ #baseUrl;
  /** @type {Object} */ #defaultHeaders;

  /**
   * @param {string} baseUrl
   * @param {Object} [defaultHeaders={}]
   */
  constructor(baseUrl, defaultHeaders = {}) {
    this.#baseUrl = baseUrl;
    this.#defaultHeaders = defaultHeaders;
  }

  /**
   * @param {string} token - JWT bearer token
   */
  setAuthToken(token) {
    this.#defaultHeaders['Authorization'] = `Bearer ${token}`;
  }

  /**
   * @param {string} url
   * @param {Object} [headers={}]
   * @returns {Promise<any>}
   */
  async get(url, headers = {}) {
    const response = await fetch(`${this.#baseUrl}${url}`, {
      method: 'GET',
      headers: { ...this.#defaultHeaders, ...headers }
    });
    return this._handleResponse(response);
  }

  /**
   * @param {string} url
   * @param {any} body
   * @param {Object} [headers={}]
   * @returns {Promise<any>}
   */
  async post(url, body, headers = {}) {
    const isFormData = body instanceof FormData;
    const fetchHeaders = { ...this.#defaultHeaders, ...headers };
    if (!isFormData) {
      fetchHeaders['Content-Type'] = 'application/json';
    }

    const response = await fetch(`${this.#baseUrl}${url}`, {
      method: 'POST',
      headers: fetchHeaders,
      body: isFormData ? body : JSON.stringify(body)
    });
    return this._handleResponse(response);
  }

  /**
   * @param {string} url
   * @param {any} body
   * @param {Object} [headers={}]
   * @returns {Promise<any>}
   */
  async put(url, body, headers = {}) {
    const response = await fetch(`${this.#baseUrl}${url}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json', ...this.#defaultHeaders, ...headers },
      body: JSON.stringify(body)
    });
    return this._handleResponse(response);
  }

  /**
   * @param {Response} response
   * @returns {Promise<any>}
   */
  async _handleResponse(response) {
    if (!response.ok) {
      const body = await response.text().catch(() => '');
      this._handleError(new Error(`HTTP ${response.status}: ${body || response.statusText}`));
    }
    const contentType = response.headers.get('content-type') ?? '';
    if (contentType.includes('application/json')) {
      return response.json();
    }
    return response.text();
  }

  /**
   * @param {Error} error
   * @throws {Error} normalised application error
   */
  _handleError(error) {
    throw Object.assign(new Error(error.message), {
      name: 'AppError',
      originalError: error
    });
  }
}
