# HARE Platform — Cursor Development Plan
## Adversarial-Robust Skin Cancer Detection Web Interface

> **How to use this file:**  
> Open this file in Cursor. Use it as the primary reference context for every code generation prompt. When asking Cursor to implement a file, reference it as: *"Following the HARE Platform plan, implement `domain/DiagnosisResult.js`"*.

---

## 0. Quick Reference

| Item | Value |
|------|-------|
| **System name** | HARE — Hybrid Adversarially Robust Ensemble |
| **Model** | HAREMaster (ResNet50 + ViT-Small-Patch16, 46.4M params) |
| **Dataset** | ISIC-2019, binary MEL vs Non-MEL |
| **Stage 1 AUC** | 0.8741 (clean baseline) |
| **Stage 2 v8 AUC** | 0.8711 (post PGD-AT, all clinical targets met) |
| **GA threshold** | θ = 0.3985, τ = 0.7671, α = 0.5467 → bal_acc = 0.7980 |
| **Adv robustness** | adv_bal_acc = 0.1432 (v8 + GA under PGD-10) — TRADES v9 pending |
| **Principles** | SOLID, OOP, Clean Code, Single Responsibility per file |
| **Frontend host** | Azure Static Web Apps (Free Tier) |
| **Backend host** | Azure Container Apps (education credit) |
| **Repo strategy** | Single monorepo — `/ui` + `/api` + `/shared` |

---

## 1. Repository Structure

Single GitHub repository. Frontend and backend coexist; GitHub Actions deploys them independently.

```
hare-platform/
│
├── .github/
│   └── workflows/
│       ├── deploy-ui.yml           ← Azure Static Web Apps deployment on push to main
│       └── deploy-api.yml          ← Docker build + push to Azure Container Registry
│
├── ui/                             ← Frontend (Vanilla ES2022, no build step)
│   ├── index.html                  ← Single HTML entry point
│   ├── style.css                   ← Design tokens + base stylesheet
│   ├── app.js                      ← Bootstrap: ServiceLocator.configure() → Router.init()
│   │
│   ├── infrastructure/
│   │   ├── HttpClient.js
│   │   ├── BaseApiService.js
│   │   ├── HareApiService.js
│   │   ├── MockApiService.js
│   │   └── ServiceLocator.js
│   │
│   ├── domain/
│   │   ├── DiagnosisResult.js
│   │   ├── ClinicalMetrics.js
│   │   ├── ModelVersion.js
│   │   ├── ExperimentConfig.js
│   │   ├── GAParameters.js
│   │   ├── RobustnessTier.js
│   │   ├── MetricsFormatter.js
│   │   └── ThresholdValidator.js
│   │
│   ├── application/
│   │   ├── StateStore.js
│   │   ├── EventBus.js
│   │   ├── Router.js
│   │   ├── AuthGuard.js
│   │   ├── BaseController.js
│   │   ├── ClinicalController.js
│   │   ├── ResearchController.js
│   │   ├── AdminController.js
│   │   └── SystemController.js
│   │
│   └── presentation/
│       ├── components/
│       │   ├── BaseComponent.js
│       │   ├── RobustnessBadge.js
│       │   ├── MetricsKPICard.js
│       │   ├── ConfidenceBar.js
│       │   ├── AuditLogRow.js
│       │   ├── DiagnosisResultCard.js
│       │   ├── GradCAMViewer.js
│       │   ├── ExperimentTableRow.js
│       │   ├── TradeoffChart.js
│       │   ├── TRADESBetaSweepChart.js
│       │   └── GAParameterEditor.js
│       └── views/
│           ├── BaseView.js
│           ├── LoginView.js
│           ├── ClinicalView.js
│           ├── ResearchView.js
│           ├── AdminView.js
│           └── SystemView.js
│
├── api/                            ← Backend (FastAPI + PyTorch)
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py                     ← FastAPI app entry point
│   │
│   ├── core/
│   │   ├── config.py               ← Environment variables, model paths
│   │   ├── model_loader.py         ← Load HAREMaster checkpoints
│   │   └── security.py             ← JWT auth, role validation
│   │
│   ├── domain/
│   │   ├── schemas.py              ← Pydantic request/response models
│   │   ├── robustness_tier.py      ← Python enum mirroring JS RobustnessTier
│   │   └── clinical_targets.py     ← Constant clinical thresholds
│   │
│   ├── services/
│   │   ├── base_service.py         ← Abstract base (SRP, DIP)
│   │   ├── prediction_service.py   ← HARE model inference
│   │   ├── gradcam_service.py      ← GradCAM heatmap generation
│   │   ├── attack_service.py       ← PGD-10 attack simulation
│   │   ├── metrics_service.py      ← Model performance metrics
│   │   └── experiment_service.py   ← Version history data
│   │
│   └── routers/
│       ├── predict.py              ← POST /api/predict
│       ├── gradcam.py              ← GET  /api/gradcam/{image_id}
│       ├── attack.py               ← POST /api/attack/simulate
│       ├── metrics.py              ← GET  /api/metrics/{version}
│       ├── experiments.py          ← GET  /api/experiments
│       ├── admin.py                ← GET  /api/admin/usage, /audit
│       └── system.py               ← GET/PUT /api/system/models, /parameters
│
└── shared/
    ├── constants.json              ← Clinical targets, model versions — shared by UI + API
    └── mock-fixtures/
        ├── diagnosis-result.json
        ├── version-history.json
        ├── metrics.json
        └── ga-parameters.json
```

---

## 2. Design Goals

### 2.1 Functional Goals

| ID | Goal | Stakeholder | Priority |
|----|------|-------------|----------|
| FG-01 | Clinicians upload a dermoscopic image → receive HARE diagnosis (MEL/Non-MEL) with confidence score and GradCAM heatmap | Clinician | 🔴 Critical |
| FG-02 | Robustness tier badge (Baseline / Partial / TRADES-Certified) on every diagnosis result | Clinician, Admin | 🔴 Critical |
| FG-03 | Research engineers inspect the full v3–v9 experiment version history with AT hyperparameters and metrics | Research Engineer | 🔴 Critical |
| FG-04 | Adversarial trade-off chart visualising the w_adv phase-transition at 0.20–0.25 | Research Engineer | 🟡 High |
| FG-05 | Hospital admins view scan volume, MEL detection rate, and referral rate trends | Hospital Admin | 🟡 High |
| FG-06 | System admins manage model checkpoints, edit GA-optimised parameters (θ, τ, α), view runtime health | System Admin | 🟡 High |
| FG-07 | Role-based access: each stakeholder sees only their authorised views | All | 🔴 Critical |
| FG-08 | UI functions with mock data when inference backend is offline (dev mode) | Engineering | 🟡 High |

### 2.2 Non-Functional Goals

| ID | Goal | Metric |
|----|------|--------|
| NFG-01 | Performance | LCP < 2.0s, INP < 150ms |
| NFG-02 | Accessibility | WCAG AA — 4.5:1 contrast, full keyboard nav |
| NFG-03 | Responsiveness | 768px → 1920px breakpoints |
| NFG-04 | Deployability | Zero build-step static output → Azure Static Web Apps |
| NFG-05 | Maintainability | Every class ≤ 1 responsibility; cyclomatic complexity ≤ 10 per method |
| NFG-06 | Testability | All services mockable; 0 global state |

### 2.3 Clinical Safety Constraints (Non-Negotiable)

These are hard rules — Cursor must enforce them in every relevant component:

1. **Borderline flag:** If `|confidence - θ| < 0.05` → always render an explicit "Borderline — Recommend Specialist Review" banner. Never hide it.
2. **Model version visible:** Every `DiagnosisResultCard` must display the exact checkpoint used (e.g., `v8-PGD-AT`, `v9-TRADES`).
3. **TRADES model first:** When v9-TRADES is available, default to it. Any downgrade requires admin override + audit log entry.
4. **GradCAM mandatory on MEL-positive:** `DiagnosisResult.requiresGradCAM` returns `true` whenever `isMelanoma === true`. The view must auto-trigger it.

---

## 3. Technology Stack

### Frontend (`/ui`)

| Concern | Choice | Reason |
|---------|--------|--------|
| Language | Vanilla ES2022 (class syntax) | No build tools; direct Azure Static Web Apps deploy |
| Styling | CSS3 Custom Properties (Nexus Design System) | Theme tokens, zero dependency |
| Charts | Chart.js 4 via CDN | Scatter + line support for trade-off visualisation |
| Icons | Lucide Icons via CDN | Accessible SVG set |
| Fonts | Satoshi (Fontshare) + Instrument Serif | Clinical precision aesthetics |
| Module system | ES Modules (`<script type="module">`) | Native browser modules; no bundler |

### Backend (`/api`)

| Concern | Choice |
|---------|--------|
| Framework | FastAPI (Python 3.11+) |
| ML runtime | PyTorch 2.x + torchvision |
| Model | HAREMaster (ResNet50 + ViT-Small-Patch16) |
| GradCAM | `pytorch-grad-cam` library |
| Validation | Pydantic v2 |
| Auth | python-jose (JWT) |
| Container | Docker (python:3.11-slim base) |

### Infrastructure

| Service | Purpose | Cost |
|---------|---------|------|
| Azure Static Web Apps (Free) | Host `/ui` | $0 |
| Azure Container Apps | Host `/api` Docker container | From ~$70 education credit |
| Azure Container Registry (Basic) | Store Docker image | ~$5/month from credit |
| Azure Blob Storage (Free 5GB) | Store model checkpoints | $0 |
| GitHub Actions | CI/CD for both services | $0 |

---

## 4. SOLID Principles — Enforcement Rules for Cursor

When implementing any file, Cursor must enforce all five principles:

### S — Single Responsibility
- One class = one reason to change
- If a class does X and Y → split into two classes
- **Check:** Can you describe what the class does in one sentence without using "and"?

### O — Open/Closed
- New features → new subclasses or new implementations
- Never modify `BaseComponent`, `BaseController`, or `BaseApiService`
- **Check:** Would adding a new view require editing an existing class? If yes, the base is wrong.

### L — Liskov Substitution
- `MockApiService` must be a drop-in replacement for `HareApiService`
- Every subclass method must honour the parent's contract (same return shape, same error types)
- **Check:** Replace every `HareApiService` with `MockApiService` — does the app still work?

### I — Interface Segregation
- 7 focused service interfaces, not one fat `IApiService`
- A component that only needs `IGradCAMService` must not import the whole `BaseApiService`
- **Check:** Does the consumer use every method it imports? If not, the interface is too fat.

### D — Dependency Inversion
- High-level modules (`ClinicalController`) depend on abstractions (`IPredictionService`), not concretions
- All services resolved via `ServiceLocator.resolve()`, never `new ServiceName()` inside a controller
- **Check:** Can you run `ClinicalController` with a `MockApiService` injected? If not, DIP is violated.

---

## 5. Layered Architecture

### Dependency Rule
```
Presentation → Application → Domain ← Infrastructure
```
- Domain layer imports nothing from any other layer
- Infrastructure implements interfaces declared in the domain
- Application layer imports domain + infrastructure
- Presentation imports application + domain

### Layer Responsibilities

| Layer | Path | Responsibility |
|-------|------|---------------|
| Infrastructure | `ui/infrastructure/` | HTTP transport, API service implementations, DI container |
| Domain | `ui/domain/` | Business logic, value objects, formatters, validators |
| Application | `ui/application/` | Routing, state, event bus, controller orchestration |
| Presentation | `ui/presentation/` | HTML rendering, DOM events, component lifecycle |

---

## 6. Complete Class Specifications

### 6.1 Infrastructure Layer

#### `BaseApiService.js`
```
Abstract class. Defines the full API contract.
All methods throw Error('Not implemented') — subclasses override.

Methods:
  async predict(imageData: File): Promise<Object>
  async getHeatmap(imageId: string): Promise<Object>
  async getModelMetrics(version: string): Promise<Object>
  async getVersionHistory(): Promise<Array>
  async runPGDAttack(imageData: File, config: Object): Promise<Object>
  async getUsageStats(period: string): Promise<Object>
  async getAuditLog(filters: Object): Promise<Array>
  async getModelRegistry(): Promise<Array>
  async updateParameters(params: Object): Promise<Object>
```

#### `HareApiService.js`
```
Extends BaseApiService. Production implementation.
Constructor: (httpClient: HttpClient, baseUrl: string)
Calls Azure Container App FastAPI endpoints.

Endpoint map:
  predict()         → POST  {baseUrl}/api/predict
  getHeatmap()      → GET   {baseUrl}/api/gradcam/{imageId}
  getModelMetrics() → GET   {baseUrl}/api/metrics/{version}
  getVersionHistory()→ GET  {baseUrl}/api/experiments
  runPGDAttack()    → POST  {baseUrl}/api/attack/simulate
  getUsageStats()   → GET   {baseUrl}/api/admin/usage?period={period}
  getAuditLog()     → GET   {baseUrl}/api/admin/audit
  getModelRegistry()→ GET   {baseUrl}/api/system/models
  updateParameters()→ PUT   {baseUrl}/api/system/parameters
```

#### `MockApiService.js`
```
Extends BaseApiService. Development/demo implementation.
Returns hardcoded fixtures from /shared/mock-fixtures/*.json
All methods return Promises (simulate async) with 300–800ms delay.
Constructor: (fixtures: Object)  ← inject fixture data, not loaded directly

IMPORTANT: Return shapes must be IDENTICAL to HareApiService responses.
```

#### `HttpClient.js`
```
Single responsibility: HTTP transport.
No business logic.

Methods:
  async get(url: string, headers?: Object): Promise<any>
  async post(url: string, body: any, headers?: Object): Promise<any>
  async put(url: string, body: any, headers?: Object): Promise<any>
  _handleResponse(response: Response): Promise<any>
  _handleError(error: Error): never   ← throws normalised AppError

Never import domain classes. No side effects except HTTP calls.
```

#### `ServiceLocator.js`
```
Static DI container. Single source of truth for all service instances.
No instances created outside this class (except in tests).

Static methods:
  static register(name: string, instance: any): void
  static resolve(name: string): any           ← throws if not registered
  static configure(env: 'production'|'mock'): void
    → 'production': registers HareApiService, real HttpClient
    → 'mock':       registers MockApiService with fixture data

Services registered:
  'predictionService'   → IPredictionService
  'gradCAMService'      → IGradCAMService
  'metricsService'      → IMetricsService
  'experimentService'   → IExperimentService
  'attackService'       → IAttackService
  'adminService'        → IAdminService
  'systemService'       → ISystemService
  'store'               → StateStore
  'eventBus'            → EventBus
  'router'              → Router
```

---

### 6.2 Domain Layer

#### `DiagnosisResult.js`
```
Immutable value object. Wraps a single HARE prediction.

Constructor: ({
  imageId: string,
  prediction: 'MEL' | 'NON_MEL',
  confidence: number,           ← raw softmax probability [0–1]
  threshold: number,            ← θ = 0.3985 (from GA)
  modelVersion: ModelVersion,
  timestamp: Date
})

Getters (no setters — immutable):
  get isMelanoma(): boolean           → prediction === 'MEL'
  get isBorderline(): boolean         → |confidence - threshold| < 0.05
  get requiresGradCAM(): boolean      → isMelanoma === true
  get riskLevel(): 'high'|'borderline'|'low'
  get formattedConfidence(): string   → uses MetricsFormatter

Methods:
  toJSON(): Object
  static fromApiResponse(data: Object): DiagnosisResult
```

#### `ClinicalMetrics.js`
```
Immutable value object. Encapsulates the four clinical performance metrics.

Constructor: ({
  auc: number,
  balancedAccuracy: number,
  melanomaSensitivity: number,
  nonMelSpecificity: number,
  modelVersion: string,
  evaluationType: 'clean' | 'adversarial'
})

Static constants:
  static TARGETS = { auc: 0.80, balAcc: 0.65, sens: 0.40, spec: 0.82 }

Getters:
  get allTargetsMet(): boolean
  get failedTargets(): Array<string>

Methods:
  meetsTarget(metricName: string): boolean
  deltaFrom(other: ClinicalMetrics): Object   ← returns signed deltas
  toDisplayFormat(): Object                   ← formatted strings via MetricsFormatter
  static fromApiResponse(data: Object): ClinicalMetrics
```

#### `ModelVersion.js`
```
Immutable value object. Represents a single HARE model checkpoint.

Constructor: ({
  id: string,              ← 'v3', 'v4', ..., 'v8', 'v9-trades'
  label: string,           ← 'Stage 2 v8 — PGD-AT (w_adv=0.05)'
  checkpoint: string,      ← 'stage2_v8.pth'
  stage: 1 | 2 | 3,
  robustnessTier: RobustnessTier,
  isActive: boolean,
  isPending: boolean       ← true for v9-TRADES until experiments complete
})

Getters:
  get isTradesCertified(): boolean
  get displayLabel(): string
  get statusTag(): 'active'|'deprecated'|'pending'
```

#### `ExperimentConfig.js`
```
Immutable value object. AT hyperparameters for a single experiment version.

Constructor: ({
  version: string,          ← 'v3'–'v8'
  advLossWeight: number,    ← w_adv
  epsilon: number,          ← ε (L∞ budget)
  pgdSteps: number,         ← K
  pgdAlpha: number,         ← inner loop step size
  learningRate: number,
  epochs: number,
  cleanMetrics: ClinicalMetrics,
  advMetrics: ClinicalMetrics | null
})

Static constants:
  static PHASE_TRANSITION_THRESHOLD = 0.225  ← midpoint of 0.20–0.25 boundary

Getters:
  get isAbovePhaseTransition(): boolean  → advLossWeight >= PHASE_TRANSITION_THRESHOLD
  get riskLevel(): 'safe'|'transition'|'forgetting'
  get rowCSSClass(): string              ← for colour-coded table rows
```

#### `GAParameters.js`
```
Mutable value object (edited by System Admin).

Constructor: ({
  alpha: number,    ← CNN fusion weight ∈ [0, 1]
  tau: number,      ← temperature ∈ [0.5, 2.0]
  theta: number     ← decision threshold ∈ [0.3, 0.7]
})

Default values: { alpha: 0.5467, tau: 0.7671, theta: 0.3985 }

Methods:
  validate(): { valid: boolean, errors: Array<string> }
  toPayload(): Object   ← serialised for API PUT request
  static defaults(): GAParameters
```

#### `RobustnessTier.js`
```
Enum-like class. Eliminates all magic strings for tier comparisons.

Static instances:
  static BASELINE = new RobustnessTier('BASELINE', 'Baseline', '🟡', 'warning', 0)
  static PARTIAL  = new RobustnessTier('PARTIAL',  'Partial Robustness', '🟠', 'partial', 1)
  static TRADES   = new RobustnessTier('TRADES',   'TRADES-Certified', '🟢', 'success', 2)

Constructor: (code, label, icon, cssClass, level)

Static factory methods:
  static fromAdvLossWeight(w: number): RobustnessTier
    → w <= 0.05  → BASELINE
    → w <= 0.20  → PARTIAL
    → w > 0.20   → check if TRADES loss was used

  static fromModelVersion(versionId: string): RobustnessTier

Methods:
  isAtLeast(tier: RobustnessTier): boolean   ← level-based comparison
  toString(): string
```

#### `MetricsFormatter.js`
```
Pure static utility. No constructor, no state, no side effects.

Static methods:
  static formatAUC(value: number): string             → '0.8741'
  static formatPercent(value: number): string         → '79.8%'
  static formatDelta(a: number, b: number): string    → '+0.003' or '−0.047'
  static formatConfidence(value: number): string      → '73.2%'
  static formatAdvLossWeight(w: number): string       → 'w_adv = 0.05'
  static formatEpsilon(e: number): string             → 'ε = 0.01 (L∞)'
  static metricLabel(name: string): string            → human-readable metric name
```

#### `ThresholdValidator.js`
```
Pure static utility. No constructor, no state.

Static constants:
  static CLINICAL_TARGETS = { auc: 0.80, balAcc: 0.65, sens: 0.40, spec: 0.82 }
  static BORDERLINE_MARGIN = 0.05

Static methods:
  static validateMetrics(metrics: ClinicalMetrics): { passed: boolean, failures: Array }
  static isBorderline(confidence: number, theta: number): boolean
  static validateGAParameters(params: GAParameters): { valid: boolean, errors: Array }
  static phaseTransitionRisk(advLossWeight: number): 'safe'|'transition'|'forgetting'
```

---

### 6.3 Application Layer

#### `StateStore.js`
```
Observer pattern. Central application state.
No global variables — StateStore instance injected via ServiceLocator.

Constructor: (initialState: Object)

Initial state shape:
{
  auth:         { user: null, role: null, isAuthenticated: false },
  clinical:     { currentResult: null, caseHistory: [], isLoading: false },
  research:     { metrics: {}, versionHistory: [], isLoading: false },
  admin:        { usageStats: null, auditLog: [], modelHealth: null },
  system:       { modelRegistry: [], gaParameters: null, runtimeHealth: null },
  ui:           { activeRoute: null, error: null, notification: null }
}

Methods:
  getState(): Object                    ← returns deep-frozen copy
  setState(partial: Object): void       ← shallow merge + notify
  subscribe(listener: Function): Function  ← returns unsubscribe fn
  _notify(): void                       ← private; calls all listeners
  _freeze(obj: Object): Object          ← deep freeze to prevent mutation
```

#### `EventBus.js`
```
Pub/Sub. Static class — no instantiation needed.

Static methods:
  static on(event: string, handler: Function): Function  ← returns off fn
  static off(event: string, handler: Function): void
  static emit(event: string, payload?: any): void
  static once(event: string, handler: Function): void   ← auto-removes after first call

Event catalogue (document all events here — no undocumented events):
  'auth:login'              payload: { user, role }
  'auth:logout'             payload: null
  'navigate'                payload: { path: string }
  'image:selected'          payload: { file: File }
  'diagnosis:complete'      payload: { result: DiagnosisResult }
  'gradcam:requested'       payload: { imageId: string }
  'attack:started'          payload: { config: Object }
  'attack:complete'         payload: { result: Object }
  'parameters:updated'      payload: { params: GAParameters }
  'model:activated'         payload: { versionId: string }
  'error'                   payload: { code: string, message: string, context?: any }
  'notification'            payload: { type: 'success'|'warning'|'info', message: string }
```

#### `Router.js`
```
Hash-based SPA router. Maps #/path → Controller + View pairs.

Constructor: (authGuard: AuthGuard, serviceLocator: ServiceLocator)

Route registry (defined in app.js, registered via Router.register()):
  #/login          → LoginView             (no auth required)
  #/clinical       → ClinicalView + ClinicalController    (role: clinician)
  #/research       → ResearchView + ResearchController    (role: research)
  #/admin          → AdminView + AdminController           (role: admin)
  #/system         → SystemView + SystemController         (role: system)

Methods:
  register(path: string, ViewClass: Class, ControllerClass: Class, role: string): void
  navigate(path: string): void
  _onHashChange(): void        ← window.addEventListener('hashchange', ...)
  _initRoute(route: Object): void
  _teardownCurrentRoute(): void
  _getContainer(): HTMLElement ← '#app-content'
```

#### `AuthGuard.js`
```
Route protection. Checks role before navigation.

Static constants:
  static ROLES = {
    CLINICIAN: 'clinician',
    RESEARCH:  'research',
    ADMIN:     'admin',
    SYSTEM:    'system'
  }

Methods:
  canActivate(requiredRole: string, userRole: string): boolean
  redirectToLogin(): void   ← EventBus.emit('navigate', { path: '#/login' })
```

#### `BaseController.js`
```
Abstract. Provides common controller lifecycle.

Constructor: (store: StateStore, eventBus: typeof EventBus)

Methods:
  async init(): Promise<void>           ← override in subclasses; called by Router
  handleError(err: Error): void         ← emits 'error' event via EventBus
  destroy(): void                       ← cleanup subscriptions; override in subclasses
  _setLoading(slice: string, val: bool) ← updates store loading flag for a state slice
```

#### `ClinicalController.js`
```
Extends BaseController.
Constructor: (predictionService, gradCAMService, store, eventBus)
Depends on ABSTRACTIONS — never imports HareApiService directly.

Methods:
  async init(): Promise<void>
  async submitImage(file: File): Promise<void>
    1. Validate file (type: image/*, size < 10MB)
    2. _setLoading('clinical', true)
    3. predictionService.predict(file)
    4. Wrap response in DiagnosisResult.fromApiResponse()
    5. Check ThresholdValidator.isBorderline()
    6. store.setState({ clinical: { currentResult, isLoading: false } })
    7. EventBus.emit('diagnosis:complete', { result })
    8. If result.requiresGradCAM → fetchGradCAM(result.imageId)

  async fetchGradCAM(imageId: string): Promise<void>
    1. gradCAMService.getHeatmap(imageId)
    2. EventBus.emit('gradcam:ready', { imageId, heatmapData })

  async exportCase(caseId: string): Promise<void>
    ← generates PDF blob from case data; triggers download

  destroy(): void   ← unsubscribe all store listeners
```

#### `ResearchController.js`
```
Extends BaseController.
Constructor: (metricsService, experimentService, attackService, store, eventBus)

Methods:
  async init(): Promise<void>
    ← loads metrics for all versions + version history in parallel
  async loadModelMetrics(version: string): Promise<void>
  async loadVersionHistory(): Promise<void>
  async runAttackSimulation(config: Object): Promise<void>
    1. Validate config (epsilon, steps, alpha)
    2. _setLoading('research', true)
    3. attackService.runPGDAttack(null, config)
    4. Store result + emit 'attack:complete'
```

#### `AdminController.js`
```
Extends BaseController.
Constructor: (adminService, store, eventBus)

Methods:
  async init(): Promise<void>
  async loadUsageStats(period: 'daily'|'weekly'|'monthly'): Promise<void>
  async loadAuditLog(filters: Object): Promise<void>
  async loadModelHealth(): Promise<void>
```

#### `SystemController.js`
```
Extends BaseController.
Constructor: (systemService, store, eventBus)

Methods:
  async init(): Promise<void>
  async loadModelRegistry(): Promise<void>
  async activateModel(versionId: string): Promise<void>
    1. Confirm user has 'system' role (AuthGuard)
    2. systemService activates model
    3. Logs audit entry
    4. Emit 'model:activated'

  async updateGAParameters(params: GAParameters): Promise<void>
    1. params.validate() — throw if invalid
    2. ThresholdValidator.validateGAParameters(params)
    3. systemService.updateParameters(params.toPayload())
    4. Emit 'parameters:updated'
    5. Emit 'notification' success

  async loadRuntimeHealth(): Promise<void>
```

---

### 6.4 Presentation Layer

#### `BaseComponent.js`
```
Abstract base for all UI components.
Constructor: (container: HTMLElement, store: StateStore)

Lifecycle methods (Template Method pattern):
  mount(): void
    1. this._container.innerHTML = this.render()
    2. this._bindEvents()
    3. this._subscribeToStore()

  unmount(): void
    1. this._unbindEvents()
    2. this._unsubscribeFromStore()
    3. this._container.innerHTML = ''

  render(): string         ← MUST override; returns HTML string
  _bindEvents(): void      ← override to attach DOM listeners
  _unbindEvents(): void    ← override to clean up listeners
  _subscribeToStore(): void← override if component is store-reactive
  _onStoreChange(state): void ← called when store changes; triggers re-render
  update(): void           ← unmount() + mount()
```

#### `RobustnessBadge.js`
```
Extends BaseComponent.
Atomic. Renders a tier badge with icon, label, and tooltip.
Constructor: (container, tier: RobustnessTier)

render(): string
  → <span class="robustness-badge robustness-badge--{tier.cssClass}"
          data-tooltip="{tier.description}">
       {tier.icon} {tier.label}
     </span>
```

#### `MetricsKPICard.js`
```
Extends BaseComponent.
Atomic. Renders a single metric KPI.
Constructor: (container, { label, value, target, delta, unit })

render(): string
  → card with: metric label, large value, target line, delta badge
  → delta colour: green if positive, red if negative
  → target met indicator (✅/❌)
  → uses MetricsFormatter for all number formatting
```

#### `ConfidenceBar.js`
```
Extends BaseComponent.
Atomic. Animated probability bar with threshold marker.
Constructor: (container, { confidence: number, threshold: number })

render(): string
  → progress bar (CSS width = confidence * 100%)
  → threshold marker line at threshold * 100%
  → colour: green if > threshold, amber if borderline, red if very low

mount():
  super.mount()
  requestAnimationFrame(() => animate bar from 0 to confidence)
```

#### `DiagnosisResultCard.js`
```
Extends BaseComponent.
Molecule. Composes RobustnessBadge + ConfidenceBar + GradCAM trigger.
Constructor: (container, result: DiagnosisResult, store)

render(): string
  → prediction label (MEL / Non-MEL) with colour coding
  → confidence bar
  → RobustnessBadge for result.modelVersion.robustnessTier
  → model version chip: result.modelVersion.label
  → if result.isBorderline → WARNING BANNER (clinical safety rule)
  → if result.requiresGradCAM → GradCAM toggle button
  → export button, referral flag button

_bindEvents():
  → gradcam toggle → EventBus.emit('gradcam:requested', { imageId })
  → export → controller.exportCase(result.imageId)
  → referral flag → EventBus.emit('referral:flagged', { result })
```

#### `GradCAMViewer.js`
```
Extends BaseComponent.
Molecule. Canvas overlay with opacity slider.
Constructor: (container, { imageUrl: string, heatmapData: Array })

render(): string
  → wrapper div with: <img> (original), <canvas> (overlay), opacity slider

mount():
  super.mount()
  this._drawHeatmap()

_drawHeatmap(): void
  → draws heatmap data onto canvas using jet colormap
  → applies current opacity from slider
  → PURE canvas logic — no external dependencies

_bindEvents():
  → opacity slider input → redraw with new alpha
```

#### `ExperimentTableRow.js`
```
Extends BaseComponent.
Molecule. Single row in the version history table.
Constructor: (container, config: ExperimentConfig)

render(): string
  → <tr class="{config.rowCSSClass}">
       version | w_adv | ε | pgd_steps | clean AUC | bal_acc | adv_bal_acc | status
     </tr>
  → phase transition rows highlighted with amber background
  → forgetting rows highlighted with red background
```

#### `TradeoffChart.js`
```
Extends BaseComponent.
Molecule. Chart.js scatter plot — w_adv vs AUC + adv_bal_acc.
Constructor: (container, experiments: Array<ExperimentConfig>)

mount():
  super.mount()
  this._initChart()

_initChart(): void
  → Chart.js scatter with two datasets:
    1. Clean AUC (blue dots)
    2. Adversarial bal_acc (red dots)
  → X-axis: w_adv (0 → 0.45), labelled "Adversarial Loss Weight (w_adv)"
  → Y-axis: metric value (0 → 1.0)
  → Vertical dashed line at x = 0.225 labelled "Phase Transition"
  → Annotation: "Forgetting Zone" for x > 0.225
  → tabular-nums on all number labels

_drawPhaseTransitionLine(): void
  → Chart.js annotation plugin or manual canvas draw
```

#### `TRADESBetaSweepChart.js`
```
Extends BaseComponent.
Molecule. Chart.js line chart — β vs expected clean AUC + adv_bal_acc.
Constructor: (container, betaResults: Array)

Default data (theoretical — from thesis Chapter 8 Table):
  β    | Clean AUC | adv_bal_acc
  1.0  | 0.865     | 0.425
  2.0  | 0.850     | 0.550
  3.0  | 0.835     | 0.650  ← target range
  6.0  | 0.800     | 0.715

_initChart(): void
  → Line chart, two datasets: Clean AUC + adv_bal_acc
  → X-axis: β (1.0 → 6.0)
  → Horizontal dashed line at adv_bal_acc = 0.65 (target)
  → v9-PENDING badge if results not yet available
```

#### `GAParameterEditor.js`
```
Extends BaseComponent.
Molecule. Three sliders for α, τ, θ with real-time validation.
Constructor: (container, params: GAParameters, onSave: Function)

render(): string
  → Three range inputs: α [0,1], τ [0.5,2.0], θ [0.3,0.7]
  → Current value display next to each slider
  → Clinical impact tooltip on each: "θ controls MEL sensitivity threshold"
  → Save button (disabled if validation fails)
  → Warning if θ set above 0.50 (increases false negatives — clinical risk)

_bindEvents():
  → slider input → update displayed value + live validation
  → save click → params.validate() → if valid → onSave(params)
```

#### Views (`LoginView`, `ClinicalView`, `ResearchView`, `AdminView`, `SystemView`)
```
All extend BaseView which extends BaseComponent.
Each view:
  - Constructor: (container, store, eventBus)
  - Has a setController(controller) method
  - render() returns the page HTML shell (nav + content areas)
  - Instantiates and mounts child components after mount()
  - Cleans up child components on unmount()

ClinicalView sections:
  1. Image upload panel (drag-and-drop + file input)
  2. Diagnosis result area (DiagnosisResultCard + GradCAMViewer)
  3. Case history table (last 10 cases)

ResearchView sections:
  1. KPI row: 4x MetricsKPICard (AUC, bal_acc, sens, spec vs. targets)
  2. Version history table (ExperimentTableRow × 6 versions)
  3. Trade-off chart (TradeoffChart)
  4. TRADES panel (TRADESBetaSweepChart + v9-PENDING banner)
  5. PGD attack simulator (config form + result display)

AdminView sections:
  1. Usage stats row (scan volume, MEL rate, referral rate)
  2. Model health chart (AUC trend over time)
  3. Audit log table (filterable, paginated)

SystemView sections:
  1. Model registry (ModelVersion cards with activate/deprecate actions)
  2. GA parameter editor (GAParameterEditor component)
  3. Runtime health (GPU/CPU, latency, queue depth)
```

---

## 7. Mock Data Fixtures

Location: `shared/mock-fixtures/`

### `metrics.json`
```json
{
  "stage1": {
    "auc": 0.8741, "balancedAccuracy": 0.7983,
    "melanomaSensitivity": 0.7549, "nonMelSpecificity": 0.8416,
    "modelVersion": "stage1", "evaluationType": "clean"
  },
  "v8_clean": {
    "auc": 0.8711, "balancedAccuracy": 0.7515,
    "melanomaSensitivity": 0.5573, "nonMelSpecificity": 0.9457,
    "modelVersion": "v8", "evaluationType": "clean"
  },
  "v8_adversarial": {
    "auc": null, "balancedAccuracy": 0.1432,
    "melanomaSensitivity": 0.085, "nonMelSpecificity": null,
    "modelVersion": "v8", "evaluationType": "adversarial"
  },
  "v8_ga": {
    "auc": 0.8711, "balancedAccuracy": 0.7980,
    "melanomaSensitivity": 0.7550, "nonMelSpecificity": 0.8416,
    "modelVersion": "v8_ga", "evaluationType": "clean"
  }
}
```

### `version-history.json`
```json
[
  { "version": "v3", "advLossWeight": 0.40, "epsilon": 0.03, "pgdSteps": 5, "pgdAlpha": 0.007, "lr": 2e-5, "epochs": 10, "bestAUC": 0.8253, "bestBalAcc": 0.5876, "bestSensMel": 0.221, "status": "partial" },
  { "version": "v4", "advLossWeight": 0.40, "epsilon": 0.03, "pgdSteps": 5, "pgdAlpha": 0.003, "lr": 5e-6, "epochs": 5,  "bestAUC": 0.7873, "bestBalAcc": 0.5018, "bestSensMel": 0.004, "status": "failed" },
  { "version": "v5", "advLossWeight": 0.25, "epsilon": 0.01, "pgdSteps": 7, "pgdAlpha": 0.003, "lr": 1.5e-5,"epochs": 15, "bestAUC": 0.7912, "bestBalAcc": 0.6374, "bestSensMel": 0.293, "status": "partial" },
  { "version": "v6", "advLossWeight": 0.25, "epsilon": 0.01, "pgdSteps": 7, "pgdAlpha": 0.003, "lr": 1e-5,  "epochs": 20, "bestAUC": 0.8100, "bestBalAcc": 0.6101, "bestSensMel": 0.198, "status": "partial" },
  { "version": "v7", "advLossWeight": 0.15, "epsilon": 0.02, "pgdSteps": 7, "pgdAlpha": 0.003, "lr": 8e-6,  "epochs": 10, "bestAUC": 0.8311, "bestBalAcc": 0.6443, "bestSensMel": 0.287, "status": "partial" },
  { "version": "v8", "advLossWeight": 0.05, "epsilon": 0.01, "pgdSteps": 3, "pgdAlpha": 0.003, "lr": 5e-6,  "epochs": 8,  "bestAUC": 0.8711, "bestBalAcc": 0.7515, "bestSensMel": 0.5573,"status": "active" }
]
```

### `ga-parameters.json`
```json
{
  "alpha": 0.5467,
  "tau": 0.7671,
  "theta": 0.3985,
  "recoveredBalAcc": 0.7980,
  "generations": 30,
  "populationSize": 20,
  "fitnessMetric": "balanced_accuracy"
}
```

---

## 8. Design System (CSS Tokens)

Apply these exact tokens in `style.css`. No other colour values anywhere in the codebase.

### Palette — Nexus (Clinical variant)
```css
:root, [data-theme="light"] {
  /* Surfaces */
  --color-bg:             #f7f6f2;
  --color-surface:        #f9f8f5;
  --color-surface-2:      #fbfbf9;
  --color-surface-offset: #f3f0ec;
  --color-divider:        #dcd9d5;
  --color-border:         #d4d1ca;

  /* Text */
  --color-text:           #28251d;
  --color-text-muted:     #7a7974;
  --color-text-faint:     #bab9b4;
  --color-text-inverse:   #f9f8f4;

  /* Primary accent — Hydra Teal (trust, clinical positive) */
  --color-primary:        #01696f;
  --color-primary-hover:  #0c4e54;
  --color-primary-highlight: #cedcd8;

  /* Clinical MEL positive (maroon — serious, not alarming) */
  --color-error:          #a12c7b;
  --color-error-highlight: #e0ced7;

  /* Warning — borderline predictions */
  --color-warning:        #964219;
  --color-warning-highlight: #ddcfc6;

  /* Success — targets met, Non-MEL */
  --color-success:        #437a22;
  --color-success-highlight: #d4dfcc;

  /* Robustness tiers */
  --color-tier-baseline:  #d19900;   /* Altana Gold */
  --color-tier-partial:   #da7101;   /* Costa Orange */
  --color-tier-trades:    #01696f;   /* Hydra Teal */

  /* Typography */
  --font-body:    'Satoshi', 'Inter', sans-serif;
  --font-display: 'Instrument Serif', Georgia, serif;
  --font-mono:    'JetBrains Mono', 'Fira Code', monospace;

  /* Spacing (4px base) */
  --space-1: 0.25rem; --space-2: 0.5rem;  --space-3: 0.75rem;
  --space-4: 1rem;    --space-6: 1.5rem;  --space-8: 2rem;
  --space-10: 2.5rem; --space-12: 3rem;   --space-16: 4rem;

  /* Type scale (fluid) */
  --text-xs:   clamp(0.75rem,  0.7rem  + 0.25vw, 0.875rem);
  --text-sm:   clamp(0.875rem, 0.8rem  + 0.35vw, 1rem);
  --text-base: clamp(1rem,     0.95rem + 0.25vw, 1.125rem);
  --text-lg:   clamp(1.125rem, 1rem    + 0.75vw, 1.5rem);
  --text-xl:   clamp(1.5rem,   1.2rem  + 1.25vw, 2.25rem);

  /* Radius */
  --radius-sm: 0.375rem; --radius-md: 0.5rem;
  --radius-lg: 0.75rem;  --radius-xl: 1rem;
  --radius-full: 9999px;

  /* Shadows */
  --shadow-sm: 0 1px 2px oklch(0.2 0.01 80 / 0.06);
  --shadow-md: 0 4px 12px oklch(0.2 0.01 80 / 0.08);
  --shadow-lg: 0 12px 32px oklch(0.2 0.01 80 / 0.12);

  /* Transitions */
  --transition: 180ms cubic-bezier(0.16, 1, 0.3, 1);

  /* Numeric data — always tabular */
  --font-numeric: 'Satoshi', monospace;
  --font-feature-numeric: "tnum" 1, "lnum" 1;
}
```

### Numbers Rule
All metric values (`AUC`, `bal_acc`, `ε`, `w_adv`, etc.) must use:
```css
font-variant-numeric: tabular-nums lining-nums;
font-feature-settings: "tnum" 1, "lnum" 1;
```

---

## 9. API Contract (FastAPI → Frontend)

### POST `/api/predict`
```
Request:  multipart/form-data { image: File }
Response: {
  image_id: string,
  prediction: "MEL" | "NON_MEL",
  confidence: number,       // softmax probability
  threshold: number,        // current θ from GA
  model_version: string,    // "v8-pgd-at" | "v9-trades"
  robustness_tier: string,  // "BASELINE" | "PARTIAL" | "TRADES"
  inference_time_ms: number,
  timestamp: string         // ISO 8601
}
```

### GET `/api/gradcam/{image_id}`
```
Response: {
  image_id: string,
  heatmap_data: number[][],  // 14x14 attention map (ViT) or 7x7 (ResNet)
  original_url: string,
  overlay_url: string
}
```

### GET `/api/experiments`
```
Response: Array<{
  version: string,
  adv_loss_weight: number,
  epsilon: number,
  pgd_steps: number,
  pgd_alpha: number,
  lr: number,
  epochs: number,
  best_auc: number,
  best_bal_acc: number,
  best_sens_mel: number,
  adv_bal_acc: number | null,
  status: "active" | "partial" | "failed" | "pending"
}>
```

### PUT `/api/system/parameters`
```
Request:  { alpha: number, tau: number, theta: number }
Response: { success: boolean, applied_bal_acc: number }
```

---

## 10. GitHub Actions — CI/CD

### `.github/workflows/deploy-ui.yml`
```yaml
name: Deploy UI to Azure Static Web Apps
on:
  push:
    branches: [main]
    paths: ['ui/**']

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Deploy to Azure Static Web Apps
        uses: Azure/static-web-apps-deploy@v1
        with:
          azure_static_web_apps_api_token: ${{ secrets.AZURE_STATIC_WEB_APPS_API_TOKEN }}
          repo_token: ${{ secrets.GITHUB_TOKEN }}
          action: "upload"
          app_location: "/ui"
          output_location: ""
```

### `.github/workflows/deploy-api.yml`
```yaml
name: Deploy API to Azure Container Apps
on:
  push:
    branches: [main]
    paths: ['api/**']

jobs:
  build-push-deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Login to Azure Container Registry
        uses: azure/docker-login@v1
        with:
          login-server: ${{ secrets.ACR_LOGIN_SERVER }}
          username: ${{ secrets.ACR_USERNAME }}
          password: ${{ secrets.ACR_PASSWORD }}
      - name: Build and push Docker image
        run: |
          docker build -t ${{ secrets.ACR_LOGIN_SERVER }}/hare-api:${{ github.sha }} ./api
          docker push ${{ secrets.ACR_LOGIN_SERVER }}/hare-api:${{ github.sha }}
      - name: Deploy to Azure Container Apps
        uses: azure/container-apps-deploy-action@v2
        with:
          appSourcePath: ${{ github.workspace }}/api
          acrName: ${{ secrets.ACR_NAME }}
          containerAppName: hare-api
          resourceGroup: ${{ secrets.AZURE_RESOURCE_GROUP }}
          imageToDeploy: ${{ secrets.ACR_LOGIN_SERVER }}/hare-api:${{ github.sha }}
```

---

## 11. Development Phases

| Phase | Deliverable | Files Created | Est. |
|-------|------------|---------------|------|
| **0 — Design Proof** | `design-test.html` — validate tokens, fonts, surfaces | 1 file | 0.5d |
| **1 — Domain + Infra** | All domain value objects + service layer + ServiceLocator | 13 files | 1d |
| **2 — Application** | Router, StateStore, EventBus, AuthGuard, all controllers | 9 files | 0.5d |
| **3 — Base Components** | BaseComponent + all atomic components | 6 files | 0.5d |
| **4 — Clinical Portal** | ClinicalView + DiagnosisResultCard + GradCAMViewer | 4 files | 1d |
| **5 — Research Dashboard** | ResearchView + ExperimentTable + TradeoffChart + TRADES panel | 5 files | 1.5d |
| **6 — Admin + System** | AdminView + SystemView + GAParameterEditor | 4 files | 1d |
| **7 — Backend API** | FastAPI app + all service classes + routers | 15 files | 2d |
| **8 — CI/CD + Deploy** | GitHub Actions + Azure Static Web Apps config | 3 files | 0.5d |

**Total estimated: ~8 days**

---

## 12. Coding Standards

### Naming Conventions
| Element | Convention | Example |
|---------|-----------|---------|
| Classes | PascalCase | `DiagnosisResultCard` |
| Methods | camelCase, verb-first | `submitImage()`, `fetchGradCAM()` |
| Private methods | `_camelCase` | `_drawHeatmap()`, `_bindEvents()` |
| Private fields | `_camelCase` | `this._store`, `this._container` |
| Constants | UPPER_SNAKE | `CLINICAL_TARGETS`, `PHASE_TRANSITION_THRESHOLD` |
| CSS classes | BEM kebab-case | `.diagnosis-result-card__badge--borderline` |
| Events | `namespace:action` | `'image:selected'`, `'model:activated'` |
| Files | PascalCase.js | `DiagnosisResult.js`, `ClinicalView.js` |

### Method Length Rule
**No method exceeds 20 lines of logic.** If it does → extract a `_privateHelper()`.

### Comment Style
- JSDoc on every public class and public method
- Inline comments only for non-obvious logic
- No commented-out code in commits

### Error Handling
```javascript
// ALL async methods follow this pattern:
async submitImage(file) {
  try {
    // ... logic
  } catch (err) {
    this.handleError(err); // delegates to BaseController → EventBus.emit('error', ...)
  }
}
```

### ES Module Imports
```javascript
// Always use explicit file extensions in import paths
import { DiagnosisResult } from '../domain/DiagnosisResult.js';
import { RobustnessTier }  from '../domain/RobustnessTier.js';
// Never use bare specifiers (no bundler = no module resolution)
```

---

## 13. Cursor Prompt Templates

Use these as starting prompts when developing each layer:

**Domain layer:**
> "Following the HARE Platform plan section 6.2, implement `ui/domain/DiagnosisResult.js`. Apply SRP: this class only wraps prediction data. Include JSDoc. No imports from other layers."

**Infrastructure layer:**
> "Following the HARE Platform plan section 6.1, implement `ui/infrastructure/MockApiService.js`. It must extend `BaseApiService`, return data shaped exactly as `HareApiService`, simulate 300–800ms async delay, and use the fixtures defined in section 7."

**Component:**
> "Following the HARE Platform plan section 6.4, implement `ui/presentation/components/DiagnosisResultCard.js`. Extend `BaseComponent`. Apply the clinical safety constraints from section 2.3. Use only CSS tokens from section 8."

**Controller:**
> "Following the HARE Platform plan section 6.3, implement `ui/application/ClinicalController.js`. Depend on `IPredictionService` abstraction via constructor injection (DIP). Never import `HareApiService` directly."

