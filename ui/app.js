import { ServiceLocator } from './infrastructure/ServiceLocator.js';
import { StateStore } from './application/StateStore.js';
import { EventBus } from './application/EventBus.js';
import { AuthGuard } from './application/AuthGuard.js';
import { Router } from './application/Router.js';

import { LoginView } from './presentation/views/LoginView.js';
import { ClinicalView } from './presentation/views/ClinicalView.js';
import { ResearchView } from './presentation/views/ResearchView.js';
import { ThesisView } from './presentation/views/ThesisView.js';
import { AdminView } from './presentation/views/AdminView.js';
import { SystemView } from './presentation/views/SystemView.js';

import { ClinicalController } from './application/ClinicalController.js';
import { ResearchController } from './application/ResearchController.js';
import { ThesisController } from './application/ThesisController.js';
import { AdminController } from './application/AdminController.js';
import { SystemController } from './application/SystemController.js';

/**
 * HARE Platform bootstrap.
 * Configures DI container, registers routes, initialises the router.
 */
async function bootstrap() {
  const searchParams = new URLSearchParams(window.location.search);
  const env = searchParams.get('env') === 'production'
    ? 'production'
    : 'mock';
  const apiBaseUrl = searchParams.get('apiBaseUrl')
    || window.HARE_API_BASE_URL
    || (window.location.hostname === 'localhost' ? 'http://localhost:8000' : '');

  await ServiceLocator.configure(env, { apiBaseUrl });

  const store = new StateStore();
  ServiceLocator.register('store', store);
  ServiceLocator.register('eventBus', EventBus);

  // Expose for LoginView (no bundler = no dynamic import workaround)
  window.__ServiceLocator = ServiceLocator;

  const authGuard = new AuthGuard(store);
  const router = new Router(authGuard);

  router.register('#/login', LoginView, null, null);
  router.register('#/clinical', ClinicalView, ClinicalController, 'clinician');
  router.register('#/thesis', ThesisView, ThesisController, 'research');
  router.register('#/research', ResearchView, ResearchController, 'research');
  router.register('#/admin', AdminView, AdminController, 'admin');
  router.register('#/system', SystemView, SystemController, 'system');

  ServiceLocator.register('router', router);

  _initNotificationListener();

  router.init();
}

function _initNotificationListener() {
  EventBus.on('notification', ({ type, message }) => {
    const slot = document.getElementById('notification-slot');
    if (!slot) return;
    const toast = document.createElement('div');
    toast.className = `notification-toast notification-toast--${type}`;
    toast.textContent = message;
    slot.appendChild(toast);
    setTimeout(() => toast.remove(), 4000);
  });

  EventBus.on('error', ({ message }) => {
    const slot = document.getElementById('notification-slot');
    if (!slot) return;
    const toast = document.createElement('div');
    toast.className = 'notification-toast notification-toast--error';
    toast.textContent = message;
    slot.appendChild(toast);
    setTimeout(() => toast.remove(), 5000);
  });
}

bootstrap().catch(err => {
  console.error('HARE Platform bootstrap failed:', err);
  const content = document.getElementById('app-content');
  if (content) {
    content.innerHTML = `
      <div style="text-align:center;padding:4rem">
        <h1>HARE Platform</h1>
        <p>Failed to initialise. Check the console for details.</p>
        <pre style="color:red">${err.message}</pre>
      </div>
    `;
  }
});
