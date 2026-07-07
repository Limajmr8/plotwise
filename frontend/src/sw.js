const CACHE_NAME = 'plotwise-v5';

// Core shell + offline stack. Anything here is available with zero network
// after the first online load — including the on-device AI model.
const STATIC_ASSETS = [
  '/?desktop=1',
  '/mobile',
  '/static/plotwise-core.js',
  '/static/manifest.json',
  '/static/icon-192.png',
  '/static/icon-512.png',
  '/static/chart.umd.min.js',
  // Offline stack
  '/static/offline-data.js',
  '/static/offline-engine.js',
  '/static/offline-ai.js',
  '/static/tf.min.js',
  '/static/tfjs-model/model.json',
  '/static/tfjs-model/group1-shard1of3.bin',
  '/static/tfjs-model/group1-shard2of3.bin',
  '/static/tfjs-model/group1-shard3of3.bin',
  // Desktop-only extras (best-effort)
  'https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=DM+Sans:wght@300;400;500;600&display=swap',
  'https://cdn.jsdelivr.net/npm/gsap@3.12.5/dist/gsap.min.js',
  'https://cdn.jsdelivr.net/npm/gsap@3.12.5/dist/ScrollTrigger.min.js',
  'https://unpkg.com/lenis@1.1.18/dist/lenis.min.js',
];

// Install — cache resiliently: one missing asset must not fail the whole SW.
self.addEventListener('install', (e) => {
  e.waitUntil(
    caches.open(CACHE_NAME).then((cache) =>
      Promise.allSettled(STATIC_ASSETS.map((a) => cache.add(a).catch(() => {})))
    )
  );
  self.skipWaiting();
});

// Activate — clean old caches
self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k)))
    )
  );
  self.clients.claim();
});

self.addEventListener('fetch', (e) => {
  const url = new URL(e.request.url);

  // Live API calls & POSTs: network only (never cache prices/detect/etc.).
  if (e.request.method !== 'GET' || url.pathname.startsWith('/api') ||
      url.pathname.startsWith('/disease') || url.pathname.startsWith('/prices') ||
      url.pathname.startsWith('/calendar') || url.pathname.startsWith('/schemes') ||
      url.pathname.startsWith('/dashboard') || url.pathname.startsWith('/health')) {
    return;
  }

  // Page navigations: network-first so server redirects (phone -> /mobile) and
  // fresh HTML win; fall back to the cached shell when offline.
  if (e.request.mode === 'navigate') {
    e.respondWith(
      fetch(e.request).catch(() => {
        const fallback = url.pathname.startsWith('/mobile') ? '/mobile' : '/?desktop=1';
        return caches.match(fallback).then((c) => c || caches.match('/mobile'));
      })
    );
    return;
  }

  // Static assets (incl. the tfjs model shards): cache-first, then network,
  // and store on first fetch so a partial precache still fills in over time.
  e.respondWith(
    caches.match(e.request).then((cached) => {
      if (cached) return cached;
      return fetch(e.request).then((res) => {
        if (res.ok) {
          const clone = res.clone();
          caches.open(CACHE_NAME).then((cache) => cache.put(e.request, clone));
        }
        return res;
      });
    })
  );
});
