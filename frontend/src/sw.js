const CACHE_NAME = 'plotwise-v1';
const STATIC_ASSETS = [
  '/',
  '/static/manifest.json',
  'https://fonts.googleapis.com/css2?family=Raleway:wght@300;400;500;600;700;800;900&family=Nunito:wght@300;400;500;600;700&display=swap',
  'https://unpkg.com/aos@2.3.1/dist/aos.css',
  'https://unpkg.com/aos@2.3.1/dist/aos.js',
  'https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js',
];

// Install — cache static assets
self.addEventListener('install', (e) => {
  e.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(STATIC_ASSETS))
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

// Fetch — network first for API, cache first for static
self.addEventListener('fetch', (e) => {
  const url = new URL(e.request.url);

  // API calls & POST requests: network only (disease detection needs live server)
  if (e.request.method !== 'GET' || url.pathname.startsWith('/api') ||
      url.pathname.startsWith('/disease') || url.pathname.startsWith('/prices') ||
      url.pathname.startsWith('/calendar') || url.pathname.startsWith('/schemes') ||
      url.pathname.startsWith('/dashboard')) {
    return;
  }

  // Static assets: cache first, fallback to network
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
    }).catch(() => {
      // Offline fallback for navigation
      if (e.request.mode === 'navigate') {
        return caches.match('/');
      }
    })
  );
});
