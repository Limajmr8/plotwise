# Changelog

All notable changes to Plotwise are documented in this file.

---

## [2.0.0] — 2026-05-16

**Production-ready release for June 2026 B2G demo.**

### Added
- **Environment configuration** — .env support (CORS_ORIGINS, PLOTWISE_DB_PATH, MODEL_PATH, LOG_LEVEL)
- **Weather API caching** — 15-minute TTL, 77x speedup on repeat requests
- **Security headers** — X-Content-Type-Options, X-Frame-Options, HSTS, Referrer-Policy, Permissions-Policy
- **Rate limiting** — all endpoints protected (30/min data, 10/min exports, 5/min detection)
- **Shared JS module** (`plotwise-core.js`) — eliminates code duplication between desktop and mobile
- **API fetch wrapper** — timeout (AbortController), retry with backoff, in-memory caching, localStorage offline fallback
- **Standardized loading/error UI** — `pwLoading()` and `pwError()` helpers across all screens
- **Demo user profiles** — lightweight name/role/district entry (localStorage, no passwords)
- **Reporter tracking** — every disease detection logged with who reported it
- **Audit logging middleware** — request counts, errors, endpoint usage in /health
- **Enhanced PDF reports** — branding, "Prepared for: Director of Agriculture", disease surveillance section, page numbers
- **Disease heatmap** — recent reports section with reporter identity
- **Docker optimization** — separate COPY layers (deps → model → data → code), HEALTHCHECK
- **Seed data script** (`scripts/seed_demo_data.py`) — 80+ realistic disease reports for demo
- **API test suite** (`tests/test_api.py`) — 40+ tests across 13 test classes
- **Demo checklist** (`docs/DEMO_CHECKLIST.md`) — complete pre-demo verification workflow
- **Demo script** (`docs/DEMO_SCRIPT.md`) — 15-minute presentation flow for government stakeholders
- **Signed APK workflow** — GitHub Actions supports release keystore via secrets
- **Capacitor Camera plugin** — native camera in Android app, file input fallback on web
- **Capacitor Network plugin** — native network status listener for offline detection
- **Offline graceful degradation** — cached data with offline bar, retry buttons

### Changed
- Bumped app version to 2.0.0 (versionCode 3)
- Service worker updated to v3 (caches plotwise-core.js)
- Mobile UI: all fetch calls use apiFetch() with appropriate TTLs
- .dockerignore expanded (android/, node_modules/, docs/, tests/, scripts/)

### Fixed
- Weather requests no longer hang the demo (cached after first call)
- Tab switching no longer re-fetches within cache TTL
- Disease detection correctly passes reporter info to backend
- Inconsistent error messages replaced with standard retry UI

---

## [1.1.0] — 2026-05-13

### Added
- Mobile-optimized app UI (`mobile.html`) with bottom tab navigation
- Capacitor Android app (com.plotwise.app)
- GitHub Actions CI/CD for APK builds
- PDF export endpoint + buttons in both UIs
- Responsive breakpoints (768px + 480px)
- ARIA labels for accessibility
- CORS whitelisting and rate limiting (slowapi)
- File upload size limits
- Global error handler

### Changed
- Repo cleaned: 116MB of binary artifacts removed from git history

---

## [1.0.0] — 2026-05-10

### Added
- Initial release: FastAPI backend with 19 API endpoints
- EfficientNetB0 disease detection (24 classes, 9 crops)
- Real 2023-24 agriculture data (576 records, 16 districts, 44 crops)
- Market prices with MSP anchors (33 crops)
- Planting calendar with seasonal status badges
- Government scheme finder with eligibility matching
- Weather integration (Open-Meteo, all 16 districts)
- Chat with 7 intent types + Nagamese support
- Premium landing page with GSAP animations
- PWA support (manifest, service worker, offline shell)
- Railway deployment
