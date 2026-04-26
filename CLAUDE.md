# Plotwise — CLAUDE.md

## Project
Crop disease detection + agriculture dashboard for Nagaland farmers.
Author: Limawapang Jamir. Intended for real deployment to farmers + Nagaland Agriculture Department.

## Stack
- **Backend**: FastAPI (`backend/src/main.py`) — serves frontend + API
- **Frontend**: Single-page HTML/CSS/JS + Chart.js + GSAP + Lenis (`frontend/src/index.html`)
- **ML**: EfficientNetB0 (TensorFlow) — `ml/train_disease_model.py`
- **Model**: `ml/saved_models/disease_model.h5` + `ml/saved_models/class_indices.json`
- **Fonts**: Plus Jakarta Sans (headings) + DM Sans (body) — Porsche Next alternatives
- **Animation**: GSAP 3.12.5 + ScrollTrigger + Lenis smooth scroll (replaced AOS)

## ML Model
- Trained on Google Colab (PlantVillage dataset)
- 98.83% test accuracy, 6 classes: Chilli_LeafCurl, Healthy, Maize_CommonRust, Maize_NorthernLeafBlight, Potato_EarlyBlight, Potato_LateBlight
- Architecture: EfficientNetB0 → GlobalAveragePooling2D → BatchNorm → Dropout(0.4) → Dense(256, relu) → BatchNorm → Dropout(0.3) → Dense(6, softmax)
- Input: 224x224 RGB, NO manual rescale (EfficientNetB0 has built-in preprocessing)

## Running the Backend (Windows / Anaconda Prompt)
```
D:
cd projects\plotwise
uvicorn backend.src.main:app
```
- Do NOT use `--reload` (causes TF subprocess crash on Windows)
- Python 3.11.5, TF 2.17.0
- Frontend at http://127.0.0.1:8000, API status at /api/status

## Critical pip pins (DO NOT upgrade these)
```
tensorflow==2.17.0       # 2.21 breaks with DLL errors on this CPU
optree==0.12.1           # newer optree crashes with access violation
tf-keras==2.17.0         # needed as TF companion package
keras==3.10.0            # MUST match the Keras version that saved the model on Colab
                         # 3.13.2 has a regression loading .h5 files saved by 3.10.0
```

## Frontend Architecture
- Single HTML file with inline CSS + JS
- GSAP ScrollTrigger for all scroll animations (clip-path reveals, parallax, counters, dissolve)
- Lenis for smooth scroll (synced with GSAP)
- Particle canvas in hero section
- PWA: manifest.json, sw.js, icons at frontend/src/
- Static files served via FastAPI mount at /static/
- API URLs use `window.location.origin` (works locally + deployed)
- **i18n**: EN/NAG language toggle in nav — `data-i18n` attributes on ~40 elements, `setLang()` function, stored in localStorage
- **Images**: All converted to WebP (originals kept as .jpg fallback), `loading="lazy"` on below-fold `<img>` tags
- **Dissolve effect**: Feature images use blur-to-sharp focus-pull + grain overlay fade on scroll
- **Chatbot**: Floating chat bubble → frosted glass panel, `/api/chat` POST endpoint with intent detection (disease/price/planting/scheme/district/weather/greeting), word-boundary regex crop matching, contextual suggestion chips, typing indicator
- **Weather**: Live weather via Open-Meteo (free, no API key) — `/api/weather?district=X`, 7-day forecast, farming advisories, weather tab in app
- **Export**: CSV download for market prices (`/api/export/prices`) and yield data (`/api/export/yield`)
- **SEO**: Meta description, keywords, Open Graph tags, Twitter cards, preconnect hints

## Known Issues
- CPU lacks AVX2/AVX512 — TF prints warnings on startup, runs fine (just slower)
- Do NOT upgrade keras above 3.10.0 — newer versions can't load this model's .h5 format

## Content Policy
- **NEVER add fabricated testimonials, fake farmer names, or invented quotes** — this app will be presented to the Nagaland Agriculture Department (Joint Director level). Only use real data or clearly marked placeholders.

## Roadmap
- [x] Fix model loading locally — keras==3.10.0 was the key pin
- [x] Backend runs: `uvicorn backend.src.main:app` → http://127.0.0.1:8000
- [x] Disease detection endpoint tested (82% confidence)
- [x] Frontend served from backend (FileResponse at /)
- [x] PWA support (manifest, service worker, icons)
- [x] WebP image optimization (93-97% reduction)
- [x] Nagamese language toggle (EN | NAG)
- [x] Dissolve/focus-pull effects on feature images
- [x] AI farming chatbot (floating UI + intent detection + knowledge base)
- [x] Live weather integration (Open-Meteo, 7-day forecast, farming advisories)
- [x] CSV export for prices + yield data
- [x] SEO + Open Graph + preconnect optimization
- [x] Deploy to Railway — https://plotwise-production.up.railway.app
- [ ] **Frontend premium redesign (IN PROGRESS)** — Porsche-inspired
- [ ] Android APK (TWA for Play Store, then Capacitor later)
- [ ] Farmer profile + yield history
- [ ] Real farmer testimonials (after field testing)
