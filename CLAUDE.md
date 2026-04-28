# Plotwise — CLAUDE.md

## Project
Agricultural intelligence platform for Nagaland — crop disease detection, market prices, planting calendar, weather, and government scheme finder.
Author: Limawapang L Jamir. B2G deployment to Nagaland Agriculture Department (June 2026 presentation).

## Stack
- **Backend**: FastAPI (`backend/src/main.py`) — serves frontend + API
- **Frontend**: Single-page HTML/CSS/JS + Chart.js + GSAP + Lenis (`frontend/src/index.html`)
- **ML**: EfficientNetB0 (TensorFlow) — `ml/train_disease_model.py`
- **Model**: `ml/saved_models/disease_model.h5` + `ml/saved_models/class_indices.json`
- **Data**: Real 2023-24 data from Director of Agriculture, Nagaland (`data/sample/`)
- **Fonts**: Plus Jakarta Sans (headings) + DM Sans (body)
- **Animation**: GSAP 3.12.5 + ScrollTrigger + Lenis smooth scroll

## Data Sources
- **Crop data**: "District-wise Area, Production and Yield (Achievement) 2023-24" — Director of Agriculture, Nagaland, Kohima. Verified 09-May-2024, checked 30-Jul-2024. 576 records, 16 districts, 44 crops.
- **Market prices**: MSP 2023-24 (CACP) + Agmarknet APMC rates for Nagaland. 33 crops with price anchors.
- **Weather**: Open-Meteo (free, no API key). GPS coords for all 16 district HQs.
- **Disease detection**: PlantVillage dataset (Penn State). 24 classes, 9 crop types.

## Districts (all 16, consistent spelling everywhere)
Kohima, Tseminyu, Phek, Mokokchung, Tuensang, Noklak, Shamator, Mon, Dimapur, Niuland, Chumoukedima, Wokha, Zunheboto, Peren, Kiphire, Longleng

## ML Model
- Trained on Kaggle (PlantVillage dataset, T4 x2 GPU)
- 24 classes: Apple (2), Chilli (1), Grape (2), Healthy (5), Maize (3), Orange (1), Pepper (1), Potato (2), Soybean (1), Tomato (6)
- Architecture: EfficientNetB0 → GlobalAveragePooling2D → BatchNorm → Dropout(0.4) → Dense(256, relu) → BatchNorm → Dropout(0.3) → Dense(24, softmax)
- 3-tier confidence: confident (>=70%), low (55-70%), uncertain (<55% or small top-2 gap)
- Input: 224x224 RGB, NO manual rescale (EfficientNetB0 has built-in preprocessing)
- ML_SUPPORTED_CROPS in backend: Apple, Chilli, Grape, Maize, Orange, Pepper, Potato, Soyabean, Tomato

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
- **i18n**: EN/NAG language toggle in nav — `data-i18n` attributes, `setLang()` function, stored in localStorage
- **About section**: Data sources, methodology, developer credit — for government presentation credibility
- **Chatbot**: Floating chat bubble, `/api/chat` POST with intent detection, suggestion chips
- **Weather**: Open-Meteo, 7-day forecast, farming advisories, all 16 districts
- **Export**: CSV download for market prices and yield data
- **SEO**: Meta description, keywords, Open Graph, Twitter cards, preconnect hints

## Known Issues
- CPU lacks AVX2/AVX512 — TF prints warnings on startup, runs fine (just slower)
- Do NOT upgrade keras above 3.10.0 — newer versions can't load this model's .h5 format

## Content Policy
- **NEVER add fabricated testimonials, fake farmer names, or invented quotes** — this app will be presented to the Nagaland Agriculture Department (Joint Director level). Only use real data or clearly marked placeholders.
- **No "free" / "open source" claims** — this is a B2G product being pitched for a government contract.
- **No tech jargon in user-facing copy** — "EfficientNetB0" stays in About/technical sections only, never in marketing copy.

## Roadmap
- [x] ML model: 24-class EfficientNetB0, ~99% validation accuracy
- [x] Backend: FastAPI with real 2023-24 agriculture data (576 records)
- [x] Disease detection: 9 crop types, 3-tier confidence, treatment recommendations
- [x] Frontend: Premium Porsche-inspired design, GSAP animations
- [x] All 16 districts in every dropdown (consistent naming)
- [x] 44 crops tracked with real Area/Production/Yield data
- [x] Market prices anchored to MSP 2023-24 (33 crops)
- [x] PWA, WebP images, Nagamese i18n, chatbot, weather, CSV export
- [x] About/Methodology section for government credibility
- [x] Removed all false claims ("free", "no data collected", "works offline", "open source")
- [x] Deploy to Railway — https://plotwise-production.up.railway.app
- [ ] June 2026 department presentation (B2G pitch)
- [ ] Android APK (TWA for Play Store, then Capacitor later)
- [ ] Farmer profile + yield history
- [ ] Real farmer testimonials (after field testing)
