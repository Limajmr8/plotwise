---
title: Plotwise
emoji: 🌾
colorFrom: green
colorTo: gray
sdk: docker
app_port: 8080
pinned: false
---

# 🌾 Plotwise — Smart Farming Platform for Nagaland

> An AI-powered farming platform built for Nagaland — crop disease detection, market price tracking, planting calendars, government scheme discovery, and a department-level analytics dashboard.

**Author:** Limawapang L Jamir
**Built for:** Nagaland Agriculture Department (B2G)
**Stack:** Python · FastAPI · TensorFlow (EfficientNetB0) · HTML/CSS/JS · Chart.js · Capacitor (Android)

---

## 🎯 Why This Exists

Nagaland is still a food deficit state despite over 60% of its population farming the land. The core problems:

- Farmers sell cheaply to middlemen because they don't know market prices
- Crop diseases go undetected until it's too late
- Farmers miss out on government schemes they qualify for
- The Agriculture Department has no unified data view across districts

**Plotwise addresses all four.**

---

## ✨ Features

### For Farmers
| Feature | Description |
|---------|-------------|
| 🔬 Disease Detection | Photograph a leaf → AI identifies the disease, severity, and treatment |
| 📈 Market Prices | Indicative mandi prices anchored to MSP 2023-24 + Agmarknet APMC rates (33 crops) |
| 📅 Planting Calendar | District-specific sow/harvest windows with live seasonal status |
| 🏛️ Scheme Finder | Enter crop + district → matching government schemes |
| 🌤️ Weather | Live conditions + 7-day forecast with farming advisories (all 16 district HQs) |
| 💬 Assistant | Chat in English or Nagamese about prices, diseases, planting, schemes |

### For the Agriculture Department
| Feature | Description |
|---------|-------------|
| 📊 Yield Analytics | District-wise area/production/yield from real 2023-24 records |
| 🗺️ Disease Heatmap | Live surveillance map of disease reports with reporter identity |
| 📄 PDF Reports | One-click district intelligence report (production, surveillance, prices) |
| 📋 Audit Trail | Every detection logged — who reported, where, when |

---

## 📊 Data

- **Crop records**: 576 real records — "District-wise Area, Production and Yield (Achievement) 2023-24", Director of Agriculture, Nagaland. **16 districts, 44 crops.**
- **Market prices**: MSP 2023-24 (CACP) + Agmarknet APMC anchors for 33 crops. Prices shown are *indicative*, not a live market feed.
- **Weather**: Open-Meteo (no API key), GPS coordinates for all 16 district HQs.
- **Disease model**: PlantVillage dataset (Penn State), filtered to Nagaland-relevant crops.

## 🧠 ML Model

**EfficientNetB0** (transfer learning) — **24 classes across 9 crop types**:
Apple, Chilli, Grape, Maize, Orange, Pepper, Potato, Soybean, Tomato (plus healthy-leaf classes).

Three-tier confidence: **confident** (≥70%), **low confidence** (55–70%, flagged for expert verification), **uncertain** (<55% or ambiguous top-2 — the app says so honestly instead of guessing). Crops outside the model (e.g. paddy, ginger) fall back to a curated knowledge base.

---

## 🚀 Quick Start (local)

```bash
git clone https://github.com/Limajmr8/plotwise.git
cd plotwise
pip install -r requirements.txt
uvicorn backend.src.main:app --port 8000
```

Then open **http://127.0.0.1:8000** (desktop UI) or **/mobile** (mobile UI).

> ⚠️ Do **not** use `--reload` on Windows — it crashes the TensorFlow subprocess.
> Keep the pinned versions in `requirements.txt` (TF 2.17.0 / Keras 3.10.0) — newer versions cannot load the model.

Optional configuration via `.env` (see `.env.example`): DB path, CORS origins, model path, demo seeding.

### Docker

```bash
docker build -t plotwise .
docker run -p 8080:8080 -e PLOTWISE_SEED_ON_EMPTY=1 plotwise
```

---

## 🌐 API Endpoints (selected)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health + audit stats |
| GET | `/api/status` | App status, version, data summary |
| GET | `/districts` · `/crops` | Reference data (16 districts, 44 crops) |
| GET | `/prices` | Market prices (filters: `crop`, `district`) |
| GET | `/calendar` | Planting calendar with live status (filters: `district`, `crop`) |
| POST | `/schemes` | Scheme matching by crop/district/land |
| POST | `/disease/detect` | Leaf image upload → ML prediction (multipart) |
| GET | `/api/weather` | Live weather + 7-day forecast per district |
| POST | `/api/chat` | Assistant (EN/Nagamese intent detection) |
| GET | `/dashboard/yield` · `/dashboard/disease-heatmap` | Analytics |
| GET | `/api/export/report` | Branded district PDF report |
| GET | `/api/export/prices` · `/api/export/yield` | CSV exports |

Interactive docs at `/docs` when running locally.

### Example: Detect disease
```bash
curl -X POST http://localhost:8000/disease/detect \
  -F "file=@leaf.jpg" \
  -F "crop=Potato" -F "district=Kohima"
```

---

## 📱 Android App

Capacitor wraps the mobile-optimized UI (`frontend/src/mobile.html`) with native camera and network plugins. APKs are built by GitHub Actions on every push to `main` (signed when release keystore secrets are configured — see `docs/SIGNING_SETUP.md`).

---

## 🧪 Tests

```bash
pytest tests/ -q
```

51 tests cover every endpoint, the ML confidence tiers, security headers, exports, and edge cases. Tests run against an isolated temporary database.

---

## 🗺️ Nagaland Coverage

All **16 districts**: Kohima · Tseminyu · Phek · Mokokchung · Tuensang · Noklak · Shamator · Mon · Dimapur · Niuland · Chumoukedima · Wokha · Zunheboto · Peren · Kiphire · Longleng

---

## 🔮 Roadmap

- [x] Nagamese language support (EN/NAG toggle)
- [x] PWA + offline caching for low-connectivity areas
- [x] Android APK via Capacitor (native camera + network)
- [ ] Farmer profile and yield history tracking
- [ ] SMS-based alerts for disease outbreaks
- [ ] Integration with state APMC price API (live feed)

---

## 👤 Author

**Limawapang Jamir** | B.Tech CSE, Bennett University (2020–2024)
From Mokokchung, Nagaland
📧 limawapang8@gmail.com | [LinkedIn](https://linkedin.com/in/limajmr8) | [GitHub](https://github.com/Limajmr8)

*Built with the goal of putting real tools in the hands of Nagaland's farming community.*
