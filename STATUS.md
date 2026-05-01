# Plotwise — Project Status
**Author:** Limawapang Jamir
**Goal:** Agriculture intelligence platform for Nagaland (B2G deployment)

---

## What Was Built

A full-stack smart farming platform with:

- **Backend** (`backend/src/main.py`) — FastAPI server with these endpoints:
  - `GET /` — health check
  - `POST /disease/detect` — upload crop leaf image → get disease + treatment advice
  - `GET /prices` — market prices (real MSP/Agmarknet 2023-24 data)
  - `GET /calendar` — planting/harvest calendar by district
  - `POST /schemes` — government schemes a farmer qualifies for
  - `GET /dashboard/yield` — yield analytics by district/crop
  - `GET /dashboard/disease-heatmap` — disease report heatmap from SQLite
  - `GET /districts`, `GET /crops`, `GET /district/{name}`

- **Frontend** (`frontend/src/index.html`) — Single-page HTML/CSS/JS app with Chart.js

- **ML Model** (`ml/saved_models/disease_model.h5`) — EfficientNetB0, trained on Google Colab
  - **98.83% test accuracy**
  - 6 classes: Chilli_LeafCurl, Healthy, Maize_CommonRust, Maize_NorthernLeafBlight, Potato_EarlyBlight, Potato_LateBlight
  - 400 train / 100 test images per class (PlantVillage dataset)

- **Data** — Real Nagaland agriculture data 2023-24 (verified records)
  - `data/sample/nagaland_agriculture_2023_24.json`
  - `data/sample/nagaland_crop_data_2023_24.csv`

---

## What's Happening Right Now (Today, 2026-03-19)

Trying to get the backend running locally for the first time.

### Issues hit and fixed today:
1. ✅ `pip install -r requirements.txt` — run from Anaconda Prompt (not Cursor)
2. ✅ `uvicorn backend.src.main:app` — server crashed silently on startup
3. ✅ Root cause: `optree` C extension DLL was crashing (access violation) → fixed with `pip install "optree==0.12.1" --force-reinstall`
4. ⏳ **Current issue:** Model `.h5` was saved with Keras 2 (on Colab) but local machine has Keras 3 (comes with TF 2.17) → shapes format mismatch error

### Current blocker:
Model was saved on Colab with a newer Keras 3.x format, but local Keras has a version mismatch loading it.

**Errors hit and what was tried:**
- `tf.keras.models.load_model` → Keras 3 shape tuple bug: `Cannot convert '((None, 7, 7, 1280),)' to a shape`
- `tf_keras.models.load_model` (Keras 2 compat layer) → `Unrecognized keyword arguments: ['batch_shape']`

**Working state right now:**
- `import tensorflow as tf` ✅ (TF 2.17.0)
- Model loading ❌ (Keras version mismatch with .h5 file)

**Next step to try:**
```
pip show keras
```
Then likely upgrade keras to a newer 3.x that fixes the H5 loading bug, OR re-save the model on Colab in `.keras` format instead of `.h5`.

---

## What Comes Next (in order)

- [ ] Finish fixing model loading (tf-keras compatibility)
- [ ] Start the server: `uvicorn backend.src.main:app`
- [ ] Test disease detection endpoint (upload a leaf image)
- [ ] Open `frontend/src/index.html` in browser and test the full app
- [ ] Deploy to a server (so farmers can actually access it)
- [ ] PWA / offline mode — critical for rural low-connectivity areas
- [ ] Nagamese language support
- [ ] Android APK via Capacitor

---

## How to Start the Server (once fixed)

Open **Anaconda Prompt**:
```
D:
cd projects\plotwise
uvicorn backend.src.main:app
```
Then open: http://127.0.0.1:8000

**Do NOT use `--reload`** — causes TF subprocess issues on Windows.

---

## Key Notes
- Python 3.11.5, TF 2.17.0 (not 2.21 — had DLL errors)
- Model loads with `compile=False`
- If disease model fails to load, the app still works — falls back to knowledge base for disease detection
