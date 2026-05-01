# 🌾 Plotwise — Smart Farming Platform for Nagaland

> An AI-powered full-stack farming platform built specifically for Nagaland farmers — combining crop disease detection, real-time market prices, planting calendars, government scheme discovery, and a department-level analytics dashboard.

**Author:** Limawapang Jamir  
**Built for:** Nagaland Agriculture Sector  
**Stack:** Python · FastAPI · TensorFlow · EfficientNet · HTML/CSS/JS · Chart.js

---

## 🎯 Why This Exists

Nagaland is still a food deficit state despite over 60% of its population being farmers. The core problems:

- Farmers sell cheaply to middlemen because they don't know current market prices
- Crop diseases go undetected until it's too late
- Farmers miss out on government schemes they qualify for
- The Agriculture Department has no unified data view across districts

**Plotwise solves all four.**

---

## ✨ Features

### For Farmers
| Feature | Description |
|---------|-------------|
| 🔬 Disease Detection | Upload a leaf photo → AI identifies disease + treatment |
| 📈 Live Prices | Real-time mandi prices across Nagaland districts |
| 📅 Planting Calendar | District-specific sow/harvest windows for 10 crops |
| 🏛️ Scheme Finder | Enter crop + district → see all schemes you qualify for |

### For the Agriculture Department
| Feature | Description |
|---------|-------------|
| 📊 Yield Analytics | District-wise crop production trends across years |
| 🗺️ Disease Heatmap | Which crops/districts have most disease reports |
| 💰 Scheme Tracking | Disbursement data and farmer enrollment |

---

## 📁 Project Structure

```
plotwise/
│
├── backend/
│   └── src/
│       └── main.py              # FastAPI backend — all endpoints
│
├── frontend/
│   └── src/
│       └── index.html           # Full single-page web app
│
├── ml/
│   └── train_disease_model.py   # EfficientNetB0 crop disease model
│
├── data/sample/                 # Sample crop data CSVs
├── docs/                        # API documentation
│
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install
```bash
git clone https://github.com/Limajmr8/plotwise.git
cd plotwise
pip install -r requirements.txt
```

### 2. Start the API
```bash
uvicorn backend.src.main:app --reload --port 8000
```

### 3. Open the app
```bash
# Just open in your browser:
open frontend/src/index.html
```

That's it. The app runs locally without any configuration.

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| POST | `/disease/detect` | Upload leaf image → disease prediction |
| GET | `/prices` | Market prices (optional: crop, district filter) |
| GET | `/calendar` | Planting calendar by district |
| POST | `/schemes` | Find government schemes for crop + district |
| GET | `/dashboard/yield` | Yield analytics for department dashboard |
| GET | `/districts` | List all Nagaland districts |
| GET | `/crops` | List all supported crops |

### Example: Detect disease
```bash
curl -X POST http://localhost:8000/disease/detect \
  -F "file=@leaf.jpg" \
  -F "crop=Rice"
```

### Example: Get prices
```bash
curl "http://localhost:8000/prices?district=Kohima&crop=Ginger"
```

### Example: Find schemes
```bash
curl -X POST http://localhost:8000/schemes \
  -H "Content-Type: application/json" \
  -d '{"district": "Mokokchung", "crop": "Rice", "land_acres": 2.5}'
```

---

## 🧠 Disease Detection Model

The ML model uses **EfficientNetB0** with transfer learning trained on the PlantVillage dataset (54,000 images), filtered to crops relevant to Nagaland:

- Rice Blast, Bacterial Blight, Brown Spot
- Maize Gray Leaf Spot, Northern Leaf Blight
- Ginger Soft Rot, Bacterial Wilt
- Potato Early/Late Blight
- Chilli Leaf Curl
- Healthy (negative class)

```bash
# Train the disease model
python ml/train_disease_model.py --epochs 20 --batch 32
```

---

## 🗺️ Nagaland Coverage

All 12 districts supported with district-specific crop recommendations:
Kohima · Dimapur · Mokokchung · Wokha · Zunheboto · Tuensang · Mon · Phek · Peren · Kiphire · Longleng · Noklak

---

## 🔮 Roadmap

- [ ] Offline mode (PWA) for low-connectivity areas
- [ ] Nagamese language support
- [ ] SMS-based alerts for disease outbreaks
- [ ] Integration with state APMC price API
- [ ] Android APK via Capacitor
- [ ] Farmer profile and yield history tracking

---

## 👤 Author

**Limawapang Jamir** | B.Tech CSE, Bennett University (2020–2024)  
From Mokokchung, Nagaland  
📧 limawapang8@gmail.com | [LinkedIn](https://linkedin.com/in/limajmr8) | [GitHub](https://github.com/Limajmr8)

*Built with the goal of putting real tools in the hands of Nagaland's farming community.*
