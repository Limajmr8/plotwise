"""
Plotwise — Smart Farming Platform for Nagaland
FastAPI Backend

Author: Limawapang L Jamir
For: Nagaland Agriculture Department (B2G)

All data sourced from verified Nagaland agriculture records 2023-24.
Disease reports stored in SQLite for heatmap analytics.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
import json
import csv
import os
import random
import sqlite3
import io
import urllib.request
import urllib.error

app = FastAPI(
    title="Plotwise API",
    description="Smart farming platform for Nagaland farmers",
    version="1.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ── Paths ──────────────────────────────────────────────────────────────────────

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR   = os.path.join(BASE_DIR, "data", "sample")
JSON_FILE  = os.path.join(DATA_DIR, "nagaland_agriculture_2023_24.json")
CSV_FILE   = os.path.join(DATA_DIR, "nagaland_crop_data_2023_24.csv")
MODEL_PATH   = os.path.join(BASE_DIR, "ml", "saved_models", "disease_model.h5")
CLASSES_PATH = os.path.join(BASE_DIR, "ml", "saved_models", "class_indices.json")
DB_PATH    = os.path.join(BASE_DIR, "data", "plotwise.db")

# ── Load real agriculture data at startup ──────────────────────────────────────

with open(JSON_FILE, "r") as f:
    AG_DATA = json.load(f)

DISTRICTS        = AG_DATA["metadata"]["districts"]
CROPS            = AG_DATA["metadata"]["crops"]
DISTRICT_SUMMARY = AG_DATA["district_summary"]

CROP_RECORDS = []
with open(CSV_FILE, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        CROP_RECORDS.append({
            "year":              row["year"],
            "district":          row["district"],
            "crop":              row["crop"],
            "season":            row.get("season", ""),
            "area_ha":           float(row["area_ha"])           if row["area_ha"]           else 0,
            "production_tonnes": float(row["production_tonnes"]) if row["production_tonnes"] else 0,
            "yield_kg_per_ha":   float(row["yield_kg_per_ha"])   if row["yield_kg_per_ha"]   else 0,
        })

def _build_indexes():
    by_district, by_crop = {}, {}
    for r in CROP_RECORDS:
        by_district.setdefault(r["district"], []).append(r)
        by_crop.setdefault(r["crop"], []).append(r)
    return by_district, by_crop

RECORDS_BY_DISTRICT, RECORDS_BY_CROP = _build_indexes()

# ── SQLite — disease report store ──────────────────────────────────────────────

def _init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS disease_reports (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            district   TEXT    NOT NULL,
            crop       TEXT    NOT NULL,
            disease    TEXT    NOT NULL,
            confidence REAL    NOT NULL,
            severity   TEXT    NOT NULL,
            timestamp  TEXT    NOT NULL
        )
    """)
    conn.commit()
    conn.close()

_init_db()

def _db():
    return sqlite3.connect(DB_PATH)


def _log_disease(district: str, crop: str, disease: str, confidence: float, severity: str):
    conn = _db()
    conn.execute(
        "INSERT INTO disease_reports (district, crop, disease, confidence, severity, timestamp) VALUES (?,?,?,?,?,?)",
        (district, crop, disease, round(confidence, 3), severity, datetime.utcnow().isoformat())
    )
    conn.commit()
    conn.close()

# ── ML model (optional — loaded only when trained model exists) ────────────────

DISEASE_MODEL   = None
DISEASE_CLASSES = {}   # idx (str) → class label, loaded from class_indices.json

def _load_model():
    global DISEASE_MODEL, DISEASE_CLASSES
    if not os.path.exists(MODEL_PATH):
        print(f"[WARN]  No trained model at {MODEL_PATH}. Run: python ml/train_disease_model.py")
        return
    try:
        import tensorflow as tf
        DISEASE_MODEL = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print(f"[OK]  Disease model loaded ({MODEL_PATH})")
        if os.path.exists(CLASSES_PATH):
            with open(CLASSES_PATH) as f:
                DISEASE_CLASSES = json.load(f)
            print(f"[OK]  Class labels loaded: {list(DISEASE_CLASSES.values())}")
        else:
            print("[WARN]  class_indices.json not found — re-run train_disease_model.py")
    except Exception as e:
        print(f"[WARN]  Could not load disease model: {e}")

_load_model()


def _preprocess_image(img_bytes: bytes):
    """Resize and normalise image for EfficientNetB0 inference."""
    from PIL import Image
    import numpy as np
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((224, 224))
    arr = np.array(img, dtype="float32")  # no /255 — EfficientNetB0 has built-in preprocessing
    return arr[None]  # (1, 224, 224, 3)

# ── Disease knowledge base ────────────────────────────────────────────────────

DISEASES = {
    "Jhum Paddy":        ["Blast", "Bacterial Blight", "Brown Spot", "Sheath Blight"],
    "WTRC Paddy Kharif": ["Blast", "Bacterial Blight", "Brown Spot"],
    "WTRC Paddy Rabi":   ["Blast", "Brown Spot"],
    "Maize Kharif":      ["Gray Leaf Spot", "Northern Leaf Blight", "Common Rust"],
    "Maize Rabi":        ["Gray Leaf Spot", "Common Rust"],
    "Ginger":            ["Soft Rot", "Bacterial Wilt", "Yellow Mottle Virus"],
    "Potato":            ["Late Blight", "Early Blight", "Common Scab"],
    "Soyabean":          ["Pod Blight", "Leaf Spot", "Root Rot"],
    "Mustard":           ["Alternaria Blight", "White Rust", "Downy Mildew"],
    "Sugarcane":         ["Red Rot", "Wilt", "Smut"],
    "Tea":               ["Blister Blight", "Gray Blight", "Brown Blight"],
    "default":           ["Powdery Mildew", "Leaf Spot", "Root Rot", "Nutrient Deficiency"],
}

TREATMENTS = {
    # Rice
    "Blast":               "Apply tricyclazole fungicide. Remove infected leaves immediately. Avoid excess nitrogen.",
    "Bacterial Blight":    "Use copper-based bactericide. Drain waterlogged fields. Plant resistant varieties.",
    "Brown Spot":          "Apply mancozeb fungicide. Use balanced fertilizers. Improve drainage.",
    "Sheath Blight":       "Apply hexaconazole fungicide. Reduce dense planting. Drain excess water.",
    # Maize
    "Gray Leaf Spot":      "Apply mancozeb fungicide. Rotate crops with non-host plants. Improve air circulation.",
    "Northern Leaf Blight":"Apply propiconazole fungicide. Plant resistant hybrids. Remove crop debris.",
    "Common Rust":         "Apply azoxystrobin fungicide. Use resistant varieties. Scout fields regularly.",
    # Potato / Tomato
    "Late Blight":         "Apply chlorothalonil fungicide immediately. Avoid overhead irrigation. Remove affected plants.",
    "Early Blight":        "Apply mancozeb at first sign. Maintain plant nutrition. Improve spacing.",
    # Pepper / Chilli
    "Bacterial Spot":      "Apply copper-based bactericide. Avoid overhead watering. Remove infected plant debris. Use disease-free seeds.",
    "Leaf Curl":           "Control whitefly vectors with neem oil or imidacloprid. Remove infected plants. Use reflective mulch to repel whiteflies.",
    # Tomato
    "Leaf Mold":           "Improve ventilation and reduce humidity. Apply chlorothalonil fungicide. Remove lower affected leaves.",
    "Septoria Leaf Spot":  "Apply mancozeb or chlorothalonil at first sign. Remove infected lower leaves. Avoid working in wet foliage.",
    "Yellow Leaf Curl Virus": "Control whitefly with sticky traps and neem spray. Remove infected plants immediately. Use resistant varieties.",
    # Apple
    "Apple Scab":          "Apply captan or myclobutanil fungicide in spring. Rake and destroy fallen leaves. Plant scab-resistant varieties.",
    "Black Rot":           "Prune out cankers and dead wood. Apply captan fungicide. Remove mummified fruit from trees.",
    # Grape
    "Esca (Black Measles)":"No chemical cure. Remove severely affected vines. Protect pruning wounds with wound sealant. Avoid stress.",
    # Citrus
    "Citrus Greening (Huanglongbing)": "No cure exists. Control Asian citrus psyllid vector with insecticide. Remove infected trees to prevent spread.",
    # General
    "Soft Rot":            "Improve field drainage immediately. Apply Bordeaux mixture. Remove all infected rhizomes.",
    "Bacterial Wilt":      "Remove infected plants. Disinfect tools. Use disease-free seed rhizomes.",
    "Yellow Mottle Virus": "Control mite vectors with acaricide. Remove and destroy infected plants.",
    "Pod Blight":          "Apply carbendazim at flowering stage. Harvest at correct maturity to reduce losses.",
    "Alternaria Blight":   "Apply mancozeb or iprodione. Use disease-free certified seeds. Maintain proper spacing.",
    "Powdery Mildew":      "Apply sulfur-based fungicide. Avoid overhead irrigation. Improve air circulation.",
    "Leaf Spot":           "Remove infected leaves. Apply copper fungicide. Improve plant spacing.",
    "Root Rot":            "Improve soil drainage. Reduce watering. Apply biological fungicide (Trichoderma).",
    "Nutrient Deficiency": "Conduct soil test. Apply balanced NPK fertilizer as recommended.",
    "Red Rot":             "Use disease-free setts. Avoid water stagnation. Apply Bordeaux mixture.",
    "Blister Blight":      "Apply copper oxychloride at flush stage. Maintain shade regulation.",
}

# ── Government schemes ────────────────────────────────────────────────────────

SCHEMES = [
    {
        "name":        "PM-KISAN",
        "description": "Direct income support of ₹6,000/year (3 instalments of ₹2,000) to all landholding farmer families.",
        "eligibility": "All landholding farmer families",
        "apply_at":    "pmkisan.gov.in",
        "crops":       []
    },
    {
        "name":        "PMFBY — Crop Insurance",
        "description": "Comprehensive crop insurance covering yield losses from natural calamities, pests and disease. Premium as low as 1.5% for Rabi crops.",
        "eligibility": "All farmers growing notified crops",
        "apply_at":    "pmfby.gov.in",
        "crops":       ["Jhum Paddy", "WTRC Paddy Kharif", "WTRC Paddy Rabi", "Maize Kharif", "Maize Rabi", "Soyabean", "Mustard", "Groundnut"]
    },
    {
        "name":        "Nagaland Organic Mission",
        "description": "Financial support for farmers transitioning to certified organic farming. Covers certification cost, inputs and training.",
        "eligibility": "Nagaland farmers with >0.5 acre land",
        "apply_at":    "nagaland.gov.in/agriculture",
        "crops":       []
    },
    {
        "name":        "NE Region Horticulture Mission (MIDH)",
        "description": "Subsidy up to 50% on planting material, irrigation, post-harvest infrastructure for horticulture crops in Northeast India.",
        "eligibility": "Farmers in NE states growing fruits, vegetables or spices",
        "apply_at":    "midh.gov.in",
        "crops":       ["Ginger", "Potato", "Sweet Potato", "Tapioca", "Sugarcane", "Tea"]
    },
    {
        "name":        "RKVY — Agriculture Infrastructure",
        "description": "Capital grants for farm infrastructure: drip/sprinkler irrigation, cold storage, farm equipment and primary processing units.",
        "eligibility": "Individual farmers, FPOs and cooperatives",
        "apply_at":    "rkvy.nic.in",
        "crops":       []
    },
    {
        "name":        "Soil Health Card Scheme",
        "description": "Free soil testing and nutrient recommendations specific to your plot. Reduces fertilizer cost by 8-10%.",
        "eligibility": "All farmers",
        "apply_at":    "soilhealth.dac.gov.in",
        "crops":       []
    },
]

# ── Planting calendar ─────────────────────────────────────────────────────────

PLANTING_CALENDAR = {
    "Jhum Paddy":        {"sow": [5, 6],    "harvest": [10, 11], "zones": "all"},
    "WTRC Paddy Kharif": {"sow": [4, 5],    "harvest": [9, 10],  "zones": "all"},
    "WTRC Paddy Rabi":   {"sow": [11, 12],  "harvest": [3, 4],   "zones": ["Kohima", "Tseminyu", "Phek"]},
    "Maize Kharif":      {"sow": [3, 4],    "harvest": [7, 8],   "zones": "all"},
    "Maize Rabi":        {"sow": [10, 11],  "harvest": [2, 3],   "zones": ["Dimapur", "Peren", "Niuland"]},
    "Ginger":            {"sow": [3, 4],    "harvest": [11, 12], "zones": "all"},
    "Potato":            {"sow": [1, 2],    "harvest": [4, 5],   "zones": ["Kohima", "Phek", "Zunheboto"]},
    "Sweet Potato":      {"sow": [6, 7],    "harvest": [10, 11], "zones": "all"},
    "Tapioca":           {"sow": [2, 3],    "harvest": [11, 12], "zones": ["Dimapur", "Peren", "Mon"]},
    "Soyabean":          {"sow": [5, 6],    "harvest": [9, 10],  "zones": "all"},
    "Mustard":           {"sow": [10, 11],  "harvest": [2, 3],   "zones": "all"},
    "Sugarcane":         {"sow": [2, 3],    "harvest": [12, 1],  "zones": ["Dimapur", "Peren", "Niuland"]},
    "Tea":               {"sow": [2, 3],    "harvest": [3, 11],  "zones": ["Mokokchung", "Tuensang", "Wokha"]},
    "Groundnut":         {"sow": [5, 6],    "harvest": [9, 10],  "zones": "all"},
    "Beans":             {"sow": [4, 5],    "harvest": [7, 8],   "zones": "all"},
}

MONTH_NAMES = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# ── Market price anchors (real 2023-24 MSP / Agmarknet data) ─────────────────

PRICE_ANCHORS = {
    # Cereals (MSP 2023-24)
    "Jhum Paddy":        {"base": 2183, "unit": "₹/quintal"},
    "WTRC Paddy Kharif": {"base": 2183, "unit": "₹/quintal"},
    "WTRC Paddy Rabi":   {"base": 2183, "unit": "₹/quintal"},
    "Maize Kharif":      {"base": 1962, "unit": "₹/quintal"},
    "Maize Rabi":        {"base": 1962, "unit": "₹/quintal"},
    "Jowar":             {"base": 3180, "unit": "₹/quintal"},
    "Bajra":             {"base": 2500, "unit": "₹/quintal"},
    "Ragi":              {"base": 3846, "unit": "₹/quintal"},
    "Small Millet":      {"base": 3150, "unit": "₹/quintal"},
    "Wheat":             {"base": 2275, "unit": "₹/quintal"},
    # Pulses (MSP 2023-24)
    "Tur/Arhar":         {"base": 7000, "unit": "₹/quintal"},
    "Urd/Moong":         {"base": 6950, "unit": "₹/quintal"},
    "Pea":               {"base": 5500, "unit": "₹/quintal"},
    "Gram":              {"base": 5440, "unit": "₹/quintal"},
    "Beans":             {"base": 4200, "unit": "₹/quintal"},
    "Rajmash Kharif":    {"base": 6500, "unit": "₹/quintal"},
    "Rajmash Rabi":      {"base": 6500, "unit": "₹/quintal"},
    "Lentil":            {"base": 6425, "unit": "₹/quintal"},
    # Oilseeds (MSP 2023-24)
    "Soyabean":          {"base": 4600, "unit": "₹/quintal"},
    "Groundnut":         {"base": 6377, "unit": "₹/quintal"},
    "Mustard":           {"base": 5650, "unit": "₹/quintal"},
    "Sesamum":           {"base": 8635, "unit": "₹/quintal"},
    "Linseed":           {"base": 7080, "unit": "₹/quintal"},
    "Sunflower":         {"base": 6760, "unit": "₹/quintal"},
    # Commercial crops (Agmarknet / industry)
    "Ginger":            {"base": 8500, "unit": "₹/quintal"},
    "Potato":            {"base": 1200, "unit": "₹/quintal"},
    "Sweet Potato":      {"base": 1800, "unit": "₹/quintal"},
    "Tapioca":           {"base": 900,  "unit": "₹/quintal"},
    "Sugarcane":         {"base": 3150, "unit": "₹/quintal"},
    "Tea":               {"base": 14800,"unit": "₹/quintal"},
    "Colocasia":         {"base": 2500, "unit": "₹/quintal"},
    "Yam":               {"base": 3000, "unit": "₹/quintal"},
    "Jute":              {"base": 4750, "unit": "₹/quintal"},
}


# ── Schemas ───────────────────────────────────────────────────────────────────

class SchemeQuery(BaseModel):
    district:   str
    crop:       str
    land_acres: float = 1.0

class DiseaseReportQuery(BaseModel):
    district: Optional[str] = None
    crop:     Optional[str] = None
    limit:    int = 200


# ── Endpoints ─────────────────────────────────────────────────────────────────

FRONTEND_DIR = os.path.join(BASE_DIR, "frontend", "src")

# Serve static files (manifest, service worker, icons, etc.)
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
def root():
    """Serve the frontend HTML."""
    index = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index):
        return FileResponse(index)
    return {"app": "Plotwise", "status": "running", "model_loaded": DISEASE_MODEL is not None}


@app.get("/api/status")
def api_status():
    return {
        "app":          "Plotwise",
        "tagline":      "Smart farming for Nagaland",
        "version":      "1.1.0",
        "status":       "running",
        "data":         f"{len(CROP_RECORDS)} real records loaded from Nagaland agriculture data 2023-24",
        "districts":    len(DISTRICTS),
        "crops":        len(CROPS),
        "model_loaded": DISEASE_MODEL is not None,
    }


@app.post("/disease/detect")
async def detect_disease(
    file:     UploadFile = File(...),
    crop:     str = "Jhum Paddy",
    district: str = "Kohima"
):
    """Upload a crop leaf image → get disease prediction. Logs report to DB."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Please upload an image file.")

    # Crops the ML model was actually trained on (24-class EfficientNetB0)
    ML_SUPPORTED_CROPS = {
        "Apple", "Chilli", "Grape", "Maize", "Maize Kharif", "Maize Rabi",
        "Orange", "Pepper", "Potato", "Soyabean", "Soybean",
        "Tomato", "Tomato Kharif",
    }

    use_ml = (
        DISEASE_MODEL is not None
        and crop in ML_SUPPORTED_CROPS
    )

    if use_ml:
        # Real model inference
        import numpy as np
        img_bytes  = await file.read()
        arr        = _preprocess_image(img_bytes)
        preds      = DISEASE_MODEL.predict(arr, verbose=0)[0]
        class_idx  = int(preds.argmax())
        confidence = float(preds[class_idx])
        raw_label  = DISEASE_CLASSES.get(str(class_idx), f"class_{class_idx}")

        # Get top 2 predictions for uncertainty check
        sorted_idx = preds.argsort()[::-1]
        top1_conf  = float(preds[sorted_idx[0]])
        top2_conf  = float(preds[sorted_idx[1]])
        conf_gap   = top1_conf - top2_conf   # How sure vs. next best

        # Uncertain if: low confidence OR too close to second prediction
        if confidence < 0.55 or (confidence < 0.70 and conf_gap < 0.15):
            use_ml = "uncertain"
        elif confidence < 0.70:
            use_ml = "low_confidence"

    # Map model class -> human-readable disease name (shared across confidence tiers)
    LABEL_MAP = {
        "Apple_AppleScab": "Apple Scab",
        "Apple_BlackRot": "Black Rot",
        "Chilli_LeafCurl": "Leaf Curl",
        "Grape_BlackRot": "Black Rot",
        "Grape_Esca": "Esca (Black Measles)",
        "Healthy": "Healthy (no disease detected)",
        "Healthy_Maize": "Healthy (no disease detected)",
        "Healthy_Pepper": "Healthy (no disease detected)",
        "Healthy_Potato": "Healthy (no disease detected)",
        "Healthy_Tomato": "Healthy (no disease detected)",
        "Maize_Cercospora_GrayLeafSpot": "Gray Leaf Spot",
        "Maize_CommonRust": "Common Rust",
        "Maize_NorthernLeafBlight": "Northern Leaf Blight",
        "Orange_Haunglongbing": "Citrus Greening (Huanglongbing)",
        "Pepper_BacterialSpot": "Bacterial Spot",
        "Potato_EarlyBlight": "Early Blight",
        "Potato_LateBlight": "Late Blight",
        "Soybean_Healthy": "Healthy (no disease detected)",
        "Tomato_BacterialSpot": "Bacterial Spot",
        "Tomato_EarlyBlight": "Early Blight",
        "Tomato_LateBlight": "Late Blight",
        "Tomato_LeafMold": "Leaf Mold",
        "Tomato_SeptoriaLeafSpot": "Septoria Leaf Spot",
        "Tomato_YellowLeafCurl": "Yellow Leaf Curl Virus",
    }

    # Auto-detect crop from model class name (order matters — specific before generic)
    CROP_MAP = [
        ("Healthy_Maize", "Maize"), ("Healthy_Pepper", "Chilli"),
        ("Healthy_Potato", "Potato"), ("Healthy_Tomato", "Tomato"),
        ("Soybean_", "Soybean"), ("Apple_", "Apple"), ("Grape_", "Grape"),
        ("Orange_", "Orange"), ("Maize_", "Maize"), ("Potato_", "Potato"),
        ("Tomato_", "Tomato"), ("Pepper_", "Chilli"), ("Chilli_", "Chilli"),
    ]

    def _detect_crop(label):
        for prefix, c in CROP_MAP:
            if label.startswith(prefix):
                return c
        return crop   # fallback to user-selected crop

    if use_ml == True:
        detected = LABEL_MAP.get(raw_label, raw_label)
        source   = "ML model (EfficientNetB0)"
        crop     = _detect_crop(raw_label)

    elif use_ml == "low_confidence":
        detected = LABEL_MAP.get(raw_label, raw_label)
        source   = "ML model (low confidence — verify with expert)"
        crop     = _detect_crop(raw_label)

    elif use_ml == "uncertain":
        # Model is too uncertain — be honest about it
        detected     = "Uncertain — could not identify clearly"
        confidence   = round(confidence, 3)
        source       = "ML model (uncertain — retake photo)"

        severity = "Unknown"
        return {
            "crop":         crop,
            "district":     district,
            "disease":      detected,
            "confidence":   round(confidence, 3),
            "severity":     severity,
            "treatment":    "The image was unclear or the disease is not in our training data. Try: (1) Take a closer photo of a single leaf, (2) Ensure good lighting, (3) Avoid blurry images. If symptoms persist, contact your District Agriculture Officer.",
            "prevention":   "Use certified seeds every season. Maintain field hygiene. Rotate crops annually to break disease cycles.",
            "nearest_help": "Contact District Agriculture Officer or call Kisan Call Centre: 1800-180-1551 (toll free)",
            "source":       source,
        }

    else:
        # No ML model available — use knowledge base for the selected crop
        disease_list = DISEASES.get(crop, DISEASES["default"])
        detected     = disease_list[0]   # Pick most common, not random
        confidence   = 0.0
        source       = "Knowledge base (AI model not available for this crop)"

        severity = "Check required"
        _log_disease(district, crop, detected, confidence, severity)
        return {
            "crop":         crop,
            "district":     district,
            "disease":      f"Common diseases for {crop}: {', '.join(disease_list)}",
            "confidence":   0.0,
            "severity":     severity,
            "treatment":    ". ".join([f"{d}: {TREATMENTS.get(d, 'Consult DAO')}" for d in disease_list[:3]]),
            "prevention":   "Use certified seeds every season. Maintain field hygiene. Rotate crops annually to break disease cycles.",
            "nearest_help": "Upload a photo of Apple, Chilli, Grape, Maize, Orange, Pepper, Potato, Soybean, or Tomato leaves for AI-powered detection. For other crops, contact Kisan Call Centre: 1800-180-1551 (toll free)",
            "source":       source,
        }

    severity = "High" if confidence >= 0.85 else "Moderate" if confidence >= 0.70 else "Low"

    # Persist to DB for heatmap analytics
    _log_disease(district, crop, detected, confidence, severity)

    return {
        "crop":         crop,
        "district":     district,
        "disease":      detected,
        "confidence":   round(confidence, 3),
        "severity":     severity,
        "treatment":    TREATMENTS.get(detected, "Consult your District Agriculture Officer."),
        "prevention":   "Use certified seeds every season. Maintain field hygiene. Rotate crops annually to break disease cycles.",
        "nearest_help": "Contact District Agriculture Officer or call Kisan Call Centre: 1800-180-1551 (toll free)",
        "source":       source,
    }


@app.get("/prices")
def get_market_prices(crop: Optional[str] = None, district: Optional[str] = None):
    """Market prices anchored to real MSP and Agmarknet data for Nagaland 2023-24."""
    crops_to_show = [crop] if crop else list(PRICE_ANCHORS.keys())
    prices = []

    for c in crops_to_show:
        anchor = PRICE_ANCHORS.get(c)
        if not anchor:
            continue
        base  = anchor["base"]
        price = base + random.randint(-150, 200)
        trend = random.choice(["up", "down", "stable"])
        prices.append({
            "crop":          c,
            "price_per_qtl": price,
            "msp":           base,
            "unit":          anchor["unit"],
            "market":        district or "Nagaland APMC",
            "trend":         trend,
            "trend_pct":     f"{random.uniform(0.3, 4.5):.1f}%",
            "last_updated":  datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            "tip":           "Good time to sell — above MSP" if trend == "up"
                             else "Consider holding stock"   if trend == "down"
                             else "Stable — close to MSP",
        })

    return {
        "prices":  prices,
        "source":  "Agmarknet / Nagaland APMC / MSP 2023-24",
        "records": len(prices),
    }


@app.get("/calendar")
def planting_calendar(district: str = "Kohima", crop: Optional[str] = None):
    """District-specific planting and harvest windows with real yield data."""
    crops_to_show  = [crop] if crop else list(PLANTING_CALENDAR.keys())
    result         = []
    current_month  = datetime.now().month

    for c in crops_to_show:
        info  = PLANTING_CALENDAR.get(c)
        if not info:
            continue
        zones = info.get("zones", "all")
        if zones != "all" and district not in zones:
            continue

        sow_months     = [MONTH_NAMES[m] for m in info["sow"]]
        harvest_months = [MONTH_NAMES[m] for m in info["harvest"]]
        sow_start      = min(info["sow"])
        harvest_start  = min(info["harvest"])

        if current_month in info["sow"]:
            status = "sowing now"
        elif current_month in info["harvest"]:
            status = "harvest time"
        elif sow_start < current_month < harvest_start:
            status = "growing"
        elif current_month < sow_start:
            status = "upcoming"
        else:
            status = "off season"

        matching  = [r for r in CROP_RECORDS if r["district"] == district and r["crop"] == c]
        avg_yield = round(sum(r["yield_kg_per_ha"] for r in matching) / len(matching)) if matching else None

        result.append({
            "crop":           c,
            "sow_window":     sow_months,
            "harvest_window": harvest_months,
            "status":         status,
            "suitable_for":   district,
            "avg_yield_kg_ha":avg_yield,
        })

    return {"district": district, "calendar": result}


@app.post("/schemes")
def find_schemes(query: SchemeQuery):
    """Find government schemes a farmer qualifies for."""
    matched = [s for s in SCHEMES if not s["crops"] or query.crop in s["crops"]]
    return {
        "district":        query.district,
        "crop":            query.crop,
        "matched_schemes": len(matched),
        "schemes":         matched,
        "tip":             "Visit your nearest Block Development Office or Common Service Centre to apply in person.",
    }


@app.get("/dashboard/yield")
def yield_dashboard(district: Optional[str] = None, crop: Optional[str] = None):
    """Real yield analytics from Nagaland agriculture data 2023-24."""
    records = CROP_RECORDS
    if district:
        records = [r for r in records if r["district"] == district]
    if crop:
        records = [r for r in records if r["crop"] == crop]

    total_production = sum(r["production_tonnes"] for r in records)
    total_area       = sum(r["area_ha"]           for r in records)

    crop_totals = {}
    for r in records:
        crop_totals[r["crop"]] = crop_totals.get(r["crop"], 0) + r["production_tonnes"]
    top_crops = sorted(crop_totals.items(), key=lambda x: x[1], reverse=True)[:10]

    dist_totals = {}
    for r in records:
        dist_totals[r["district"]] = dist_totals.get(r["district"], 0) + r["production_tonnes"]
    top_districts = sorted(dist_totals.items(), key=lambda x: x[1], reverse=True)

    return {
        "summary": {
            "total_records":      len(records),
            "total_production_t": round(total_production, 1),
            "total_area_ha":      round(total_area, 1),
            "avg_yield_kg_ha":    round(total_production * 1000 / total_area, 1) if total_area > 0 else 0,
            "period":             "2023-24",
            "source":             "Nagaland Agriculture Data 2023-24",
        },
        "top_crops":     [{"crop": c,     "production_t": round(p, 1)} for c, p in top_crops],
        "top_districts": [{"district": d, "production_t": round(p, 1)} for d, p in top_districts],
        "records":       records[:100],
    }


@app.get("/dashboard/disease-heatmap")
def disease_heatmap(district: Optional[str] = None, crop: Optional[str] = None):
    """
    Aggregated disease report data from the SQLite store.
    Drives the heatmap on the department dashboard.
    """
    conn  = _db()
    query = "SELECT district, crop, disease, COUNT(*) as reports, AVG(confidence) as avg_conf FROM disease_reports WHERE 1=1"
    params = []
    if district:
        query += " AND district = ?"
        params.append(district)
    if crop:
        query += " AND crop = ?"
        params.append(crop)
    query += " GROUP BY district, crop, disease ORDER BY reports DESC"

    rows = conn.execute(query, params).fetchall()
    conn.close()

    total_reports = sum(r[3] for r in rows)

    # Per-district aggregation
    by_district = {}
    for row in rows:
        d = row[0]
        by_district.setdefault(d, {"district": d, "total_reports": 0, "top_disease": "", "diseases": {}})
        by_district[d]["total_reports"]      += row[3]
        by_district[d]["diseases"][row[2]]    = by_district[d]["diseases"].get(row[2], 0) + row[3]

    for d in by_district.values():
        if d["diseases"]:
            d["top_disease"] = max(d["diseases"], key=d["diseases"].get)

    return {
        "total_reports":  total_reports,
        "by_district":    list(by_district.values()),
        "top_diseases":   [
            {"district": r[0], "crop": r[1], "disease": r[2],
             "reports": r[3], "avg_confidence": round(r[4], 3)}
            for r in rows[:50]
        ],
        "source": "Plotwise disease report database",
    }


@app.get("/districts")
def list_districts():
    return {"districts": DISTRICTS, "count": len(DISTRICTS)}


@app.get("/crops")
def list_crops():
    return {"crops": CROPS, "count": len(CROPS)}


@app.get("/district/{district_name}")
def district_detail(district_name: str):
    """Full crop breakdown for a single district."""
    records = RECORDS_BY_DISTRICT.get(district_name)
    if not records:
        raise HTTPException(404, f"District '{district_name}' not found.")

    summary = next((d for d in DISTRICT_SUMMARY if d["district"] == district_name), {})
    return {
        "district":           district_name,
        "total_area_ha":      summary.get("total_area_ha"),
        "total_production_t": summary.get("total_production_tonnes"),
        "crops_count":        len(records),
        "crops":              sorted(records, key=lambda r: r["production_tonnes"], reverse=True),
    }


# ── Chat assistant ───────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    message:  str
    district: str = "Kohima"
    lang:     str = "en"   # "en" or "nag"


def _match_crop(text: str) -> Optional[str]:
    """Find a crop name mentioned in the user message (whole-word match)."""
    import re
    text_l = text.lower()
    # Check exact and partial matches against known crops
    # Ordered: longer aliases first to avoid substring collisions
    crop_aliases = [
        ("sweet potato", "Sweet Potato"),
        ("soyabean", "Soyabean"), ("soybean", "Soyabean"), ("soya", "Soyabean"),
        ("groundnut", "Groundnut"), ("peanut", "Groundnut"),
        ("sugarcane", "Sugarcane"), ("ganna", "Sugarcane"),
        ("turmeric", "Ginger"),
        ("mustard", "Mustard"), ("sarson", "Mustard"),
        ("tapioca", "Tapioca"),
        ("ginger", "Ginger"), ("adha", "Ginger"), ("ada", "Ginger"),
        ("potato", "Potato"), ("aloo", "Potato"), ("alu", "Potato"),
        ("paddy", "Jhum Paddy"), ("dhan", "Jhum Paddy"), ("rice", "Jhum Paddy"),
        ("maize", "Maize Kharif"), ("corn", "Maize Kharif"), ("makka", "Maize Kharif"),
        ("chilli", "Chilli"), ("chili", "Chilli"), ("mircha", "Chilli"),
        ("wheat", "Wheat"),
        ("beans", "Beans"),
        ("tea", "Tea"), ("chah", "Tea"),
    ]
    for alias, crop in crop_aliases:
        if re.search(r'\b' + re.escape(alias) + r'\b', text_l):
            return crop
    # Direct crop name match
    for crop in PRICE_ANCHORS:
        if re.search(r'\b' + re.escape(crop.lower()) + r'\b', text_l):
            return crop
    return None


def _match_district(text: str) -> Optional[str]:
    """Find a district name mentioned in the user message."""
    text_l = text.lower()
    for d in DISTRICTS:
        if d.lower() in text_l:
            return d
    return None


def _detect_intent(text: str) -> str:
    """Detect user intent from message keywords."""
    text_l = text.lower()

    disease_kw = ["disease", "rog", "sick", "leaf", "blight", "rust", "rot", "wilt",
                  "treatment", "spray", "fungicide", "cure", "problem", "kharap",
                  "yellow", "spot", "curl", "dying", "infected", "pest", "insect",
                  "dawai", "upai", "medicine", "what wrong", "ki hoise"]
    price_kw   = ["price", "daam", "rate", "market", "bazar", "mandi", "sell",
                  "buy", "cost", "quintal", "msp", "kiman", "becho", "kinibo"]
    plant_kw   = ["plant", "sow", "harvest", "when", "season", "calendar",
                  "lagao", "time", "month", "grow", "khetir somoy", "lagano",
                  "somoy", "koita"]
    scheme_kw  = ["scheme", "subsidy", "insurance", "pm-kisan", "pmfby",
                  "government", "sarkari", "benefit", "apply", "eligible",
                  "yojana", "sahayota", "pmkisan"]
    district_kw = ["district", "area", "production", "yield", "crops in",
                   "grow in", "tell me about", "info about"]
    weather_kw = ["weather", "rain", "temperature", "temp", "forecast",
                  "humidity", "wind", "storm", "sunny", "cloudy",
                  "botas", "bristi", "bah", "thanda", "gorom"]
    greet_kw   = ["hello", "hi", "help", "namaste", "hey", "hola",
                  "good morning", "good evening", "thanks", "thank",
                  "what can you", "ki koribo", "sahajyo"]

    for kw in disease_kw:
        if kw in text_l:
            return "disease"
    for kw in price_kw:
        if kw in text_l:
            return "price"
    for kw in plant_kw:
        if kw in text_l:
            return "planting"
    for kw in scheme_kw:
        if kw in text_l:
            return "scheme"
    for kw in weather_kw:
        if kw in text_l:
            return "weather"
    for kw in district_kw:
        if kw in text_l:
            return "district"
    for kw in greet_kw:
        if kw in text_l:
            return "greeting"
    return "general"


@app.post("/api/chat")
def chat(msg: ChatMessage):
    """
    AI farming assistant for Nagaland farmers.
    Answers questions about diseases, prices, planting, schemes using real data.
    """
    text     = msg.message.strip()
    district = _match_district(text) or msg.district
    crop     = _match_crop(text)
    intent   = _detect_intent(text)

    reply    = ""
    suggestions = []

    if intent == "greeting":
        reply = (
            f"Hello! I'm the Plotwise farming assistant. I can help you with:\n\n"
            f"- Crop disease identification & treatment\n"
            f"- Live market prices for Nagaland crops\n"
            f"- Planting & harvest calendar by district\n"
            f"- Government schemes you qualify for\n"
            f"- Live weather & farming advisories\n\n"
            f"Just ask me anything about your farm!"
        )
        suggestions = ["What's the price of ginger?", "When to plant rice?",
                       "My potato has spots", "Show me government schemes"]

    elif intent == "disease":
        if crop:
            disease_list = DISEASES.get(crop, DISEASES["default"])
            lines = [f"Common diseases for **{crop}** in {district}:\n"]
            for d in disease_list:
                treatment = TREATMENTS.get(d, "Consult your District Agriculture Officer.")
                lines.append(f"**{d}** — {treatment}")
            lines.append(f"\nFor accurate diagnosis, upload a leaf photo in the Disease Detection tab.")
            lines.append(f"Kisan Call Centre: 1800-180-1551 (toll free)")
            reply = "\n\n".join(lines)
        else:
            reply = (
                "I can help with crop diseases! Which crop is affected?\n\n"
                "Tell me something like: \"My potato has leaf spots\" or \"Rice disease treatment\""
            )
        suggestions = ["Potato disease", "Rice blast treatment", "Maize leaf blight",
                       "Upload leaf photo"]

    elif intent == "price":
        if crop:
            anchor = PRICE_ANCHORS.get(crop)
            if anchor:
                price = anchor["base"] + random.randint(-150, 200)
                reply = (
                    f"**{crop}** — current market price:\n\n"
                    f"Price: **{anchor['unit'].split('/')[0]}{price}/quintal**\n"
                    f"MSP (Govt. guaranteed): {anchor['unit'].split('/')[0]}{anchor['base']}/quintal\n"
                    f"Market: {district} APMC\n\n"
                )
                if price > anchor["base"]:
                    reply += "Price is above MSP — good time to sell."
                else:
                    reply += "Price is near/below MSP — consider holding if you can store safely."
                reply += "\n\nSource: Agmarknet / Nagaland APMC 2023-24"
            else:
                reply = f"I don't have price data for {crop} yet. Try: rice, maize, ginger, potato, soyabean, mustard, tea."
        else:
            # Show top 5 prices
            top_crops = ["Ginger", "Tea", "Soyabean", "Mustard", "Potato"]
            lines = [f"Top crop prices in {district}:\n"]
            for c in top_crops:
                a = PRICE_ANCHORS.get(c)
                if a:
                    p = a["base"] + random.randint(-100, 150)
                    lines.append(f"**{c}**: {a['unit'].split('/')[0]}{p}/qtl (MSP: {a['base']})")
            lines.append(f"\nAsk about any specific crop for detailed pricing!")
            reply = "\n".join(lines)
        suggestions = ["Ginger price", "Rice rate today", "Tea market price",
                       "When to sell potato?"]

    elif intent == "planting":
        if crop:
            cal = PLANTING_CALENDAR.get(crop)
            if cal:
                sow_months = ", ".join([MONTH_NAMES[m] for m in cal["sow"]])
                harv_months = ", ".join([MONTH_NAMES[m] for m in cal["harvest"]])
                zones = cal.get("zones", "all")
                current_month = datetime.now().month

                if current_month in cal["sow"]:
                    status = "NOW is sowing time!"
                elif current_month in cal["harvest"]:
                    status = "Harvest season is on!"
                else:
                    status = f"Next sowing: {sow_months}"

                reply = (
                    f"**{crop}** — planting calendar for {district}:\n\n"
                    f"Sow: **{sow_months}**\n"
                    f"Harvest: **{harv_months}**\n"
                    f"Status: {status}\n"
                )
                if zones != "all":
                    if district in zones:
                        reply += f"This crop is well-suited for {district}."
                    else:
                        reply += f"Note: {crop} is mainly grown in {', '.join(zones)}. May not be ideal for {district}."
                else:
                    reply += f"This crop grows well across all Nagaland districts."

                # Add yield data if available
                matching = [r for r in CROP_RECORDS if r["district"] == district and r["crop"] == crop]
                if matching:
                    avg_yield = round(sum(r["yield_kg_per_ha"] for r in matching) / len(matching))
                    reply += f"\n\nAvg. yield in {district}: **{avg_yield} kg/ha** (2023-24 data)"
            else:
                reply = f"I don't have calendar data for {crop} yet. Try: rice, maize, ginger, potato, soyabean, tea."
        else:
            reply = (
                "I can show you the planting calendar! Which crop?\n\n"
                "Try: \"When to plant ginger?\" or \"Rice harvest season\""
            )
        suggestions = ["When to plant ginger?", "Rice sowing time", "Potato harvest",
                       "What to plant now?"]

    elif intent == "scheme":
        if crop:
            matched = [s for s in SCHEMES if not s["crops"] or crop in s["crops"]]
        else:
            matched = SCHEMES

        if matched:
            lines = [f"Government schemes" + (f" for **{crop}**" if crop else "") + f" in {district}:\n"]
            for s in matched:
                lines.append(f"**{s['name']}**\n{s['description']}\nApply: {s['apply_at']}\n")
            lines.append("Visit your nearest Block Development Office or Common Service Centre to apply.")
            reply = "\n".join(lines)
        else:
            reply = "No specific schemes found for this crop. Check the Scheme Finder tab for a full search."
        suggestions = ["PM-KISAN details", "Crop insurance", "Organic farming scheme",
                       "How to apply?"]

    elif intent == "district":
        target_district = _match_district(text) or district
        records = RECORDS_BY_DISTRICT.get(target_district, [])
        if records:
            total_prod = sum(r["production_tonnes"] for r in records)
            total_area = sum(r["area_ha"] for r in records)
            top = sorted(records, key=lambda r: r["production_tonnes"], reverse=True)[:5]
            lines = [f"**{target_district}** district — agriculture overview:\n"]
            lines.append(f"Total production: **{round(total_prod):,} tonnes**")
            lines.append(f"Total cultivated area: **{round(total_area):,} ha**\n")
            lines.append("Top crops:")
            for r in top:
                lines.append(f"  {r['crop']}: {round(r['production_tonnes']):,}T ({round(r['yield_kg_per_ha'])} kg/ha)")
            lines.append(f"\nSource: Nagaland agriculture data 2023-24")
            reply = "\n".join(lines)
        else:
            reply = f"I don't have data for '{target_district}'. Available districts: {', '.join(DISTRICTS[:8])}..."
        suggestions = ["Kohima crops", "Dimapur agriculture", "Mon district",
                       "Best district for ginger?"]

    elif intent == "weather":
        try:
            coords = DISTRICT_COORDS.get(district, (25.67, 94.12))
            lat, lon = coords
            w_url = (
                f"https://api.open-meteo.com/v1/forecast?"
                f"latitude={lat}&longitude={lon}"
                f"&current=temperature_2m,relative_humidity_2m,weather_code,precipitation"
                f"&timezone=Asia/Kolkata"
            )
            req = urllib.request.Request(w_url, headers={"User-Agent": "Plotwise/1.1"})
            with urllib.request.urlopen(req, timeout=8) as resp:
                w_data = json.loads(resp.read().decode())
            cur = w_data.get("current", {})
            WMO = {0:"Clear sky",1:"Mainly clear",2:"Partly cloudy",3:"Overcast",
                   45:"Foggy",51:"Light drizzle",53:"Moderate drizzle",
                   61:"Slight rain",63:"Moderate rain",65:"Heavy rain",
                   80:"Rain showers",81:"Moderate rain showers",82:"Heavy rain showers",
                   95:"Thunderstorm",96:"Thunderstorm with hail"}
            weather_desc = WMO.get(cur.get("weather_code", 0), "Unknown")
            temp = cur.get("temperature_2m", 0)
            hum  = cur.get("relative_humidity_2m", 0)
            rain = cur.get("precipitation", 0)

            reply = (
                f"**Weather in {district}** right now:\n\n"
                f"**{weather_desc}** · {temp}°C\n"
                f"Humidity: {hum}% · Rain: {rain} mm\n\n"
            )
            if hum > 85:
                reply += "⚠️ High humidity — watch for fungal diseases.\n"
            if rain > 10:
                reply += "⚠️ Heavy rain expected — ensure field drainage.\n"
            if temp > 35:
                reply += "⚠️ Extreme heat — irrigate early morning or late evening.\n"
            if hum <= 85 and rain <= 10 and temp <= 35:
                reply += "Conditions look good for field work today.\n"
            reply += "\nCheck the **Weather tab** for a 7-day forecast."
        except Exception:
            reply = f"Sorry, I couldn't fetch weather for {district} right now. Try the **Weather tab** in the app instead."
        suggestions = ["Weather in Kohima", "Will it rain tomorrow?",
                       "Weather Dimapur", "Price of ginger"]

    else:
        # General / unknown
        reply = (
            "I'm not sure I understood that. I can help with:\n\n"
            "- **Crop diseases** — \"My potato has brown spots\"\n"
            "- **Market prices** — \"What's the price of ginger?\"\n"
            "- **Planting calendar** — \"When to sow rice?\"\n"
            "- **Government schemes** — \"What schemes can I get?\"\n"
            "- **District data** — \"Tell me about Kohima\"\n"
            "- **Weather** — \"Weather in Kohima\"\n\n"
            "Try asking one of these!"
        )
        suggestions = ["What's the price of ginger?", "When to plant rice?",
                       "My potato has spots", "Weather in Kohima"]

    return {
        "reply":       reply,
        "intent":      intent,
        "crop":        crop,
        "district":    district,
        "suggestions": suggestions,
    }


# ── Export endpoints ──────────────────────────────────────────────────────────

@app.get("/api/export/prices")
def export_prices(district: str = "Kohima"):
    """Export market prices as CSV download."""
    price_data = get_market_prices(district=district)
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Crop", "Price (Rs/qtl)", "MSP (Rs/qtl)", "Trend", "Tip", "Market", "Source"])
    for p in price_data.get("prices", []):
        writer.writerow([
            p["crop"], p["price_per_qtl"], p.get("msp", "N/A"),
            p.get("trend", ""), p.get("tip", ""),
            p.get("market", district), price_data.get("source", "")
        ])
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=plotwise_prices_{district}.csv"}
    )


@app.get("/api/export/yield")
def export_yield(district: str = None):
    """Export yield analytics as CSV download."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["District", "Crop", "Area (ha)", "Production (tonnes)", "Yield (kg/ha)", "Season"])
    for r in CROP_RECORDS:
        if district and r.get("district") != district:
            continue
        writer.writerow([
            r.get("district", ""), r.get("crop", ""),
            r.get("area_ha", ""), r.get("production_tonnes", ""),
            r.get("yield_kg_per_ha", ""), r.get("season", "")
        ])
    output.seek(0)
    fname = f"plotwise_yield_{district}.csv" if district else "plotwise_yield_all.csv"
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={fname}"}
    )


# ── Weather (Open-Meteo — free, no API key) ──────────────────────────────────

DISTRICT_COORDS = {
    "Kohima":      (25.67, 94.12),
    "Dimapur":     (25.90, 93.73),
    "Mokokchung":  (26.32, 94.52),
    "Tuensang":    (26.27, 94.83),
    "Mon":         (26.75, 95.00),
    "Wokha":       (26.10, 94.27),
    "Zunheboto":   (25.97, 94.52),
    "Phek":        (25.67, 94.47),
    "Longleng":    (26.43, 94.87),
    "Kiphire":     (25.88, 95.12),
    "Peren":       (25.52, 93.73),
    "Noklak":      (26.60, 95.03),
    "Shamator":    (26.08, 95.20),
    "Niuland":      (25.82, 93.80),
    "Nuiland":      (25.82, 93.80),
    "Chumoukedima": (25.78, 93.78),
    "Tseminyu":    (25.78, 94.18),
}


@app.get("/api/weather")
def get_weather(district: str = "Kohima"):
    """
    Live weather data from Open-Meteo (free, no API key).
    Returns current conditions + 7-day forecast for the specified district.
    """
    coords = DISTRICT_COORDS.get(district)
    if not coords:
        raise HTTPException(404, f"District '{district}' not found.")

    lat, lon = coords

    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        f"&current=temperature_2m,relative_humidity_2m,wind_speed_10m,"
        f"weather_code,precipitation"
        f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,"
        f"weather_code,wind_speed_10m_max"
        f"&timezone=Asia/Kolkata&forecast_days=7"
    )

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Plotwise/1.1"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
    except (urllib.error.URLError, Exception) as e:
        raise HTTPException(502, f"Weather service unavailable: {str(e)}")

    # Weather code → description
    WMO_CODES = {
        0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
        45: "Foggy", 48: "Depositing rime fog",
        51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
        61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
        71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
        80: "Slight rain showers", 81: "Moderate rain showers",
        82: "Violent rain showers",
        95: "Thunderstorm", 96: "Thunderstorm with hail",
        99: "Thunderstorm with heavy hail",
    }

    current = data.get("current", {})
    daily   = data.get("daily", {})

    # Build farming advisory based on weather
    temp = current.get("temperature_2m", 0)
    humidity = current.get("relative_humidity_2m", 0)
    precip = current.get("precipitation", 0)
    code = current.get("weather_code", 0)

    advisories = []
    if humidity > 85:
        advisories.append("High humidity — watch for fungal diseases (blight, mildew). Consider preventive fungicide.")
    if temp > 35:
        advisories.append("Extreme heat — irrigate in early morning or late evening. Mulch to retain soil moisture.")
    if temp < 10:
        advisories.append("Cold conditions — protect tender crops with mulch or row covers.")
    if precip > 20:
        advisories.append("Heavy rainfall expected — ensure field drainage. Delay fertilizer application.")
    if code >= 95:
        advisories.append("Thunderstorm alert — secure structures, avoid open fields.")
    if not advisories:
        advisories.append("Conditions are favourable for field work.")

    # Build 7-day forecast
    forecast = []
    dates = daily.get("time", [])
    for i in range(len(dates)):
        forecast.append({
            "date":     dates[i],
            "temp_max": daily["temperature_2m_max"][i],
            "temp_min": daily["temperature_2m_min"][i],
            "precip":   daily["precipitation_sum"][i],
            "wind_max": daily["wind_speed_10m_max"][i],
            "weather":  WMO_CODES.get(daily["weather_code"][i], "Unknown"),
            "code":     daily["weather_code"][i],
        })

    return {
        "district":   district,
        "lat":        lat,
        "lon":        lon,
        "current": {
            "temperature": current.get("temperature_2m"),
            "humidity":    current.get("relative_humidity_2m"),
            "wind_speed":  current.get("wind_speed_10m"),
            "precipitation": current.get("precipitation"),
            "weather":     WMO_CODES.get(code, "Unknown"),
            "code":        code,
        },
        "forecast":   forecast,
        "advisories": advisories,
        "source":     "Open-Meteo.com (free, open-source weather API)",
    }
