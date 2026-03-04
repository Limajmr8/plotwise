"""
Plotwise — Smart Farming Platform for Nagaland
FastAPI Backend

Author: Limawapang Jamir
For: Department of Agriculture, Nagaland

Features:
  - Crop disease detection (ML)
  - Market price tracker
  - Planting calendar by district
  - Government scheme finder
  - Yield analytics dashboard (for department use)

Usage:
    uvicorn backend.src.main:app --reload --port 8000
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import numpy as np
from datetime import datetime
import random

app = FastAPI(
    title="Plotwise API",
    description="Smart farming platform for Nagaland farmers",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ── Data ───────────────────────────────────────────────────────────────────────

NAGALAND_DISTRICTS = [
    "Kohima", "Dimapur", "Mokokchung", "Wokha", "Zunheboto",
    "Tuensang", "Mon", "Phek", "Peren", "Kiphire", "Longleng", "Noklak"
]

CROPS = [
    "Rice", "Maize", "Millet", "Soybean", "Ginger",
    "Turmeric", "Chilli", "Potato", "Cabbage", "Mustard"
]

DISEASES = {
    "Rice":    ["Blast", "Bacterial Blight", "Brown Spot", "Sheath Blight"],
    "Maize":   ["Gray Leaf Spot", "Northern Leaf Blight", "Common Rust"],
    "Ginger":  ["Soft Rot", "Bacterial Wilt", "Yellow Mottle Virus"],
    "default": ["Powdery Mildew", "Leaf Spot", "Root Rot", "Nutrient Deficiency"]
}

SCHEMES = [
    {
        "name":        "PM-KISAN",
        "description": "Direct income support of ₹6,000/year to all farmer families",
        "eligibility": "All landholding farmer families",
        "apply_at":    "pmkisan.gov.in",
        "crops":       []  # All crops
    },
    {
        "name":        "PMFBY (Crop Insurance)",
        "description": "Comprehensive crop insurance covering yield losses",
        "eligibility": "All farmers growing notified crops",
        "apply_at":    "pmfby.gov.in",
        "crops":       ["Rice", "Maize", "Soybean"]
    },
    {
        "name":        "Nagaland Organic Mission",
        "description": "Financial support for transitioning to organic farming",
        "eligibility": "Nagaland farmers with >0.5 acre land",
        "apply_at":    "nagaland.gov.in/agriculture",
        "crops":       []
    },
    {
        "name":        "NE Region Horticulture Mission",
        "description": "Subsidy for horticulture development in Northeast India",
        "eligibility": "Farmers in NE states growing fruits/vegetables/spices",
        "apply_at":    "midh.gov.in",
        "crops":       ["Ginger", "Turmeric", "Chilli", "Potato", "Cabbage"]
    },
    {
        "name":        "RKVY (Agriculture Infrastructure)",
        "description": "Grant for farm infrastructure: irrigation, storage, equipment",
        "eligibility": "Individual farmers and FPOs",
        "apply_at":    "rkvy.nic.in",
        "crops":       []
    }
]

PLANTING_CALENDAR = {
    "Rice":     {"sow": [4, 5],    "harvest": [9, 10], "zones": ["Kohima", "Mokokchung", "Wokha"]},
    "Maize":    {"sow": [3, 4],    "harvest": [7, 8],  "zones": "all"},
    "Ginger":   {"sow": [3, 4],    "harvest": [11, 12],"zones": ["Dimapur", "Peren", "Phek"]},
    "Turmeric": {"sow": [4, 5],    "harvest": [12, 1], "zones": "all"},
    "Potato":   {"sow": [1, 2],    "harvest": [4, 5],  "zones": ["Kohima", "Phek", "Zunheboto"]},
    "Chilli":   {"sow": [2, 3],    "harvest": [8, 9],  "zones": "all"},
    "Soybean":  {"sow": [5, 6],    "harvest": [9, 10], "zones": "all"},
    "Millet":   {"sow": [4, 5],    "harvest": [8, 9],  "zones": "all"},
    "Cabbage":  {"sow": [8, 9],    "harvest": [11, 12],"zones": ["Kohima", "Phek"]},
    "Mustard":  {"sow": [10, 11],  "harvest": [2, 3],  "zones": "all"},
}


# ── Schemas ────────────────────────────────────────────────────────────────────

class SchemeQuery(BaseModel):
    district: str
    crop:     str
    land_acres: float = 1.0

class YieldEntry(BaseModel):
    district: str
    crop:     str
    year:     int
    yield_kg_per_acre: float


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "app":     "Plotwise",
        "tagline": "Smart farming for Nagaland 🌾",
        "version": "1.0.0",
        "status":  "running"
    }


@app.post("/disease/detect")
async def detect_disease(
    file: UploadFile = File(...),
    crop: str = "Rice"
):
    """
    Upload a crop leaf image → get disease prediction.
    In production: runs through trained CNN/EfficientNet model.
    For demo: returns realistic mock prediction.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Please upload an image file.")

    # Load model in production:
    # model = tf.keras.models.load_model("ml/crop_disease_model.h5")
    # img = preprocess_image(await file.read())
    # pred = model.predict(img)

    disease_list = DISEASES.get(crop, DISEASES["default"])
    detected     = random.choice(disease_list)
    confidence   = round(random.uniform(0.72, 0.97), 3)

    treatments = {
        "Blast":               "Apply tricyclazole fungicide. Remove infected leaves. Avoid excess nitrogen.",
        "Bacterial Blight":    "Use copper-based bactericide. Drain fields. Plant resistant varieties.",
        "Soft Rot":            "Improve drainage. Reduce humidity. Apply Bordeaux mixture.",
        "Bacterial Wilt":      "Remove infected plants. Disinfect tools. Use disease-free seed rhizomes.",
        "Gray Leaf Spot":      "Apply mancozeb fungicide. Rotate crops. Improve air circulation.",
        "Powdery Mildew":      "Apply sulfur-based fungicide. Avoid overhead irrigation.",
        "Leaf Spot":           "Remove infected leaves. Apply copper fungicide. Improve spacing.",
        "Root Rot":            "Improve soil drainage. Reduce watering. Apply biological fungicide.",
        "Nutrient Deficiency": "Conduct soil test. Apply balanced NPK fertilizer."
    }

    return {
        "crop":         crop,
        "disease":      detected,
        "confidence":   confidence,
        "severity":     "Moderate" if confidence < 0.85 else "High",
        "treatment":    treatments.get(detected, "Consult local agriculture officer."),
        "prevention":   "Use certified seeds, maintain field hygiene, rotate crops annually.",
        "nearest_help": "Contact District Agriculture Officer or call Kisan Call Centre: 1800-180-1551"
    }


@app.get("/prices")
def get_market_prices(crop: Optional[str] = None, district: Optional[str] = None):
    """Live mandi prices — in production, fetches from Agmarknet API."""
    base_prices = {
        "Rice": 2200, "Maize": 1850, "Ginger": 8500, "Turmeric": 7200,
        "Chilli": 6500, "Potato": 1200, "Soybean": 3800, "Millet": 2100,
        "Cabbage": 900, "Mustard": 4800
    }

    crops_to_show = [crop] if crop else CROPS
    prices = []

    for c in crops_to_show:
        base  = base_prices.get(c, 2000)
        price = base + random.randint(-200, 200)
        trend = random.choice(["up", "down", "stable"])
        prices.append({
            "crop":           c,
            "price_per_qtl":  price,
            "unit":           "₹/quintal",
            "market":         district or random.choice(NAGALAND_DISTRICTS),
            "trend":          trend,
            "trend_pct":      f"{random.uniform(0.5, 5.0):.1f}%",
            "last_updated":   datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            "tip":            "Good time to sell" if trend == "up" else
                              "Consider holding stock" if trend == "down" else "Stable market"
        })

    return {"prices": prices, "source": "Agmarknet / Nagaland APMC"}


@app.get("/calendar")
def planting_calendar(district: str = "Kohima", crop: Optional[str] = None):
    """Returns planting and harvest window for given district and crop."""
    month_names = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    result = []

    crops_to_show = [crop] if crop else list(PLANTING_CALENDAR.keys())

    for c in crops_to_show:
        info   = PLANTING_CALENDAR.get(c, {})
        zones  = info.get("zones", "all")
        if zones != "all" and district not in zones:
            continue

        sow_months     = [month_names[m] for m in info.get("sow", [])]
        harvest_months = [month_names[m] for m in info.get("harvest", [])]
        current_month  = datetime.now().month

        status = "upcoming" if current_month < min(info.get("sow", [13])) else \
                 "sowing now" if current_month in info.get("sow", []) else \
                 "growing" if min(info.get("sow", [0])) < current_month < min(info.get("harvest", [13])) else \
                 "harvest time" if current_month in info.get("harvest", []) else "off season"

        result.append({
            "crop":           c,
            "sow_window":     sow_months,
            "harvest_window": harvest_months,
            "status":         status,
            "suitable_for":   district
        })

    return {"district": district, "calendar": result}


@app.post("/schemes")
def find_schemes(query: SchemeQuery):
    """Find government schemes a farmer qualifies for."""
    matched = []
    for scheme in SCHEMES:
        if not scheme["crops"] or query.crop in scheme["crops"]:
            matched.append(scheme)

    return {
        "district": query.district,
        "crop":     query.crop,
        "matched_schemes": len(matched),
        "schemes":  matched,
        "tip":      "Visit your nearest Block Development Office or Common Service Centre to apply."
    }


@app.get("/dashboard/yield")
def yield_dashboard(district: Optional[str] = None):
    """
    Yield analytics for the Agriculture Department dashboard.
    In production: queries from PostgreSQL database with actual records.
    """
    districts = [district] if district else NAGALAND_DISTRICTS[:6]
    data = []

    for d in districts:
        for crop in random.sample(CROPS, 4):
            for year in [2021, 2022, 2023]:
                data.append({
                    "district":          d,
                    "crop":              crop,
                    "year":              year,
                    "yield_kg_per_acre": round(random.uniform(400, 2200), 1),
                    "area_acres":        round(random.uniform(50, 800), 1),
                    "rainfall_mm":       round(random.uniform(800, 2500), 1)
                })

    total_yield = sum(r["yield_kg_per_acre"] * r["area_acres"] for r in data)
    return {
        "summary": {
            "districts_covered": len(districts),
            "crops_tracked":     len(CROPS),
            "total_yield_kg":    round(total_yield),
            "period":            "2021–2023"
        },
        "records": data[:50]
    }


@app.get("/districts")
def list_districts():
    return {"districts": NAGALAND_DISTRICTS}


@app.get("/crops")
def list_crops():
    return {"crops": CROPS}
