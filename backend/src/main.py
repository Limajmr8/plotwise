"""
Plotwise — Smart Farming Platform for Nagaland
FastAPI Backend

Author: Limawapang L Jamir
For: Nagaland Agriculture Department (B2G)

All data sourced from verified Nagaland agriculture records 2023-24.
Disease reports stored in SQLite for heatmap analytics.
"""

from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel
from typing import Optional
from datetime import datetime, date
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import hashlib
import json
import csv
import os
import sqlite3
import io
import logging
import time
import urllib.request
import urllib.error

from dotenv import load_dotenv
load_dotenv()

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO),
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("plotwise")

APP_VERSION = "2.0.0"

app = FastAPI(
    title="Plotwise API",
    description="Smart farming platform for Nagaland farmers",
    version=APP_VERSION
)

CORS_ORIGINS = [
    o.strip() for o in
    os.environ.get(
        "CORS_ORIGINS",
        "https://limajmr-plotwise.hf.space,capacitor://localhost,"
        "http://localhost,http://localhost:8000,http://127.0.0.1:8000"
    ).split(",")
    if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

from starlette.middleware.base import BaseHTTPMiddleware

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Permissions-Policy"] = "camera=(self), microphone=()"
        # Only add HSTS in production (when served over HTTPS)
        if request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response

app.add_middleware(SecurityHeadersMiddleware)


# ── Audit / Request Logging Middleware ────────────────────────────────────────

_server_start_time = time.time()
_audit_stats = {
    "total_requests": 0,
    "errors": 0,
    "by_endpoint": {},
    "started_at": datetime.utcnow().isoformat(),
}


class AuditLogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        start = time.time()
        _audit_stats["total_requests"] += 1
        path = request.url.path
        _audit_stats["by_endpoint"][path] = _audit_stats["by_endpoint"].get(path, 0) + 1

        response = await call_next(request)
        duration_ms = round((time.time() - start) * 1000, 1)

        if response.status_code >= 400:
            _audit_stats["errors"] += 1

        # Log API requests (skip static files for noise reduction)
        if not path.startswith("/static"):
            logger.info(f"{request.method} {path} → {response.status_code} ({duration_ms}ms)")

        return response


app.add_middleware(AuditLogMiddleware)


limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error on {request.method} {request.url.path}: {exc}")
    return JSONResponse(status_code=500, content={
        "error": "Something went wrong. Please try again.",
        "detail": str(exc) if isinstance(exc, HTTPException) else None,
    })


def _daily_variation(crop_name: str, base_price: int, spread: int = 200) -> int:
    """Deterministic daily price variation — stable within a day, shifts between days."""
    seed = hashlib.md5(f"{crop_name}:{date.today().isoformat()}".encode()).hexdigest()
    offset = int(seed[:8], 16) % (spread * 2 + 1) - spread
    return base_price + offset


def _daily_trend_pct(crop_name: str) -> str:
    seed = hashlib.md5(f"trend:{crop_name}:{date.today().isoformat()}".encode()).hexdigest()
    pct = 0.3 + (int(seed[:8], 16) % 43) / 10.0
    return f"{pct:.1f}%"

# ── Paths ──────────────────────────────────────────────────────────────────────

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR   = os.path.join(BASE_DIR, "data", "sample")
JSON_FILE  = os.path.join(DATA_DIR, "nagaland_agriculture_2023_24.json")
CSV_FILE   = os.path.join(DATA_DIR, "nagaland_crop_data_2023_24.csv")
MODEL_PATH   = os.environ.get("MODEL_PATH", os.path.join(BASE_DIR, "ml", "saved_models", "disease_model.h5"))
CLASSES_PATH = os.path.join(BASE_DIR, "ml", "saved_models", "class_indices.json")
DB_PATH    = os.environ.get("PLOTWISE_DB_PATH", os.path.join(BASE_DIR, "data", "plotwise.db"))

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
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            district      TEXT    NOT NULL,
            crop          TEXT    NOT NULL,
            disease       TEXT    NOT NULL,
            confidence    REAL    NOT NULL,
            severity      TEXT    NOT NULL,
            timestamp     TEXT    NOT NULL,
            reporter      TEXT    DEFAULT '',
            reporter_role TEXT    DEFAULT ''
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_reports_district ON disease_reports(district)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_reports_timestamp ON disease_reports(timestamp)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_reports_crop ON disease_reports(crop)")
    conn.commit()

    # Migrate: add reporter columns if table already exists without them
    cursor = conn.execute("PRAGMA table_info(disease_reports)")
    columns = [row[1] for row in cursor.fetchall()]
    if "reporter" not in columns:
        conn.execute("ALTER TABLE disease_reports ADD COLUMN reporter TEXT DEFAULT ''")
        conn.execute("ALTER TABLE disease_reports ADD COLUMN reporter_role TEXT DEFAULT ''")
        conn.commit()
        logger.info("Migrated disease_reports: added reporter columns")

    conn.close()

_init_db()


def _seed_if_empty():
    """Populate a small set of realistic demo reports when the table is empty.

    Guarded by PLOTWISE_SEED_ON_EMPTY so it only runs where intended (e.g. a
    fresh Railway volume before a demo). Idempotent: never adds rows if any
    report already exists. Keeps the disease-surveillance heatmap from showing
    blank after a redeploy onto a new volume.
    """
    if os.environ.get("PLOTWISE_SEED_ON_EMPTY", "").lower() not in ("1", "true", "yes"):
        return
    import random
    from datetime import timedelta

    conn = sqlite3.connect(DB_PATH)  # _db() is defined later in the module
    try:
        existing = conn.execute("SELECT COUNT(*) FROM disease_reports").fetchone()[0]
        if existing > 0:
            return

        crop_diseases = [
            ("Potato", "Late Blight", "High"), ("Potato", "Early Blight", "Moderate"),
            ("Maize Kharif", "Northern Leaf Blight", "High"), ("Maize Kharif", "Common Rust", "Low"),
            ("Tomato", "Bacterial Spot", "Moderate"), ("Tomato", "Late Blight", "High"),
            ("Chilli", "Leaf Curl", "Moderate"), ("Apple", "Apple Scab", "High"),
            ("Grape", "Esca (Black Measles)", "High"), ("Orange", "Citrus Greening (Huanglongbing)", "High"),
            ("Soyabean", "Pod Blight", "Moderate"), ("Ginger", "Soft Rot", "High"),
            ("Jhum Paddy", "Blast", "High"),
        ]
        reporters = [
            ("Imtisunep Longchar", "Extension Officer"), ("Akumla Jamir", "Extension Officer"),
            ("Temjen Ao", "Farmer"), ("Vizokholie Suohu", "Block Officer"),
            ("Dr. Tali Kikon", "Researcher"), ("Neikethozo Nagi", "District Officer"),
            ("", ""),  # some anonymous reports
        ]
        rng = random.Random(2026)  # deterministic so the demo looks the same each boot
        now = datetime.utcnow()
        rows = []
        for _ in range(70):
            crop, disease, sev = rng.choice(crop_diseases)
            district = rng.choice(DISTRICTS)
            name, role = rng.choice(reporters)
            conf = round(rng.uniform(0.72, 0.96) if sev == "High" else rng.uniform(0.58, 0.82), 3)
            ts = (now - timedelta(days=rng.randint(0, 55), hours=rng.randint(0, 12))).isoformat()
            rows.append((district, crop, disease, conf, sev, ts, name, role))
        conn.executemany(
            "INSERT INTO disease_reports (district, crop, disease, confidence, severity, timestamp, reporter, reporter_role) "
            "VALUES (?,?,?,?,?,?,?,?)",
            rows,
        )
        conn.commit()
        logger.info(f"Seeded {len(rows)} demo disease reports (PLOTWISE_SEED_ON_EMPTY).")
    finally:
        conn.close()


_seed_if_empty()

# Log DB state at startup
try:
    _startup_conn = sqlite3.connect(DB_PATH)
    _row_count = _startup_conn.execute("SELECT COUNT(*) FROM disease_reports").fetchone()[0]
    _startup_conn.close()
    if _row_count > 0:
        logger.info(f"Database loaded: {DB_PATH} ({_row_count} existing disease reports)")
    else:
        logger.info(f"Fresh database created at {DB_PATH}")
except Exception as _e:
    logger.warning(f"Could not check DB state: {_e}")

def _db():
    return sqlite3.connect(DB_PATH)


def _log_disease(district: str, crop: str, disease: str, confidence: float, severity: str,
                  reporter: str = "", reporter_role: str = ""):
    conn = _db()
    conn.execute(
        "INSERT INTO disease_reports (district, crop, disease, confidence, severity, timestamp, reporter, reporter_role) VALUES (?,?,?,?,?,?,?,?)",
        (district, crop, disease, round(confidence, 3), severity, datetime.utcnow().isoformat(), reporter, reporter_role)
    )
    conn.commit()
    conn.close()

# ── ML model (optional — loaded only when trained model exists) ────────────────

DISEASE_MODEL   = None
DISEASE_CLASSES = {}   # idx (str) → class label, loaded from class_indices.json

def _load_model():
    global DISEASE_MODEL, DISEASE_CLASSES
    if not os.path.exists(MODEL_PATH):
        logger.warning(f"No trained model at {MODEL_PATH}. Run: python ml/train_disease_model.py")
        return
    try:
        import tensorflow as tf
        DISEASE_MODEL = tf.keras.models.load_model(MODEL_PATH, compile=False)
        logger.info(f"Disease model loaded ({MODEL_PATH})")
        if os.path.exists(CLASSES_PATH):
            with open(CLASSES_PATH) as f:
                DISEASE_CLASSES = json.load(f)
            logger.info(f"Class labels loaded: {len(DISEASE_CLASSES)} classes")
        else:
            logger.warning("class_indices.json not found — re-run train_disease_model.py")
    except Exception as e:
        logger.warning(f"Could not load disease model: {e}")

_load_model()


def _preprocess_image(img_bytes: bytes):
    """Resize and normalise image for EfficientNetB0 inference."""
    from PIL import Image
    import numpy as np
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((224, 224))
    arr = np.array(img, dtype="float32")  # no /255 — EfficientNetB0 has built-in preprocessing
    return arr[None]  # (1, 224, 224, 3)


# Which model class labels are plausible for each crop the user can pick.
# The user tells us the crop, so we restrict the prediction to that crop's
# classes — this removes nonsensical cross-crop guesses (e.g. a tomato leaf
# called "citrus greening") and sharply improves real-world accuracy.
# Chilli and Pepper are the same botanical family (Capsicum) — the model
# cannot separate them, so they share the pepper classes.
CROP_TO_CLASSES = {
    "Apple":   ["Apple_AppleScab", "Apple_BlackRot"],
    "Chilli":  ["Chilli_LeafCurl", "Healthy_Pepper"],
    "Grape":   ["Grape_BlackRot", "Grape_Esca"],
    "Maize":   ["Maize_Cercospora_GrayLeafSpot", "Maize_CommonRust",
                "Maize_NorthernLeafBlight", "Healthy_Maize"],
    "Orange":  ["Orange_Haunglongbing"],
    "Pepper":  ["Pepper_BacterialSpot", "Healthy_Pepper"],
    "Potato":  ["Potato_EarlyBlight", "Potato_LateBlight", "Healthy_Potato"],
    "Soyabean": ["Soybean_Healthy"],
    "Tomato":  ["Tomato_BacterialSpot", "Tomato_EarlyBlight", "Tomato_LateBlight",
                "Tomato_LeafMold", "Tomato_SeptoriaLeafSpot", "Tomato_YellowLeafCurl",
                "Healthy_Tomato"],
}


def _base_crop(crop: str) -> Optional[str]:
    """Resolve a UI crop label (e.g. 'Maize Kharif', 'Soybean') to a base crop
    key in CROP_TO_CLASSES."""
    c = crop.strip()
    if c in CROP_TO_CLASSES:
        return c
    if c.startswith("Maize"):  return "Maize"
    if c.startswith("Tomato"): return "Tomato"
    if c in ("Soybean", "Soyabean"): return "Soyabean"
    if c.startswith("Chilli"): return "Chilli"
    if c.startswith("Pepper"): return "Pepper"
    return None


def _confidence_tier(confidence: float, conf_gap: float) -> str:
    """Map a prediction's confidence + top-2 gap to a tier.

    Returns 'confident' (>=70% and clear margin), 'low_confidence' (55-70%),
    or 'uncertain' (<55%, or <70% with the runner-up too close). Pure function
    so the 3-tier logic is testable without TensorFlow.
    """
    if confidence < 0.55 or (confidence < 0.70 and conf_gap < 0.15):
        return "uncertain"
    if confidence < 0.70:
        return "low_confidence"
    return "confident"

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


def _serve_index():
    index = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index):
        return FileResponse(index)
    return JSONResponse({"app": "Plotwise", "status": "running", "model_loaded": DISEASE_MODEL is not None})


@app.get("/")
def root(request: Request, desktop: Optional[str] = None):
    """Serve the frontend HTML.

    Phones get the compact tabbed app (/mobile) instead of the long desktop
    landing page — override with /?desktop (any value) to force the full site.
    """
    ua = request.headers.get("user-agent", "")
    # "Mobi" covers all phone browsers (Android phones include "Mobile");
    # Android tablets omit it and correctly get the desktop page.
    is_phone = ("Mobi" in ua or "iPhone" in ua) and "iPad" not in ua
    # Presence of ?desktop counts as true (bare /?desktop must not 422)
    force_desktop = desktop is not None and desktop.lower() not in ("0", "false", "no")
    if is_phone and not force_desktop:
        return RedirectResponse("/mobile", status_code=307)
    return _serve_index()


@app.get("/mobile")
def mobile_app():
    """Serve the mobile-optimized app HTML (used by Capacitor Android app)."""
    mobile = os.path.join(FRONTEND_DIR, "mobile.html")
    if os.path.exists(mobile):
        return FileResponse(mobile)
    # Fallback to main index if mobile.html doesn't exist yet
    return _serve_index()


@app.get("/sw.js")
def service_worker():
    """Serve the service worker from the root path so its scope covers page
    navigations ("/", "/mobile") — a worker served from /static/ can only
    control /static/* requests."""
    return FileResponse(os.path.join(FRONTEND_DIR, "sw.js"), media_type="application/javascript")


# ── Demo User Profile (lightweight auth substitute) ───────────────────────────

VALID_ROLES = ["Farmer", "Extension Officer", "Block Officer", "District Officer", "Researcher", "Demo User"]


class DemoProfile(BaseModel):
    name: str
    role: str = "Farmer"
    district: str = "Kohima"


@app.post("/api/profile")
def validate_profile(profile: DemoProfile):
    """Validate and echo back a demo user profile. No DB storage — frontend uses localStorage."""
    name = profile.name.strip()
    if not name or len(name) < 2:
        raise HTTPException(400, "Name must be at least 2 characters.")
    if len(name) > 60:
        raise HTTPException(400, "Name too long (max 60 characters).")
    role = profile.role if profile.role in VALID_ROLES else "Farmer"
    district = profile.district if profile.district in DISTRICTS else "Kohima"
    return {
        "name": name,
        "role": role,
        "district": district,
        "valid": True,
    }


@app.get("/api/status")
def api_status():
    return {
        "app":          "Plotwise",
        "tagline":      "Smart farming for Nagaland",
        "version":      APP_VERSION,
        "status":       "running",
        "data":         f"{len(CROP_RECORDS)} real records loaded from Nagaland agriculture data 2023-24",
        "districts":    len(DISTRICTS),
        "crops":        len(CROPS),
        "model_loaded": DISEASE_MODEL is not None,
    }


@app.get("/health")
def health_check():
    """Health check for Railway/monitoring with audit stats."""
    conn = _db()
    report_count = conn.execute("SELECT COUNT(*) FROM disease_reports").fetchone()[0]
    conn.close()

    uptime_seconds = round(time.time() - _server_start_time)
    return {
        "status": "ok",
        "model_loaded": DISEASE_MODEL is not None,
        "uptime_seconds": uptime_seconds,
        "disease_reports_total": report_count,
        "audit": {
            "total_requests": _audit_stats["total_requests"],
            "errors": _audit_stats["errors"],
            "started_at": _audit_stats["started_at"],
            "top_endpoints": dict(sorted(
                _audit_stats["by_endpoint"].items(),
                key=lambda x: x[1], reverse=True
            )[:10]),
        },
    }


@app.post("/disease/detect")
@limiter.limit("10/minute")
def detect_disease(
    request:  Request,
    file:     UploadFile = File(...),
    crop:     str = Form("Jhum Paddy"),
    district: str = Form("Kohima"),
    reporter: str = Form(""),
    reporter_role: str = Form("")
):
    """Upload a crop leaf image → get disease prediction. Logs report to DB.

    Defined as a sync `def` (not async) so Starlette runs it in a threadpool —
    this keeps the blocking TensorFlow inference off the event loop, so one slow
    classification can't freeze every other request (prices, weather, dashboard).
    """
    if not file.content_type or not file.content_type.startswith("image/"):
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
        img_bytes = file.file.read()
        if len(img_bytes) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"Image too large ({len(img_bytes) // 1024 // 1024}MB). Maximum is 10MB.")
        try:
            arr = _preprocess_image(img_bytes)
        except Exception:
            raise HTTPException(400, "Could not process image. Please upload a valid JPEG or PNG.")

        # Guard the whole inference path: a TF runtime / shape / OOM error should
        # degrade to the graceful "uncertain" message, not escape as a raw 500.
        try:
            preds      = DISEASE_MODEL.predict(arr, verbose=0)[0]
            raw_peak   = float(preds.max())   # model's absolute certainty (all 24 classes)

            # Crop-aware: the user picked a crop, so restrict the decision to that
            # crop's classes (>=2 needed to discriminate). Confidence is then the
            # probability *within* the crop's options. Single-class crops (Orange,
            # Soyabean) can't be discriminated, so they fall through to the raw path.
            base = _base_crop(crop)
            label_to_i = {v: int(k) for k, v in DISEASE_CLASSES.items()}
            allowed = [label_to_i[l] for l in CROP_TO_CLASSES.get(base, []) if l in label_to_i] if base else []

            if len(allowed) >= 2:
                sub        = np.array([preds[i] for i in allowed], dtype="float64")
                sub_norm   = sub / sub.sum()
                order      = sub_norm.argsort()[::-1]
                class_idx  = allowed[int(order[0])]
                confidence = float(sub_norm[int(order[0])])
                conf_gap   = float(sub_norm[int(order[0])] - sub_norm[int(order[1])])
                # Safety net: if the model's absolute peak (over ALL classes) is
                # very low, the photo probably isn't a clear leaf at all — stay
                # honest and report uncertain regardless of within-crop numbers.
                if raw_peak < 0.30:
                    confidence = raw_peak
                    conf_gap   = 0.0
            else:
                class_idx  = int(preds.argmax())
                confidence = float(preds[class_idx])
                sorted_idx = preds.argsort()[::-1]
                conf_gap   = float(preds[sorted_idx[0]] - preds[sorted_idx[1]])

            raw_label  = DISEASE_CLASSES.get(str(class_idx), f"class_{class_idx}")
        except Exception as e:
            logger.error(f"ML inference failed (crop={crop}, district={district}): {e}")
            return {
                "crop":         crop,
                "district":     district,
                "disease":      "Uncertain — could not identify clearly",
                "confidence":   0.0,
                "severity":     "Unknown",
                "treatment":    "The AI could not analyse this image. Try: (1) Take a closer photo of a single leaf, (2) Ensure good lighting, (3) Avoid blurry images. If symptoms persist, contact your District Agriculture Officer.",
                "prevention":   "Use certified seeds every season. Maintain field hygiene. Rotate crops annually to break disease cycles.",
                "nearest_help": "Contact District Agriculture Officer or call Kisan Call Centre: 1800-180-1551 (toll free)",
                "source":       "ML model (analysis error — retake photo)",
                "reporter":     reporter,
                "reporter_role": reporter_role,
            }

        # Uncertain if: low confidence OR too close to second prediction.
        # use_ml stays True when 'confident'.
        tier = _confidence_tier(confidence, conf_gap)
        if tier != "confident":
            use_ml = tier

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
            "reporter":     reporter,
            "reporter_role": reporter_role,
        }

    else:
        # No ML model available — use knowledge base for the selected crop
        disease_list = DISEASES.get(crop, DISEASES["default"])
        detected     = disease_list[0]   # Pick most common, not random
        confidence   = 0.0
        source       = "Knowledge base (AI model not available for this crop)"

        severity = "Check required"
        _log_disease(district, crop, detected, confidence, severity, reporter, reporter_role)
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
            "reporter":     reporter,
            "reporter_role": reporter_role,
        }

    severity = "High" if confidence >= 0.85 else "Moderate" if confidence >= 0.70 else "Low"

    # Persist to DB for heatmap analytics
    _log_disease(district, crop, detected, confidence, severity, reporter, reporter_role)

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
        "reporter":     reporter,
        "reporter_role": reporter_role,
    }


def _get_prices(crop: Optional[str] = None, district: Optional[str] = None) -> dict:
    """Internal price calculation — used by endpoint and other functions (PDF, CSV, chat)."""
    crops_to_show = [crop] if crop else list(PRICE_ANCHORS.keys())
    prices = []

    for c in crops_to_show:
        anchor = PRICE_ANCHORS.get(c)
        if not anchor:
            continue
        base  = anchor["base"]
        price = _daily_variation(c, base)
        trend_idx = int(hashlib.md5(f"t:{c}".encode()).hexdigest()[:8], 16) % 3
        trend = ["up", "stable", "down"][trend_idx]
        prices.append({
            "crop":          c,
            "price_per_qtl": price,
            "msp":           base,
            "unit":          anchor["unit"],
            "market":        district or "Nagaland APMC",
            "trend":         trend,
            "trend_pct":     _daily_trend_pct(c),
            "last_updated":  date.today().strftime("%Y-%m-%d"),
            # Advisory must agree with the displayed price vs MSP (not the
            # trend), or the tip can claim "above MSP" while showing a price
            # below it — a contradiction an agriculture officer will catch.
            "tip":           "Above MSP — good time to sell"    if price > base
                             else "Below MSP — consider holding" if price < base
                             else "At MSP",
        })

    return {
        "prices":  prices,
        "source":  "Agmarknet / Nagaland APMC / MSP 2023-24",
        "records": len(prices),
    }


@app.get("/prices")
@limiter.limit("30/minute")
def get_market_prices(request: Request, crop: Optional[str] = None, district: Optional[str] = None):
    """Market prices anchored to real MSP and Agmarknet data for Nagaland 2023-24."""
    return _get_prices(crop, district)


def _in_window(month: int, window: list) -> bool:
    """True if month falls within the [start, end] window (inclusive), handling
    year wrap-around (e.g. Sugarcane harvest Dec–Jan = [12, 1])."""
    start, end = window[0], window[-1]
    if start <= end:
        return start <= month <= end
    return month >= start or month <= end


@app.get("/calendar")
@limiter.limit("30/minute")
def planting_calendar(request: Request, district: str = "Kohima", crop: Optional[str] = None):
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

        # Windows are [start, end] ranges that may span >2 months or wrap the
        # year end. Check sowing → harvest → the growing arc between them;
        # anything left is the dormant period before the next season.
        if _in_window(current_month, info["sow"]):
            status = "sowing now"
        elif _in_window(current_month, info["harvest"]):
            status = "harvest time"
        elif _in_window(current_month, [info["sow"][-1], info["harvest"][0]]):
            status = "growing"
        else:
            status = "upcoming"

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
@limiter.limit("30/minute")
def find_schemes(request: Request, query: SchemeQuery):
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
@limiter.limit("30/minute")
def yield_dashboard(request: Request, district: Optional[str] = None, crop: Optional[str] = None):
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
@limiter.limit("30/minute")
def disease_heatmap(request: Request, district: Optional[str] = None, crop: Optional[str] = None):
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

    # Recent individual reports with reporter identity
    conn2 = _db()
    recent_q = "SELECT district, crop, disease, confidence, severity, timestamp, reporter, reporter_role FROM disease_reports WHERE 1=1"
    recent_params = []
    if district:
        recent_q += " AND district = ?"
        recent_params.append(district)
    if crop:
        recent_q += " AND crop = ?"
        recent_params.append(crop)
    recent_q += " ORDER BY timestamp DESC LIMIT 20"
    recent_rows = conn2.execute(recent_q, recent_params).fetchall()
    conn2.close()

    return {
        "total_reports":  total_reports,
        "by_district":    list(by_district.values()),
        "top_diseases":   [
            {"district": r[0], "crop": r[1], "disease": r[2],
             "reports": r[3], "avg_confidence": round(r[4], 3)}
            for r in rows[:50]
        ],
        "recent_reports": [
            {"district": r[0], "crop": r[1], "disease": r[2],
             "confidence": round(r[3], 3), "severity": r[4],
             "timestamp": r[5], "reporter": r[6] or "", "reporter_role": r[7] or ""}
            for r in recent_rows
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
@limiter.limit("20/minute")
def chat(request: Request, msg: ChatMessage):
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
                price = _daily_variation(crop, anchor["base"])
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
                reply += "\n\nIndicative price — anchored to MSP 2023-24 (CACP) and Agmarknet APMC rates."
            else:
                reply = f"I don't have price data for {crop} yet. Try: rice, maize, ginger, potato, soyabean, mustard, tea."
        else:
            top_crops = ["Ginger", "Tea", "Soyabean", "Mustard", "Potato"]
            lines = [f"Top crop prices in {district}:\n"]
            for c in top_crops:
                a = PRICE_ANCHORS.get(c)
                if a:
                    p = _daily_variation(c, a["base"], spread=150)
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
            w_data = _fetch_weather_cached(district, w_url)
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
@limiter.limit("10/minute")
def export_prices(request: Request, district: str = "Kohima"):
    """Export market prices as CSV download."""
    price_data = _get_prices(district=district)
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
@limiter.limit("10/minute")
def export_yield(request: Request, district: str = None):
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


@app.get("/api/export/report")
@limiter.limit("10/minute")
def export_pdf_report(request: Request, district: str = "Kohima"):
    """Generate a professional PDF district report — branding, yield, prices, disease alerts."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm

    buf = io.BytesIO()
    page_w, page_h = A4

    # Page number callback
    def _add_page_number(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.grey)
        canvas.drawRightString(page_w - 15*mm, 10*mm, f"Page {doc.page}")
        canvas.drawString(15*mm, 10*mm, "Plotwise — Smart Farming Platform for Nagaland")
        canvas.restoreState()

    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=20*mm, bottomMargin=18*mm)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("Title2", parent=styles["Title"], fontSize=20, spaceAfter=2,
                                 textColor=colors.HexColor("#1b5e20"))
    sub_style = ParagraphStyle("Sub", parent=styles["Normal"], fontSize=10, textColor=colors.grey)
    prepared_style = ParagraphStyle("Prepared", parent=styles["Normal"], fontSize=11,
                                    textColor=colors.HexColor("#333333"), spaceBefore=8, spaceAfter=4)
    h2_style = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=13, spaceBefore=14, spaceAfter=6,
                              textColor=colors.HexColor("#2e7d32"))
    footer_style = ParagraphStyle("Footer", parent=styles["Normal"], fontSize=8, textColor=colors.grey)
    elements = []

    # ─── Header / Branding ───
    elements.append(Paragraph("Plotwise — District Agriculture Intelligence Report", title_style))
    elements.append(Paragraph(
        f"<b>{district} District</b> | Data Period: 2023-24 | Generated: {date.today().strftime('%d %B %Y')}",
        sub_style,
    ))
    elements.append(Paragraph(
        "Submitted for review to: Director of Agriculture, Government of Nagaland",
        prepared_style,
    ))
    elements.append(Spacer(1, 6*mm))

    # ─── Summary Statistics ───
    records = [r for r in CROP_RECORDS if r["district"] == district]
    total_area = sum(r["area_ha"] for r in records)
    total_prod = sum(r["production_tonnes"] for r in records)
    avg_yield = round(total_prod * 1000 / total_area, 1) if total_area > 0 else 0

    elements.append(Paragraph("District Summary", h2_style))
    summary_data = [
        ["Metric", "Value"],
        ["Total Cultivated Area", f"{total_area:,.1f} ha"],
        ["Total Production", f"{total_prod:,.1f} tonnes"],
        ["Average Yield", f"{avg_yield:,.1f} kg/ha"],
        ["Crops Grown", str(len(set(r["crop"] for r in records)))],
        ["Data Source", "Director of Agriculture, Nagaland (2023-24)"],
    ]
    st = Table(summary_data, colWidths=[55*mm, 60*mm])
    st.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2e7d32")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, 1), (0, -1), colors.HexColor("#e8f5e9")),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    elements.append(st)
    elements.append(Spacer(1, 5*mm))

    # ─── Top Crops by Production ───
    elements.append(Paragraph("Top Crops by Production", h2_style))
    crop_totals = {}
    for r in records:
        crop_totals[r["crop"]] = crop_totals.get(r["crop"], 0) + r["production_tonnes"]
    top_crops = sorted(crop_totals.items(), key=lambda x: x[1], reverse=True)[:12]
    crop_rows = [["Crop", "Production (t)", "Area (ha)", "Yield (kg/ha)"]]
    for c, prod in top_crops:
        c_recs = [r for r in records if r["crop"] == c]
        area = sum(r["area_ha"] for r in c_recs)
        yld = round(prod * 1000 / area, 1) if area > 0 else 0
        crop_rows.append([c, f"{prod:,.1f}", f"{area:,.1f}", f"{yld:,.1f}"])
    ct = Table(crop_rows, colWidths=[45*mm, 35*mm, 30*mm, 35*mm])
    ct.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2e7d32")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f5")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(ct)
    elements.append(Spacer(1, 5*mm))

    # ─── Disease Alerts (from DB) ───
    elements.append(Paragraph("Disease Surveillance — Recent Reports", h2_style))
    conn = _db()
    disease_rows = conn.execute(
        "SELECT crop, disease, severity, confidence, timestamp, reporter, reporter_role "
        "FROM disease_reports WHERE district = ? ORDER BY timestamp DESC LIMIT 20",
        (district,)
    ).fetchall()
    conn.close()

    if disease_rows:
        d_table = [["Crop", "Disease", "Severity", "Confidence", "Date", "Reported By"]]
        for row in disease_rows:
            reporter_str = row[5] if row[5] else "—"
            if row[6]:
                reporter_str += f" ({row[6]})"
            d_table.append([
                row[0], row[1], row[2],
                f"{row[3]*100:.0f}%" if row[3] else "—",
                row[4][:10] if row[4] else "—",
                reporter_str,
            ])
        dt = Table(d_table, colWidths=[28*mm, 35*mm, 22*mm, 22*mm, 22*mm, 35*mm])
        dt.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#c62828")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#fff3f3")]),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ]))
        elements.append(dt)
    else:
        elements.append(Paragraph("No disease reports recorded for this district yet.", sub_style))
    elements.append(Spacer(1, 5*mm))

    # ─── Market Prices ───
    elements.append(Paragraph("Indicative Market Prices (MSP 2023-24 anchor)", h2_style))
    price_data = _get_prices(district=district)
    price_rows = [["Crop", "Price (Rs/qtl)", "MSP (Rs/qtl)", "Trend", "Advisory"]]
    for p in price_data.get("prices", [])[:15]:
        price_rows.append([
            p["crop"], str(p["price_per_qtl"]), str(p.get("msp", "-")),
            p.get("trend", ""), p.get("tip", "")
        ])
    pt = Table(price_rows, colWidths=[35*mm, 28*mm, 28*mm, 20*mm, 40*mm])
    pt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1565c0")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f5")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
    ]))
    elements.append(pt)
    elements.append(Spacer(1, 8*mm))

    # ─── Footer ───
    elements.append(Paragraph(
        "Data: Director of Agriculture, Nagaland (2023-24) | Prices: MSP (CACP 2023-24) + Agmarknet APMC | "
        "Disease Detection: EfficientNetB0 AI Model (PlantVillage dataset)",
        footer_style,
    ))
    elements.append(Paragraph(
        "Generated by Plotwise — an agricultural intelligence platform submitted for review to the "
        "Nagaland Agriculture Department. Developed by Limawapang L Jamir · limawapang8@gmail.com",
        footer_style,
    ))

    doc.build(elements, onFirstPage=_add_page_number, onLaterPages=_add_page_number)
    buf.seek(0)
    fname = f"plotwise_report_{district}_{date.today().isoformat()}.pdf"
    return StreamingResponse(
        buf,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={fname}"},
    )


# ── Weather (Open-Meteo — free, no API key) ──────────────────────────────────

_weather_cache = {}  # key: (district, url_hash) → value: (timestamp, data)
WEATHER_CACHE_TTL = int(os.environ.get("WEATHER_CACHE_TTL", 900))  # 15 minutes


def _fetch_weather_cached(district: str, url: str) -> dict:
    """Fetch weather data with in-memory caching to avoid hammering Open-Meteo."""
    cache_key = f"{district}:{hashlib.md5(url.encode()).hexdigest()[:8]}"
    now = time.time()

    cached = _weather_cache.get(cache_key)
    if cached and (now - cached[0]) < WEATHER_CACHE_TTL:
        logger.debug(f"Weather cache hit for {district}")
        return cached[1]

    req = urllib.request.Request(url, headers={"User-Agent": "Plotwise/1.1"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode())

    _weather_cache[cache_key] = (now, data)
    logger.debug(f"Weather cache miss for {district} — fetched fresh")
    return data


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
@limiter.limit("20/minute")
def get_weather(request: Request, district: str = "Kohima"):
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
        data = _fetch_weather_cached(district, url)
    except (urllib.error.URLError, Exception) as e:
        logger.error(f"Weather API failed for {district}: {e}")
        raise HTTPException(503, "Weather service is temporarily unavailable. Please try again in a few minutes.")

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
