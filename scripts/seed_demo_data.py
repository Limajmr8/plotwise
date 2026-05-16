"""
Plotwise — Seed Demo Data
Populates the disease_reports table with realistic demo entries
so the heatmap and PDF reports look populated during the B2G demo.

Usage:
    python scripts/seed_demo_data.py
    python scripts/seed_demo_data.py --db data/plotwise.db --count 80

Author: Limawapang L Jamir
"""

import sqlite3
import random
import argparse
from datetime import datetime, timedelta
import os

# ── Configuration ──────────────────────────────────────────────────

DISTRICTS = [
    "Kohima", "Tseminyu", "Phek", "Mokokchung", "Tuensang", "Noklak",
    "Shamator", "Mon", "Dimapur", "Niuland", "Chumoukedima", "Wokha",
    "Zunheboto", "Peren", "Kiphire", "Longleng"
]

# Realistic crop + disease pairings from our ML model & knowledge base
CROP_DISEASES = [
    ("Potato", "Late Blight", "High"),
    ("Potato", "Early Blight", "Moderate"),
    ("Maize Kharif", "Gray Leaf Spot", "Moderate"),
    ("Maize Kharif", "Northern Leaf Blight", "High"),
    ("Maize Kharif", "Common Rust", "Low"),
    ("Tomato", "Bacterial Spot", "Moderate"),
    ("Tomato", "Late Blight", "High"),
    ("Tomato", "Leaf Mold", "Low"),
    ("Tomato", "Septoria Leaf Spot", "Moderate"),
    ("Tomato", "Yellow Leaf Curl Virus", "High"),
    ("Chilli", "Leaf Curl", "Moderate"),
    ("Apple", "Apple Scab", "High"),
    ("Apple", "Black Rot", "Moderate"),
    ("Grape", "Esca (Black Measles)", "High"),
    ("Soyabean", "Pod Blight", "Moderate"),
    ("Orange", "Citrus Greening (Huanglongbing)", "High"),
    ("Pepper", "Bacterial Spot", "Low"),
    ("Jhum Paddy", "Blast", "High"),
    ("Jhum Paddy", "Bacterial Blight", "Moderate"),
    ("Ginger", "Soft Rot", "High"),
    ("Ginger", "Bacterial Wilt", "Moderate"),
]

# Demo reporter names (realistic Nagaland names + roles)
REPORTERS = [
    ("Imtisunep Longchar", "Extension Officer"),
    ("Akumla Jamir", "Extension Officer"),
    ("Temjen Ao", "Farmer"),
    ("Sentimeren Imchen", "Farmer"),
    ("Vizokholie Suohu", "Block Officer"),
    ("Kekhriesenuo Yhome", "Farmer"),
    ("Limatoshi", "Farmer"),
    ("Moasangba Sangtam", "Farmer"),
    ("Dr. Tali Kikon", "Researcher"),
    ("Zhaputhi Zhimomi", "Extension Officer"),
    ("Chubanungla Chang", "Farmer"),
    ("Neikethozo Nagi", "District Officer"),
    ("Anungla Longkumer", "Farmer"),
    ("Imlikumzuk", "Farmer"),
    ("Throngpong Konyak", "Farmer"),
]


def seed_database(db_path: str, count: int = 80):
    """Insert realistic demo disease reports into the database."""
    os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
    conn = sqlite3.connect(db_path)

    # Ensure table exists with reporter columns
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

    # Migrate: add reporter columns if missing
    cursor = conn.execute("PRAGMA table_info(disease_reports)")
    columns = [row[1] for row in cursor.fetchall()]
    if "reporter" not in columns:
        conn.execute("ALTER TABLE disease_reports ADD COLUMN reporter TEXT DEFAULT ''")
        conn.execute("ALTER TABLE disease_reports ADD COLUMN reporter_role TEXT DEFAULT ''")
        conn.commit()
        print("  Migrated: added reporter columns to existing table.")

    # Check existing
    existing = conn.execute("SELECT COUNT(*) FROM disease_reports").fetchone()[0]
    if existing > 0:
        print(f"Database already has {existing} reports. Adding {count} more demo entries.")

    # Generate reports spread over last 60 days
    now = datetime.utcnow()
    records = []

    for _ in range(count):
        crop, disease, base_severity = random.choice(CROP_DISEASES)
        district = random.choice(DISTRICTS)
        reporter_name, reporter_role = random.choice(REPORTERS)

        # Confidence varies by severity
        if base_severity == "High":
            confidence = round(random.uniform(0.78, 0.97), 3)
        elif base_severity == "Moderate":
            confidence = round(random.uniform(0.62, 0.82), 3)
        else:
            confidence = round(random.uniform(0.55, 0.72), 3)

        # Timestamp: random point in last 60 days
        days_ago = random.randint(0, 60)
        hours_offset = random.randint(6, 18)  # During working hours
        ts = (now - timedelta(days=days_ago, hours=-hours_offset)).isoformat()

        # 30% chance of anonymous report (no reporter)
        if random.random() < 0.3:
            reporter_name = ""
            reporter_role = ""

        records.append((district, crop, disease, confidence, base_severity, ts, reporter_name, reporter_role))

    conn.executemany(
        "INSERT INTO disease_reports (district, crop, disease, confidence, severity, timestamp, reporter, reporter_role) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        records
    )
    conn.commit()

    final_count = conn.execute("SELECT COUNT(*) FROM disease_reports").fetchone()[0]
    conn.close()

    print(f"[OK] Seeded {count} demo disease reports.")
    print(f"  Total reports in DB: {final_count}")
    print(f"  Database: {os.path.abspath(db_path)}")
    print(f"  Districts covered: {len(set(r[0] for r in records))}")
    print(f"  Diseases: {len(set(r[2] for r in records))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed Plotwise demo database with realistic disease reports")
    parser.add_argument("--db", default="data/plotwise.db", help="Path to SQLite database")
    parser.add_argument("--count", type=int, default=80, help="Number of reports to generate")
    args = parser.parse_args()

    seed_database(args.db, args.count)
