"""
Plotwise — Backend API Test Suite
Tests all endpoints for correctness, error handling, and edge cases.

Run: pytest tests/ -v
"""

import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add project root to path so imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.src.main import app, DISEASE_MODEL, _confidence_tier


client = TestClient(app)


# ── Health & Status ───────────────────────────────────────────────────────────

class TestHealth:
    def test_health_check(self):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert "model_loaded" in data
        assert "uptime_seconds" in data
        assert "disease_reports_total" in data
        assert "audit" in data

    def test_api_status(self):
        r = client.get("/api/status")
        assert r.status_code == 200
        data = r.json()
        assert data["app"] == "Plotwise"
        assert data["status"] == "running"
        assert data["districts"] == 16
        assert data["crops"] > 0
        assert "576 real records" in data["data"]


# ── Static Pages ──────────────────────────────────────────────────────────────

class TestPages:
    def test_root_serves_html(self):
        r = client.get("/")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]

    def test_mobile_serves_html(self):
        r = client.get("/mobile")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]

    def test_root_redirects_phones_to_mobile(self):
        """Phone user-agents get the compact tabbed UI instead of the long landing page."""
        r = client.get("/", headers={
            "User-Agent": "Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 Mobile Safari/537.36"
        }, follow_redirects=False)
        assert r.status_code == 307
        assert r.headers["location"] == "/mobile"

    def test_root_desktop_override_on_phone(self):
        """?desktop=1 forces the full landing page even on a phone."""
        r = client.get("/?desktop=1", headers={
            "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) Mobile/15E148"
        }, follow_redirects=False)
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]

    def test_root_no_redirect_for_desktop(self):
        r = client.get("/", headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126.0 Safari/537.36"
        }, follow_redirects=False)
        assert r.status_code == 200

    def test_root_no_redirect_for_ipad(self):
        """iPads have desktop-class screens — keep the full site."""
        r = client.get("/", headers={
            "User-Agent": "Mozilla/5.0 (iPad; CPU OS 17_0 like Mac OS X) AppleWebKit/605.1.15 Mobile/15E148"
        }, follow_redirects=False)
        assert r.status_code == 200

    def test_root_no_redirect_for_android_tablet(self):
        """Android tablets omit the Mobile UA token — keep the full site."""
        r = client.get("/", headers={
            "User-Agent": "Mozilla/5.0 (Linux; Android 13; SM-X710) AppleWebKit/537.36 Chrome/126.0 Safari/537.36"
        }, follow_redirects=False)
        assert r.status_code == 200

    def test_root_bare_desktop_flag_no_422(self):
        """Bare /?desktop (no value) must force desktop, never a 422 error page."""
        for q in ["/?desktop", "/?desktop=", "/?desktop=true"]:
            r = client.get(q, headers={
                "User-Agent": "Mozilla/5.0 (Linux; Android 14; Pixel 8) Mobile Safari/537.36"
            }, follow_redirects=False)
            assert r.status_code == 200, f"{q} -> {r.status_code}"

    def test_service_worker_served_from_root(self):
        """SW must be served from / so its scope covers page navigations."""
        r = client.get("/sw.js")
        assert r.status_code == 200
        assert "javascript" in r.headers["content-type"]
        assert "plotwise-v" in r.text


# ── Districts & Crops ─────────────────────────────────────────────────────────

class TestData:
    def test_list_districts(self):
        r = client.get("/districts")
        assert r.status_code == 200
        data = r.json()
        assert data["count"] == 16
        assert "Kohima" in data["districts"]
        assert "Mon" in data["districts"]

    def test_list_crops(self):
        r = client.get("/crops")
        assert r.status_code == 200
        data = r.json()
        assert data["count"] > 30  # We have 44 crops

    def test_district_detail_valid(self):
        r = client.get("/district/Kohima")
        assert r.status_code == 200
        data = r.json()
        assert data["district"] == "Kohima"
        assert data["crops_count"] > 0
        assert "total_area_ha" in data
        assert "crops" in data

    def test_district_detail_invalid(self):
        r = client.get("/district/NonexistentPlace")
        assert r.status_code == 404


# ── Market Prices ─────────────────────────────────────────────────────────────

class TestPrices:
    def test_prices_all(self):
        r = client.get("/prices")
        assert r.status_code == 200
        data = r.json()
        assert "prices" in data
        assert len(data["prices"]) > 20  # 33 crops with anchors
        assert data["source"] == "Agmarknet / Nagaland APMC / MSP 2023-24"

    def test_prices_single_crop(self):
        r = client.get("/prices?crop=Ginger")
        assert r.status_code == 200
        data = r.json()
        assert len(data["prices"]) == 1
        assert data["prices"][0]["crop"] == "Ginger"
        assert data["prices"][0]["price_per_qtl"] > 0
        assert data["prices"][0]["msp"] == 8500

    def test_prices_with_district(self):
        r = client.get("/prices?district=Dimapur")
        assert r.status_code == 200
        data = r.json()
        assert all(p["market"] == "Dimapur" for p in data["prices"])

    def test_prices_unknown_crop(self):
        r = client.get("/prices?crop=UnknownCrop")
        assert r.status_code == 200
        data = r.json()
        assert data["prices"] == []
        assert data["records"] == 0


# ── Planting Calendar ─────────────────────────────────────────────────────────

class TestCalendar:
    def test_calendar_default(self):
        r = client.get("/calendar")
        assert r.status_code == 200
        data = r.json()
        assert data["district"] == "Kohima"
        assert len(data["calendar"]) > 0

    def test_calendar_specific_district(self):
        r = client.get("/calendar?district=Dimapur")
        assert r.status_code == 200
        data = r.json()
        assert data["district"] == "Dimapur"
        for crop in data["calendar"]:
            assert "crop" in crop
            assert "sow_window" in crop
            assert "harvest_window" in crop
            assert "status" in crop
            assert crop["status"] in ["sowing now", "harvest time", "growing", "off season", "upcoming"]

    def test_calendar_single_crop(self):
        r = client.get("/calendar?crop=Potato&district=Kohima")
        assert r.status_code == 200
        data = r.json()
        # Potato is suitable for Kohima
        assert len(data["calendar"]) == 1
        assert data["calendar"][0]["crop"] == "Potato"


# ── Schemes ───────────────────────────────────────────────────────────────────

class TestSchemes:
    def test_schemes_general(self):
        r = client.post("/schemes", json={
            "district": "Kohima",
            "crop": "Potato",
            "land_acres": 1.0
        })
        assert r.status_code == 200
        data = r.json()
        assert data["district"] == "Kohima"
        assert data["crop"] == "Potato"
        assert data["matched_schemes"] > 0
        assert any(s["name"] == "PM-KISAN" for s in data["schemes"])

    def test_schemes_horticulture_crop(self):
        r = client.post("/schemes", json={
            "district": "Mon",
            "crop": "Ginger",
            "land_acres": 2.0
        })
        assert r.status_code == 200
        data = r.json()
        # Ginger should match NE Horticulture Mission
        scheme_names = [s["name"] for s in data["schemes"]]
        assert "NE Region Horticulture Mission (MIDH)" in scheme_names


# ── Dashboard ─────────────────────────────────────────────────────────────────

class TestDashboard:
    def test_yield_dashboard_all(self):
        r = client.get("/dashboard/yield")
        assert r.status_code == 200
        data = r.json()
        assert "summary" in data
        assert data["summary"]["total_records"] > 500
        assert data["summary"]["period"] == "2023-24"
        assert len(data["top_crops"]) > 0
        assert len(data["top_districts"]) > 0

    def test_yield_dashboard_filtered(self):
        r = client.get("/dashboard/yield?district=Kohima")
        assert r.status_code == 200
        data = r.json()
        assert data["summary"]["total_records"] > 0
        assert data["summary"]["total_records"] < 576  # Less than total

    def test_disease_heatmap(self):
        r = client.get("/dashboard/disease-heatmap")
        assert r.status_code == 200
        data = r.json()
        assert "total_reports" in data
        assert "by_district" in data
        assert "top_diseases" in data
        assert "recent_reports" in data

    def test_disease_heatmap_filtered(self):
        r = client.get("/dashboard/disease-heatmap?district=Kohima")
        assert r.status_code == 200
        data = r.json()
        # Should only contain Kohima reports
        for d in data["by_district"]:
            assert d["district"] == "Kohima"


# ── Chat ──────────────────────────────────────────────────────────────────────

class TestChat:
    def test_chat_greeting(self):
        r = client.post("/api/chat", json={"message": "hello", "district": "Kohima"})
        assert r.status_code == 200
        data = r.json()
        assert data["intent"] == "greeting"
        assert "reply" in data
        assert len(data["suggestions"]) > 0

    def test_chat_price_query(self):
        r = client.post("/api/chat", json={"message": "price of ginger", "district": "Kohima"})
        assert r.status_code == 200
        data = r.json()
        assert data["intent"] == "price"
        assert data["crop"] == "Ginger"
        assert "8500" in data["reply"] or "price" in data["reply"].lower()

    def test_chat_disease_query(self):
        r = client.post("/api/chat", json={"message": "my potato has leaf spots", "district": "Kohima"})
        assert r.status_code == 200
        data = r.json()
        assert data["intent"] == "disease"
        assert data["crop"] == "Potato"

    def test_chat_planting_query(self):
        r = client.post("/api/chat", json={"message": "when to plant rice?", "district": "Kohima"})
        assert r.status_code == 200
        data = r.json()
        assert data["intent"] == "planting"

    def test_chat_scheme_query(self):
        r = client.post("/api/chat", json={"message": "government schemes for farmers", "district": "Kohima"})
        assert r.status_code == 200
        data = r.json()
        assert data["intent"] == "scheme"

    def test_chat_district_query(self):
        r = client.post("/api/chat", json={"message": "tell me about Mon district", "district": "Kohima"})
        assert r.status_code == 200
        data = r.json()
        assert data["intent"] == "district"

    def test_chat_unknown(self):
        r = client.post("/api/chat", json={"message": "xyzzy foobar", "district": "Kohima"})
        assert r.status_code == 200
        data = r.json()
        assert data["intent"] == "general"

    def test_chat_nagamese(self):
        r = client.post("/api/chat", json={"message": "ginger-r daam kiman?", "district": "Kohima", "lang": "nag"})
        assert r.status_code == 200
        data = r.json()
        assert data["intent"] == "price"


# ── Profile ───────────────────────────────────────────────────────────────────

class TestProfile:
    def test_valid_profile(self):
        r = client.post("/api/profile", json={
            "name": "Temjen Ao",
            "role": "Farmer",
            "district": "Mokokchung"
        })
        assert r.status_code == 200
        data = r.json()
        assert data["valid"] is True
        assert data["name"] == "Temjen Ao"
        assert data["role"] == "Farmer"
        assert data["district"] == "Mokokchung"

    def test_profile_short_name(self):
        r = client.post("/api/profile", json={"name": "A", "role": "Farmer", "district": "Kohima"})
        assert r.status_code == 400

    def test_profile_invalid_role_defaults(self):
        r = client.post("/api/profile", json={"name": "Test User", "role": "Hacker", "district": "Kohima"})
        assert r.status_code == 200
        data = r.json()
        assert data["role"] == "Farmer"  # Defaults to Farmer

    def test_profile_invalid_district_defaults(self):
        r = client.post("/api/profile", json={"name": "Test User", "role": "Farmer", "district": "Mumbai"})
        assert r.status_code == 200
        data = r.json()
        assert data["district"] == "Kohima"  # Defaults to Kohima


# ── Offline Report Sync ───────────────────────────────────────────────────────

class TestOfflineSync:
    def test_sync_offline_report(self):
        """Reports captured on-device offline sync into the surveillance DB."""
        r = client.post("/disease/report", json={
            "district": "Kohima", "crop": "Potato", "disease": "Late Blight",
            "confidence": 0.99, "severity": "High",
            "reporter": "Sync Test", "reporter_role": "Farmer",
        })
        assert r.status_code == 200
        assert r.json()["ok"] is True

    def test_sync_invalid_district_defaults(self):
        r = client.post("/disease/report", json={
            "district": "Atlantis", "crop": "Maize", "disease": "Common Rust", "confidence": 0.8,
        })
        assert r.status_code == 200
        assert r.json()["district"] == "Kohima"


# ── Exports ───────────────────────────────────────────────────────────────────

class TestExports:
    def test_export_prices_csv(self):
        r = client.get("/api/export/prices?district=Kohima")
        assert r.status_code == 200
        assert "text/csv" in r.headers["content-type"]
        assert "attachment" in r.headers["content-disposition"]
        # Verify CSV content
        lines = r.text.strip().split("\n")
        assert len(lines) > 1  # Header + data
        assert "Crop" in lines[0]

    def test_export_yield_csv(self):
        r = client.get("/api/export/yield?district=Kohima")
        assert r.status_code == 200
        assert "text/csv" in r.headers["content-type"]

    def test_export_pdf_report(self):
        r = client.get("/api/export/report?district=Kohima")
        assert r.status_code == 200
        assert "application/pdf" in r.headers["content-type"]
        assert "attachment" in r.headers["content-disposition"]
        # PDF magic bytes
        assert r.content[:4] == b"%PDF"


# ── Disease Detection (without model — knowledge base fallback) ───────────────

class TestDiseaseDetect:
    def test_detect_no_file(self):
        """Missing file should return 422 (validation error)."""
        r = client.post("/disease/detect", data={"crop": "Potato", "district": "Kohima"})
        assert r.status_code == 422

    def test_detect_invalid_file_type(self):
        """Non-image file should return 400."""
        r = client.post("/disease/detect",
            files={"file": ("test.txt", b"not an image", "text/plain")},
            data={"crop": "Potato", "district": "Kohima"}
        )
        assert r.status_code == 400

    def test_detect_with_reporter(self):
        """Detection with reporter info should include it in response."""
        # Create a minimal valid JPEG (just header bytes — will fail processing but tests the flow)
        # Using knowledge base fallback for non-ML crop
        import io
        from PIL import Image
        img = Image.new("RGB", (224, 224), color="green")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)

        r = client.post("/disease/detect",
            files={"file": ("leaf.jpg", buf, "image/jpeg")},
            data={
                "crop": "Jhum Paddy",  # Not in ML_SUPPORTED_CROPS — uses knowledge base
                "district": "Kohima",
                "reporter": "Test Reporter",
                "reporter_role": "Extension Officer",
            }
        )
        assert r.status_code == 200
        data = r.json()
        assert data["district"] == "Kohima"
        assert data["reporter"] == "Test Reporter"
        assert data["reporter_role"] == "Extension Officer"
        assert "source" in data

    @pytest.mark.skipif(DISEASE_MODEL is None, reason="ML model not available in this environment")
    def test_detect_ml_inference_path(self):
        """Exercise the real model branch (crop in ML_SUPPORTED_CROPS).

        Asserts the response is well-formed regardless of which confidence tier
        the image lands in — this is the headline feature and was previously
        untested (only error/knowledge-base branches had coverage)."""
        import io
        from PIL import Image
        img = Image.new("RGB", (224, 224), color="green")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)

        r = client.post("/disease/detect",
            files={"file": ("potato_leaf.jpg", buf, "image/jpeg")},
            data={"crop": "Potato", "district": "Kohima"},
        )
        assert r.status_code == 200
        data = r.json()
        # The real model ran (not the knowledge-base fallback)
        assert data["source"].startswith("ML model")
        assert 0.0 <= data["confidence"] <= 1.0
        assert data["severity"] in ["High", "Moderate", "Low", "Unknown"]
        assert data["crop"]  # a crop was returned (mapped or user-selected)
        assert "treatment" in data


# ── Confidence Tiering (pure logic — no TensorFlow needed) ────────────────────

class TestConfidenceTier:
    def test_confident_tier(self):
        # High confidence, clear margin over runner-up
        assert _confidence_tier(0.92, 0.40) == "confident"
        assert _confidence_tier(0.70, 0.20) == "confident"  # exactly at threshold

    def test_low_confidence_tier(self):
        # 55–70% with a clear-enough gap
        assert _confidence_tier(0.62, 0.30) == "low_confidence"
        assert _confidence_tier(0.69, 0.16) == "low_confidence"

    def test_uncertain_low_confidence(self):
        # Below 55% is always uncertain
        assert _confidence_tier(0.50, 0.40) == "uncertain"
        assert _confidence_tier(0.10, 0.05) == "uncertain"

    def test_uncertain_close_runner_up(self):
        # Under 70% AND the top-2 gap is too small → uncertain even if >55%
        assert _confidence_tier(0.68, 0.10) == "uncertain"
        assert _confidence_tier(0.60, 0.14) == "uncertain"


# ── Security Headers ──────────────────────────────────────────────────────────

class TestSecurity:
    def test_security_headers_present(self):
        r = client.get("/health")
        assert r.headers["x-content-type-options"] == "nosniff"
        assert r.headers["x-frame-options"] == "DENY"
        assert r.headers["referrer-policy"] == "strict-origin-when-cross-origin"
        assert "camera=(self)" in r.headers["permissions-policy"]

    def test_cors_headers(self):
        r = client.options("/api/status", headers={
            "Origin": "http://localhost:8000",
            "Access-Control-Request-Method": "GET",
        })
        # CORS preflight should succeed for allowed origin
        assert r.status_code == 200


# ── Edge Cases ────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_nonexistent_endpoint(self):
        r = client.get("/api/nonexistent")
        assert r.status_code in [404, 405]

    def test_prices_empty_params(self):
        r = client.get("/prices?crop=&district=")
        assert r.status_code == 200

    def test_chat_empty_message(self):
        r = client.post("/api/chat", json={"message": "", "district": "Kohima"})
        assert r.status_code == 200

    def test_chat_very_long_message(self):
        r = client.post("/api/chat", json={"message": "a" * 5000, "district": "Kohima"})
        assert r.status_code == 200

    def test_calendar_unknown_district(self):
        """Unknown district should still return 200 (may include general crops)."""
        r = client.get("/calendar?district=FakeDistrict")
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data["calendar"], list)
