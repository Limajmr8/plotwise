# Plotwise — Pre-Demo Checklist

**Audience**: Joint Director, Nagaland Agriculture Department
**Duration**: ~15 minutes
**Devices**: Laptop (projected) + Android phone (passed around)

---

## Environment Setup (Day Before)

- [ ] Railway deployment is live and responding at `/health`
- [ ] UptimeRobot pinging `/health` every 5 min (keeps container warm)
- [ ] Database has seed data (run `python scripts/seed_demo_data.py` if empty)
- [ ] Prepare 5-6 leaf images that produce confident results (>70%)
- [ ] Charge Android phone, install APK, verify it loads
- [ ] Mobile hotspot ready as backup internet
- [ ] Check venue projector resolution — test at 1280x720 minimum

---

## Desktop UI (Projected Laptop)

### Landing Page
- [ ] Hero section loads with animation (particles, counters, clip-path reveals)
- [ ] Stats counter animates: 16 districts, 44 crops, production figure, 24 disease classes
- [ ] Language toggle EN/NAG — all visible text switches
- [ ] Scroll animations fire (features section, how-it-works, CTA)
- [ ] Ticker at top shows live crop prices

### Disease Detection Tab
- [ ] Select crop (Potato), select district (Kohima)
- [ ] Upload a prepared leaf image
- [ ] Result appears within 5 seconds with disease name, confidence %, severity, treatment
- [ ] Heatmap section updates with new report
- [ ] "Recent Reports" shows reporter name (set profile in localStorage first)

### Market Prices Tab
- [ ] All 33 crops display with price, MSP, trend
- [ ] Filter by crop works
- [ ] Download CSV button produces valid file
- [ ] Download PDF button produces branded report

### Planting Calendar Tab
- [ ] Crops show with sow/harvest windows
- [ ] Status badges correct for current month (May = "sowing now" for Jhum Paddy)
- [ ] District filter changes which crops appear

### Scheme Finder Tab
- [ ] Enter crop + district, click "Find My Schemes"
- [ ] PM-KISAN always appears (universal)
- [ ] Crop-specific schemes appear for Ginger, Potato etc.

### Weather Tab
- [ ] Live temperature, humidity, wind for Kohima
- [ ] 7-day forecast cards render
- [ ] Farming advisory shows context-appropriate message
- [ ] District switcher loads different coordinates

### Chat Widget
- [ ] Click floating bubble to open
- [ ] Type "hello" — greeting with suggestion chips
- [ ] Click "What's the price of ginger?" — price response with MSP
- [ ] "My potato has spots" — disease info with treatment
- [ ] "Weather in Dimapur" — live weather data
- [ ] Nagamese mode: "ginger-r daam" — responds correctly

### Dashboard Section
- [ ] Heatmap grid shows all 16 districts with report counts
- [ ] Color coding: green (low) → orange (medium) → red (high)
- [ ] Recent Reports section shows reporter names + roles

---

## Mobile App (Android Phone)

### Profile Setup
- [ ] Profile card shows "Set up your profile" on first open
- [ ] Tap card → modal appears
- [ ] Enter name, select "Extension Officer", select district
- [ ] Save → profile card updates with initials avatar

### Home Screen
- [ ] KPI cards show 16 districts, 44 crops, production, area
- [ ] Weather summary loads for default district
- [ ] Price ticker scrolls horizontally
- [ ] Top crops chart renders

### Disease Detection
- [ ] Camera opens when tapping dropzone
- [ ] Take photo of prepared leaf
- [ ] "Analysing..." spinner appears
- [ ] Result shows disease + confidence + treatment
- [ ] Reporter name from profile is included

### Market Prices
- [ ] Prices load for all crops
- [ ] PDF download works on phone
- [ ] Filter by crop works

### Farm Tools
- [ ] Calendar sub-tab: crops show with status badges
- [ ] Schemes sub-tab: enter crop → relevant schemes appear
- [ ] Weather sub-tab: 7-day forecast with icons

### Chat
- [ ] Bottom nav → Chat opens
- [ ] Quick chips work (tap "Potato disease")
- [ ] Full conversation flows naturally

---

## Offline Resilience

- [ ] Turn off WiFi on phone
- [ ] Offline bar appears at top
- [ ] Previously loaded data still visible (cached)
- [ ] Chat shows "You appear to be offline" message
- [ ] Turn WiFi back on → bar disappears, data refreshes

---

## PDF Report (Key Demo Artifact)

- [ ] Generate PDF for Kohima district
- [ ] Verify: "Plotwise — District Agriculture Intelligence Report" header
- [ ] Verify: "Prepared for: Director of Agriculture, Government of Nagaland"
- [ ] Verify: Summary table with area, production, yield
- [ ] Verify: Top crops table
- [ ] Verify: Disease surveillance section with reporter names
- [ ] Verify: Market prices section
- [ ] Verify: Page numbers in footer
- [ ] Print one copy to hand to officials

---

## Failure Recovery

| Scenario | Response |
|----------|----------|
| Railway is down | Switch to local: `uvicorn backend.src.main:app` on laptop |
| No internet at venue | Use mobile hotspot; cached data still shows |
| ML model slow (>10s) | "The AI is processing — let me show you the dashboard while we wait" |
| Phone camera fails | Use pre-uploaded test images instead |
| PDF won't download | Open in browser tab: `/api/export/report?district=Kohima` |

---

## Key Talking Points

1. **"This runs on real data"** — 576 records from Director of Agriculture office
2. **"AI-powered but honest"** — Three-tier confidence; says "uncertain" when unsure
3. **"Works for all 16 districts"** — Not just Kohima demo data
4. **"Offline-capable"** — Farmers in remote areas can still access cached data
5. **"Reporter accountability"** — Every detection tracked with who reported it
6. **"One platform, all tools"** — Disease + prices + calendar + schemes + weather + chat
