# Plotwise — Railway Deployment Setup

Railway containers are **ephemeral**: their local filesystem is wiped on every
redeploy. Without a persistent volume, the SQLite database (disease reports that
feed the surveillance heatmap and the PDF report) resets to empty on each deploy.
This guide makes the data survive.

## 1. Attach a persistent volume

In the Railway dashboard → your service → **Variables / Volumes**:

1. Add a **Volume** mounted at: `/data`
2. Add environment variables:

   | Variable | Value | Why |
   |----------|-------|-----|
   | `PLOTWISE_DB_PATH` | `/data/plotwise.db` | Store the DB on the persistent volume |
   | `PLOTWISE_SEED_ON_EMPTY` | `1` | Seed ~70 demo reports on first boot if the DB is empty (idempotent) |

That's it. On the next deploy the app writes to `/data/plotwise.db`, which
persists across redeploys.

## 2. First-boot seeding (optional, for the demo)

With `PLOTWISE_SEED_ON_EMPTY=1`, the very first boot against an empty volume
inserts ~70 realistic, deterministic disease reports across all 16 districts so
the heatmap and PDF look populated. Once any report exists, the seeder does
nothing — real detections accumulate normally.

To seed manually instead (e.g. locally), run:

```
python scripts/seed_demo_data.py
```

## 3. Verify after deploy

```
curl https://plotwise-production.up.railway.app/health
```

`disease_reports_total` should be > 0 and stay stable across a redeploy.

## 4. Keep the container warm (demo day)

Railway may idle the container, causing a slow cold start (TensorFlow load).
Point a free UptimeRobot monitor at `/health` every 5 minutes during the
demo window so the first real request on stage is fast. See `docs/SIGNING_SETUP.md`
and `docs/DEMO_CHECKLIST.md` for the rest of the pre-demo checklist.
