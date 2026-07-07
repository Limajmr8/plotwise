#!/usr/bin/env bash
# Assemble the offline-capable Capacitor bundle (www/) from frontend/src.
# The mobile app + AI model + data are shipped INSIDE the APK, so disease
# detection and the data tools work with zero network. Live features (weather,
# online prices, offline-report sync) call the hosted server via API base.
set -e
cd "$(dirname "$0")/.."
SRC=frontend/src
OUT=www

rm -rf "$OUT"
mkdir -p "$OUT/static/tfjs-model"

# App shell — mobile UI becomes the bundle's index.html
cp "$SRC/mobile.html" "$OUT/index.html"

# Static assets referenced as /static/... by the app
cp "$SRC/plotwise-core.js"     "$OUT/static/"
cp "$SRC/offline-data.js"      "$OUT/static/"
cp "$SRC/offline-engine.js"    "$OUT/static/"
cp "$SRC/offline-ai.js"        "$OUT/static/"
cp "$SRC/tf.min.js"            "$OUT/static/"
cp "$SRC/chart.umd.min.js"     "$OUT/static/"
cp "$SRC/manifest.json"        "$OUT/static/"
cp "$SRC/icon-192.png"         "$OUT/static/" 2>/dev/null || true
cp "$SRC/icon-512.png"         "$OUT/static/" 2>/dev/null || true
cp "$SRC/sw.js"                "$OUT/"        2>/dev/null || true
cp "$SRC/tfjs-model/"*         "$OUT/static/tfjs-model/"

echo "Bundle built at $OUT/"
du -sh "$OUT"
find "$OUT" -type f | sed "s|$OUT/||" | sort
