/**
 * Plotwise Offline AI — runs the disease model ON the phone via TensorFlow.js,
 * so photo diagnosis works with zero network. Mirrors the server's crop-aware
 * inference exactly (restrict to the selected crop's classes, within-crop
 * confidence, raw-peak safety net, 3-tier confidence, label mapping).
 *
 * Requires: tf (tfjs), window.PLOTWISE_OFFLINE (offline-data.js).
 */
(function () {
  const D = window.PLOTWISE_OFFLINE;
  let _model = null, _loading = null;

  const MODEL_URL = '/static/tfjs-model/model.json';

  function ready() { return typeof tf !== 'undefined' && !!D; }

  async function load() {
    if (_model) return _model;
    if (_loading) return _loading;
    _loading = tf.loadGraphModel(MODEL_URL).then(m => { _model = m; return m; });
    return _loading;
  }
  // Warm the model in the background once tf is present (first inference is then instant)
  function preload() { if (ready()) load().catch(() => {}); }

  function baseCrop(crop) {
    const c = (crop || '').trim();
    if (D.crop_to_classes[c]) return c;
    if (c.startsWith('Maize')) return 'Maize';
    if (c.startsWith('Tomato')) return 'Tomato';
    if (c === 'Soybean' || c === 'Soyabean') return 'Soyabean';
    if (c.startsWith('Chilli')) return 'Chilli';
    if (c.startsWith('Pepper')) return 'Pepper';
    return null;
  }

  function tier(conf, gap) {
    if (conf < 0.55 || (conf < 0.70 && gap < 0.15)) return 'uncertain';
    if (conf < 0.70) return 'low_confidence';
    return 'confident';
  }

  function detectCropFromLabel(label, fallback) {
    const MAP = [['Healthy_Maize','Maize'],['Healthy_Pepper','Chilli'],['Healthy_Potato','Potato'],
      ['Healthy_Tomato','Tomato'],['Soybean_','Soybean'],['Apple_','Apple'],['Grape_','Grape'],
      ['Orange_','Orange'],['Maize_','Maize'],['Potato_','Potato'],['Tomato_','Tomato'],
      ['Pepper_','Chilli'],['Chilli_','Chilli']];
    for (const [p, c] of MAP) if (label.startsWith(p)) return c;
    return fallback;
  }

  // imgEl: an <img> or canvas already holding the leaf photo.
  async function detect(imgEl, crop, district, reporter, reporterRole) {
    const model = await load();
    const labelByIdx = D.class_indices;           // {"0":"Apple_AppleScab",...}
    const idxByLabel = {}; for (const k in labelByIdx) idxByLabel[labelByIdx[k]] = +k;

    // Preprocess: 224x224 RGB float32, NO /255 (EfficientNetB0 built-in preprocessing)
    const probs = tf.tidy(() => {
      const x = tf.browser.fromPixels(imgEl).resizeBilinear([224, 224]).toFloat().expandDims(0);
      return model.predict(x);
    });
    const preds = await probs.data();
    probs.dispose();

    const rawPeak = Math.max.apply(null, preds);
    const base = baseCrop(crop);
    const allowed = base ? (D.crop_to_classes[base] || []).map(l => idxByLabel[l]).filter(i => i != null) : [];

    let classIdx, confidence, gap;
    if (allowed.length >= 2) {
      const sum = allowed.reduce((s, i) => s + preds[i], 0) || 1;
      const norm = allowed.map(i => preds[i] / sum);
      const order = norm.map((v, k) => [v, k]).sort((a, b) => b[0] - a[0]);
      classIdx = allowed[order[0][1]];
      confidence = order[0][0];
      gap = order[0][0] - (order[1] ? order[1][0] : 0);
      if (rawPeak < 0.30) { confidence = rawPeak; gap = 0; }
    } else {
      let mi = 0; for (let i = 1; i < preds.length; i++) if (preds[i] > preds[mi]) mi = i;
      const sorted = Array.from(preds).sort((a, b) => b - a);
      classIdx = mi; confidence = preds[mi]; gap = sorted[0] - sorted[1];
    }

    const rawLabel = labelByIdx[String(classIdx)] || ('class_' + classIdx);
    const t = tier(confidence, gap);
    const PREV = 'Use certified seeds every season. Maintain field hygiene. Rotate crops annually to break disease cycles.';
    const HELP = 'Contact District Agriculture Officer or call Kisan Call Centre: 1800-180-1551 (toll free)';

    if (t === 'uncertain') {
      return {
        crop, district, disease: 'Uncertain — could not identify clearly',
        confidence: Math.round(confidence * 1000) / 1000, severity: 'Unknown',
        treatment: 'The image was unclear or the disease is not in our training data. Try: (1) a closer photo of a single leaf, (2) good lighting, (3) avoid blur. If symptoms persist, contact your District Agriculture Officer.',
        prevention: PREV, nearest_help: HELP,
        source: 'On-device AI (uncertain — retake photo)', reporter, reporter_role: reporterRole, offline: true,
      };
    }
    const disease = D.label_map[rawLabel] || rawLabel;
    const detectedCrop = detectCropFromLabel(rawLabel, crop);
    const severity = confidence >= 0.85 ? 'High' : confidence >= 0.70 ? 'Moderate' : 'Low';
    return {
      crop: detectedCrop, district, disease,
      confidence: Math.round(confidence * 1000) / 1000, severity,
      treatment: D.treatments[disease] || 'Consult your District Agriculture Officer.',
      prevention: PREV, nearest_help: HELP,
      source: t === 'low_confidence' ? 'On-device AI (low confidence — verify with expert)' : 'On-device AI (EfficientNetB0)',
      reporter, reporter_role: reporterRole, offline: true,
    };
  }

  window.PlotwiseAI = { load, preload, detect, ready, isLoaded: () => !!_model };
})();
