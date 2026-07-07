/**
 * Plotwise Offline Engine — reproduces the server's data logic on-device so the
 * app's core tools (prices, planting calendar, scheme finder, disease treatment
 * guide) work with ZERO network. Reads window.PLOTWISE_OFFLINE (offline-data.js).
 *
 * The on-device AI (photo diagnosis) is handled separately by offline-ai.js.
 * Weather is intentionally NOT offline — it is live external data.
 */
(function () {
  const D = window.PLOTWISE_OFFLINE;
  if (!D) { console.warn('offline-data not loaded'); return; }

  // Small deterministic hash (stable per string) — used for indicative daily
  // price variation without needing md5. Values are indicative, same framing
  // as the server ("anchored to MSP 2023-24").
  function hash32(str) {
    let h = 2166136261 >>> 0;
    for (let i = 0; i < str.length; i++) {
      h ^= str.charCodeAt(i);
      h = Math.imul(h, 16777619) >>> 0;
    }
    return h >>> 0;
  }
  function todayISO() { return new Date().toISOString().slice(0, 10); }

  function getPrices(crop, district) {
    const day = todayISO();
    const list = crop ? [crop] : Object.keys(D.price_anchors);
    const prices = [];
    for (const c of list) {
      const a = D.price_anchors[c];
      if (!a) continue;
      const spread = 200;
      const off = (hash32(c + ':' + day) % (spread * 2 + 1)) - spread;
      const price = a.base + off;
      const trend = ['up', 'stable', 'down'][hash32('t:' + c) % 3];
      const pct = (0.3 + (hash32('trend:' + c + ':' + day) % 43) / 10).toFixed(1);
      prices.push({
        crop: c, price_per_qtl: price, msp: a.base, unit: a.unit,
        market: district || 'Nagaland APMC', trend, trend_pct: pct + '%',
        last_updated: day,
        tip: price > a.base ? 'Above MSP — good time to sell'
           : price < a.base ? 'Below MSP — consider holding' : 'At MSP',
      });
    }
    return { prices, source: 'Indicative — MSP 2023-24 anchor (offline)', records: prices.length, _offline: true };
  }

  function inWindow(month, win) {
    const s = win[0], e = win[win.length - 1];
    return s <= e ? (month >= s && month <= e) : (month >= s || month <= e);
  }
  function getCalendar(district, crop) {
    const month = new Date().getMonth() + 1;
    const names = D.month_names;
    const list = crop ? [crop] : Object.keys(D.planting_calendar);
    const cal = [];
    for (const c of list) {
      const info = D.planting_calendar[c];
      if (!info) continue;
      if (info.zones !== 'all' && district && !info.zones.includes(district)) continue;
      let status;
      if (inWindow(month, info.sow)) status = 'sowing now';
      else if (inWindow(month, info.harvest)) status = 'harvest time';
      else if (inWindow(month, [info.sow[info.sow.length - 1], info.harvest[0]])) status = 'growing';
      else status = 'upcoming';
      cal.push({
        crop: c,
        sow_window: info.sow.map(m => names[m]),
        harvest_window: info.harvest.map(m => names[m]),
        status, suitable_for: district || 'all', avg_yield_kg_ha: null,
      });
    }
    return { district: district || 'Kohima', calendar: cal, _offline: true };
  }

  function getSchemes(crop, district) {
    const matched = D.schemes.filter(s => !s.crops.length || s.crops.includes(crop));
    return {
      district, crop, matched_schemes: matched.length, schemes: matched,
      tip: 'Visit your nearest Block Development Office or Common Service Centre to apply in person.',
      _offline: true,
    };
  }

  // Disease treatment guide (knowledge base) — for browsing without a photo.
  function getTreatments(crop) {
    const diseases = D.diseases[crop] || D.diseases['default'] || [];
    return diseases.map(dz => ({ disease: dz, treatment: D.treatments[dz] || 'Consult your District Agriculture Officer.' }));
  }

  window.PlotwiseOffline = { getPrices, getCalendar, getSchemes, getTreatments, inWindow, data: D };
})();
