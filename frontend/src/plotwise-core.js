/**
 * Plotwise Core — Shared utilities for desktop and mobile frontends.
 *
 * Provides: API layer (timeout + retry + caching), constants,
 * chat functions, i18n engine, offline detection.
 *
 * Usage: Include via <script src="/static/plotwise-core.js"></script>
 * before the page-specific <script> block.
 */

/* ── API Layer ────────────────────────────────────────────────────── */

// A bundled native app is served from a local origin (capacitor://localhost)
// that has no backend, so live API calls must go to the hosted server. Any
// native app (bundled OR remote-loaded) should use the server; the web uses
// its own origin.
const PLOTWISE_SERVER = 'https://limajmr-plotwise.hf.space';
const _isNativeApp = !!(window.Capacitor && window.Capacitor.isNativePlatform && window.Capacitor.isNativePlatform());
const API = _isNativeApp ? PLOTWISE_SERVER : window.location.origin;

const _apiCache = new Map();   // key → { ts, data }
const _lsPrefix = 'plotwise_cache_';

/**
 * Fetch JSON from an API endpoint with timeout, retry, and caching.
 *
 * @param {string} url           Full URL to fetch
 * @param {object} options       Fetch options (method, headers, body…)
 * @param {object} config        { timeout, retries, cacheTTL }
 *   - timeout:  ms before aborting (default 15000)
 *   - retries:  retry count on failure (default 1)
 *   - cacheTTL: ms to cache response in memory (0 = no cache)
 *   - persist:  also cache in localStorage for offline fallback
 * @returns {Promise<any>}       Parsed JSON
 */
async function apiFetch(url, options = {}, { timeout = 15000, retries = 1, cacheTTL = 0, persist = false } = {}) {
  const cacheKey = url + (options.body || '');

  // 1. Check in-memory cache
  if (cacheTTL > 0) {
    const cached = _apiCache.get(cacheKey);
    if (cached && (Date.now() - cached.ts) < cacheTTL) {
      return cached.data;
    }
  }

  // 2. Attempt fetch with timeout + retry
  let lastError;
  for (let attempt = 0; attempt <= retries; attempt++) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeout);
    try {
      const res = await fetch(url, { ...options, signal: controller.signal });
      clearTimeout(timer);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();

      // Cache on success
      if (cacheTTL > 0) _apiCache.set(cacheKey, { ts: Date.now(), data });
      if (persist) {
        try { localStorage.setItem(_lsPrefix + cacheKey, JSON.stringify({ ts: Date.now(), data })); } catch (_) {}
      }
      return data;
    } catch (err) {
      clearTimeout(timer);
      lastError = err;
      if (attempt < retries) {
        await new Promise(r => setTimeout(r, 1000 * (attempt + 1)));
      }
    }
  }

  // 3a. Offline fallback — compute the result on-device (prices/calendar/schemes)
  const offline = _offlineCompute(url, options);
  if (offline) return offline;

  // 3b. Offline fallback — replay the last cached response from localStorage
  if (persist) {
    try {
      const stored = JSON.parse(localStorage.getItem(_lsPrefix + cacheKey));
      if (stored && stored.data) {
        stored.data._fromCache = true;
        stored.data._cachedAt = stored.ts;
        return stored.data;
      }
    } catch (_) {}
  }

  throw lastError;
}

// Reproduce prices/calendar/schemes on-device via the offline engine so these
// tools work with zero network. Returns null if not offline-serviceable.
function _offlineCompute(url, options) {
  if (typeof window === 'undefined' || !window.PlotwiseOffline) return null;
  try {
    const u = new URL(url, (typeof location !== 'undefined' ? location.origin : 'http://localhost'));
    const p = u.pathname, q = u.searchParams;
    if (p.endsWith('/prices'))   return { ...window.PlotwiseOffline.getPrices(q.get('crop') || '', q.get('district') || ''), _fromCache: true };
    if (p.endsWith('/calendar')) return { ...window.PlotwiseOffline.getCalendar(q.get('district') || 'Kohima', q.get('crop') || ''), _fromCache: true };
    if (p.endsWith('/schemes')) {
      let body = {}; try { body = JSON.parse(options.body || '{}'); } catch (_) {}
      return { ...window.PlotwiseOffline.getSchemes(body.crop || '', body.district || ''), _fromCache: true };
    }
  } catch (_) {}
  return null;
}


/* ── Constants ────────────────────────────────────────────────────── */

const DISTRICTS = [
  'Kohima','Tseminyu','Phek','Mokokchung','Tuensang','Noklak',
  'Shamator','Mon','Dimapur','Niuland','Chumoukedima','Wokha',
  'Zunheboto','Peren','Kiphire','Longleng'
];

const WEATHER_ICONS = {
  0:'☀️',1:'🌤️',2:'⛅',3:'☁️',45:'🌫️',48:'🌫️',
  51:'🌦️',53:'🌧️',55:'🌧️',61:'🌦️',63:'🌧️',65:'🌧️',
  71:'🌨️',73:'🌨️',75:'❄️',80:'🌦️',81:'🌧️',82:'⛈️',
  95:'⛈️',96:'⛈️',99:'⛈️'
};

const TC = {
  "sowing now":"tag-sow","harvest time":"tag-hvst",
  "growing":"tag-grow","off season":"tag-off","upcoming":"tag-off"
};

// District + crop stats for dashboards (2023-24 verified data)
// 'c' = distinct crops recorded per district, derived from the source CSV
// (data/sample/nagaland_crop_data_2023_24.csv) so it matches the API and PDF.
const DS = [
  {d:"Mon",a:45458,p:131351,c:39},{d:"Phek",a:34318,p:109855,c:37},
  {d:"Niuland",a:29071,p:105496,c:41},{d:"Wokha",a:33033,p:102234,c:38},
  {d:"Peren",a:26110,p:93978,c:40},{d:"Chumoukedima",a:27471,p:90441,c:41},
  {d:"Mokokchung",a:29918,p:88471,c:39},{d:"Zunheboto",a:29880,p:76288,c:34},
  {d:"Kohima",a:20525,p:66622,c:33},{d:"Kiphire",a:24533,p:60808,c:35},
  {d:"Tuensang",a:21014,p:61243,c:32},{d:"Noklak",a:15886,p:46266,c:31},
  {d:"Longleng",a:15760,p:44334,c:32},{d:"Shamator",a:14241,p:37915,c:32},
  {d:"Dimapur",a:10520,p:37325,c:39},{d:"Tseminyu",a:9849,p:26430,c:33}
];

const GREENS = ['#5a9e3a','#3a6828','#8fce52','#2a4e1e','#78b840','#2e5522','#a8d866','#1a3212','#4b8a30','#6ab445'];


/* ── Output escaping ──────────────────────────────────────────────── */

/**
 * HTML-escape a value before it is interpolated into innerHTML. Use for ANY
 * string that originates from the server/DB/user (crop, disease, reporter,
 * district, chat text). This is the primary defense against stored/reflected
 * XSS in the dashboard, chat and result cards.
 */
function escHtml(s) {
  return String(s == null ? '' : s).replace(/[&<>"']/g, c => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  }[c]));
}
if (typeof window !== 'undefined') window.escHtml = escHtml;


/* ── Chat Functions ───────────────────────────────────────────────── */

// Chat element IDs — overridden per page via Plotwise.init()
const _chat = {
  input: 'chatInput',
  messages: 'chatMessages',
  typing: 'chatTyping',
  chips: 'chatChips',
  getDistrict: () => 'Kohima',
};

function sendChat(text) {
  const input = document.getElementById(_chat.input);
  const message = text || input.value.trim();
  if (!message) return;

  input.value = '';
  addChatMsg(escHtml(message), 'user');   // escape user text before innerHTML

  const typing = document.getElementById(_chat.typing);
  typing.classList.add('show');
  scrollChatBottom();

  const district = _chat.getDistrict();

  apiFetch(API + '/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, district, lang: currentLang })
  }, { timeout: 10000, retries: 0 })
  .then(data => {
    typing.classList.remove('show');
    // Escape the reply FIRST, then re-introduce only the intended **bold**/newline
    // markup — so any HTML the server may have reflected is neutralised.
    const formatted = escHtml(data.reply)
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/\n/g, '<br>');
    addChatMsg(formatted, 'bot');
    if (data.suggestions && data.suggestions.length) {
      updateChips(data.suggestions);
    }
  })
  .catch(() => {
    typing.classList.remove('show');
    if (!navigator.onLine) {
      addChatMsg('You appear to be offline. Chat requires an internet connection.', 'bot');
    } else {
      addChatMsg('Sorry, I could not connect. Please try again.', 'bot');
    }
  });
}

function addChatMsg(html, type) {
  const container = document.getElementById(_chat.messages);
  const msg = document.createElement('div');
  msg.className = 'chat-msg ' + type;
  msg.innerHTML = html;
  container.appendChild(msg);
  scrollChatBottom();
}

function scrollChatBottom() {
  const container = document.getElementById(_chat.messages);
  requestAnimationFrame(() => { container.scrollTop = container.scrollHeight; });
}

function updateChips(suggestions) {
  const container = document.getElementById(_chat.chips);
  container.innerHTML = suggestions.map(s =>
    `<button class="chat-chip" onclick="sendChat('${String(s).replace(/[\\']/g, '\\$&')}')">${escHtml(s)}</button>`
  ).join('');
}


/* ── i18n Engine ──────────────────────────────────────────────────── */

// Each page sets window._pageI18n before loading this file,
// or calls Plotwise.init({ i18n: {...} }).
// The merged dictionary is stored here.
let _i18nDict = {};
let currentLang = localStorage.getItem('plotwise-lang') || 'en';

function setLang(lang) {
  currentLang = lang;
  localStorage.setItem('plotwise-lang', lang);

  const btnEn = document.getElementById('btnEn');
  const btnNag = document.getElementById('btnNag');
  if (btnEn) btnEn.classList.toggle('active', lang === 'en');
  if (btnNag) btnNag.classList.toggle('active', lang === 'nag');

  document.querySelectorAll('[data-i18n]').forEach(el => {
    const key = el.getAttribute('data-i18n');
    if (_i18nDict[lang] && _i18nDict[lang][key]) {
      el.innerHTML = _i18nDict[lang][key];
    }
  });
  document.documentElement.lang = lang === 'nag' ? 'nag' : 'en';
}

/**
 * Translate a runtime-generated string by key, for text built in JS (result
 * cards, status pills, trend labels) that can't carry a data-i18n attribute.
 * Falls back to the English entry, then to the provided fallback, then the key
 * itself — so an untranslated string degrades to English, never to blank.
 *
 * @param {string} key       Dictionary key
 * @param {string} [fallback] Text to use if no translation exists
 * @returns {string}
 */
function t(key, fallback) {
  const dict = _i18nDict[currentLang];
  if (dict && dict[key] != null) return dict[key];
  const en = _i18nDict.en;
  if (en && en[key] != null) return en[key];
  return fallback != null ? fallback : key;
}


/* ── Loading & Error UI Helpers ───────────────────────────────────── */

/**
 * Generate a standardized loading spinner HTML.
 * @param {string} msg - Loading message to display
 * @returns {string} HTML string
 */
function pwLoading(msg = 'Loading…') {
  return `<div style="padding:24px;text-align:center">
    <div style="display:inline-block;width:24px;height:24px;border:2.5px solid rgba(106,191,59,0.15);border-top-color:var(--sprout,#6abf3b);border-radius:50%;animation:spin .8s linear infinite;margin-bottom:8px"></div>
    <div style="color:rgba(240,235,224,0.45);font-size:0.82rem">${msg}</div>
  </div>`;
}

/**
 * Generate a standardized error message with optional retry button.
 * @param {string} msg - User-friendly error message
 * @param {string} [retryFn] - Function name string for retry button (e.g. "loadPrices()")
 * @returns {string} HTML string
 */
function pwError(msg, retryFn) {
  const offline = !navigator.onLine;
  const displayMsg = offline ? 'You appear to be offline. Check your connection.' : msg;
  const retryBtn = retryFn
    ? `<button onclick="${retryFn}" style="margin-top:10px;padding:8px 18px;background:rgba(106,191,59,0.12);border:1px solid rgba(106,191,59,0.2);border-radius:8px;color:var(--sprout,#6abf3b);font-size:0.78rem;font-weight:600;cursor:pointer">Retry</button>`
    : '';
  return `<div style="padding:20px;text-align:center">
    <div style="color:#f87171;font-size:0.84rem;margin-bottom:4px">${displayMsg}</div>
    ${retryBtn}
  </div>`;
}


/* ── Offline Detection ────────────────────────────────────────────── */

let _offlineBarId = 'offline-bar';

function updateOnlineStatus() {
  const bar = document.getElementById(_offlineBarId);
  if (!bar) return;
  // Support both CSS patterns: transform-based (desktop) and class-based (mobile)
  if (bar.classList.contains('offline-bar-class')) {
    bar.classList.toggle('show', !navigator.onLine);
  } else {
    bar.style.transform = navigator.onLine ? 'translateY(-100%)' : 'translateY(0)';
  }
}

window.addEventListener('online', updateOnlineStatus);
window.addEventListener('offline', updateOnlineStatus);


/* ── Initialization ───────────────────────────────────────────────── */

/**
 * Configure Plotwise core for the current page.
 *
 * @param {object} config
 *   - chatElements: { input, messages, typing, chips } — DOM element IDs
 *   - getDistrict:  () => string — function returning current district
 *   - i18n:         { en: {...}, nag: {...} } — page-specific translations
 *   - offlineBarId: string — ID of offline notification bar
 */
function plotwiseInit(config = {}) {
  if (config.chatElements) Object.assign(_chat, config.chatElements);
  if (config.getDistrict) _chat.getDistrict = config.getDistrict;
  if (config.i18n) _i18nDict = config.i18n;
  if (config.offlineBarId) _offlineBarId = config.offlineBarId;
}

// Also check for pre-set page i18n (set before this script loads)
if (window._pageI18n) _i18nDict = window._pageI18n;
