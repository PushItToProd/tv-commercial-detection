// ── shared state (accessible from popup via getBackgroundPage()) ────────────

const captureState = {
  running: false,
  tabId: null,  // tab being monitored; capture stops if this tab is closed
  log: []       // ring buffer of { msg, type } — replayed into popup on open
};

const ALARM_NAME = 'frame-capture';
const LOG_MAX = 60;

// ── log helper ──────────────────────────────────────────────────────────────

function bgLog(msg, type = '') {
  const entry = { msg, type };
  captureState.log.push(entry);
  if (captureState.log.length > LOG_MAX) captureState.log.shift();

  // forward to popup if it's open
  const popup = browser.extension.getViews({ type: 'popup' })[0];
  if (popup) {
    popup.dispatchEvent(new popup.CustomEvent('capture-log', { detail: entry }));
  }

  console.log(`[frame-capture] ${msg}`);
}

// ── alarm ───────────────────────────────────────────────────────────────────

async function startCapture() {
  if (captureState.running) return;   // guard against double-start
  const [tab] = await browser.tabs.query({ active: true, currentWindow: true });
  if (!tab) { bgLog('No active tab — cannot start.', 'err'); return; }
  captureState.tabId = tab.id;
  captureState.running = true;
  const periodInMinutes = await getIntervalMinutes();
  browser.alarms.create(ALARM_NAME, { periodInMinutes });
  // fire immediately so the user sees it working right away
  doCapture();
}

function stopCapture() {
  captureState.running = false;
  captureState.tabId = null;
  browser.alarms.clear(ALARM_NAME);
}

function restartAlarm(intervalSeconds) {
  browser.alarms.clear(ALARM_NAME);
  browser.alarms.create(ALARM_NAME, { periodInMinutes: intervalSeconds / 60 });
}

async function getIntervalMinutes() {
  const { config = {} } = await browser.storage.local.get('config');
  return (config.interval ?? 10) / 60;
}

browser.alarms.onAlarm.addListener(alarm => {
  if (alarm.name === ALARM_NAME) doCapture();
});

browser.tabs.onRemoved.addListener(tabId => {
  if (captureState.running && tabId === captureState.tabId) {
    stopCapture();
    bgLog('Monitored tab closed — capture stopped.', 'err');
  }
});

// ── core capture ─────────────────────────────────────────────────────────────

async function doCapture() {
  if (!captureState.running) return;

  let config = {};
  try {
    ({ config = {} } = await browser.storage.local.get('config'));
  } catch (e) {
    bgLog('Could not read config: ' + e.message, 'err');
    return;
  }

  const endpoints = config.endpoints ?? [];
  if (endpoints.length === 0) {
    bgLog('No endpoints configured — stopping.', 'err');
    stopCapture();
    return;
  }

  // capture from the specific tab that was active when capture started
  let tab;
  try {
    tab = await browser.tabs.get(captureState.tabId);
  } catch (e) {
    bgLog('Monitored tab is gone — stopping.', 'err');
    stopCapture();
    return;
  }

  if (!tab) { bgLog('No active tab.', 'err'); return; }

  try {
    // 1. get video info from page
    const results = await browser.tabs.executeScript(tab.id, { file: 'content_scripts/get_video_bounds.js' });
    const info = results[0];

    // skip this tick if there's no video at all
    if (!info) { bgLog('No video found — skipping.', ''); return; }

    const isPaused = !info.playing;
    const timestamp = new Date().toISOString();

    let blob = null;
    if (!isPaused) {
      // 2. screenshot
      const dataUrl = await browser.tabs.captureTab(tab.tabId, { format: 'png' });

      // 3. crop to the video rect
      const finalUrl = await cropImage(dataUrl, info);
      blob = dataUrlToBlob(finalUrl);
    }

    // 4. POST to each endpoint concurrently
    const posts = endpoints.map(async url => {
      const form = new FormData();
      if (blob) form.append('image', blob, `frame_${timestamp}.png`);
      form.append('is_paused', isPaused ? 'true' : 'false');
      form.append('timestamp', timestamp);
      form.append('page_title', tab.title ?? '');
      form.append('page_url', tab.url ?? '');

      try {
        const res = await fetch(url, { method: 'POST', body: form });
        if (res.ok) {
          bgLog(`POST ${url} → ${res.status}${isPaused ? ' (paused)' : ''}`, 'ok');
        } else {
          bgLog(`POST ${url} → ${res.status} ${res.statusText}`, 'err');
        }
      } catch (e) {
        bgLog(`POST ${url} failed: ${e.message}`, 'err');
      }
    });

    await Promise.all(posts);

  } catch (err) {
    bgLog('Capture error: ' + err.message, 'err');
  }
}

// ── image utilities ──────────────────────────────────────────────────────────

function cropImage(dataUrl, rect) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement('canvas');
      canvas.width = rect.width;
      canvas.height = rect.height;
      canvas.getContext('2d').drawImage(
        img,
        rect.x, rect.y, rect.width, rect.height,
        0, 0, rect.width, rect.height
      );
      resolve(canvas.toDataURL('image/png'));
    };
    img.onerror = reject;
    img.src = dataUrl;
  });
}

// ── message handler (used by popup) ─────────────────────────────────────────

browser.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  switch (msg.type) {
    case 'getState':
      sendResponse({ running: captureState.running, log: captureState.log });
      break;
    case 'start':
      startCapture();
      sendResponse({ ok: true });
      break;
    case 'stop':
      stopCapture();
      sendResponse({ ok: true });
      break;
    case 'restartAlarm':
      restartAlarm(msg.interval);
      sendResponse({ ok: true });
      break;
  }
  return true;
});

// ── image utilities ──────────────────────────────────────────────────────────

function dataUrlToBlob(dataUrl) {
  const [header, data] = dataUrl.split(',');
  const mime = header.match(/:(.*?);/)[1];
  const binary = atob(data);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return new Blob([bytes], { type: mime });
}