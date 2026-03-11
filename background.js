// ── shared state (accessible from popup via getBackgroundPage()) ────────────

const captureState = {
  running: false,
  log: []   // ring buffer of { msg, type } — replayed into popup on open
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

function startCapture() {
  captureState.running = true;
  browser.alarms.create(ALARM_NAME, { periodInMinutes: getIntervalMinutes() });
  // fire immediately so the user sees it working right away
  doCapture();
}

function stopCapture() {
  captureState.running = false;
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

  // get the active tab
  let tabs;
  try {
    tabs = await browser.tabs.query({ active: true, currentWindow: true });
  } catch (e) {
    bgLog('Could not query tabs: ' + e.message, 'err');
    return;
  }

  const tab = tabs[0];
  if (!tab) { bgLog('No active tab.', 'err'); return; }

  try {
    // 1. get video rect from page
    const results = await browser.tabs.executeScript(tab.id, { file: 'content_scripts/get_video_bounds.js' });
    const rect = results[0];

    // 2. screenshot
    const dataUrl = await browser.tabs.captureVisibleTab(tab.windowId, { format: 'png' });

    // 3. crop if we found a video
    const finalUrl = rect ? await cropImage(dataUrl, rect) : dataUrl;
    if (!rect) bgLog('No video found — sending full screenshot.', '');

    // 4. POST to each endpoint concurrently
    const blob = dataUrlToBlob(finalUrl);
    const timestamp = new Date().toISOString();

    const posts = endpoints.map(async url => {
      const form = new FormData();
      form.append('image', blob, `frame_${timestamp}.png`);
      form.append('timestamp', timestamp);
      form.append('page_title', tab.title ?? '');
      form.append('page_url', tab.url ?? '');

      try {
        const res = await fetch(url, { method: 'POST', body: form });
        if (res.ok) {
          bgLog(`POST ${url} → ${res.status}`, 'ok');
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

function dataUrlToBlob(dataUrl) {
  const [header, data] = dataUrl.split(',');
  const mime = header.match(/:(.*?);/)[1];
  const binary = atob(data);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return new Blob([bytes], { type: mime });
}