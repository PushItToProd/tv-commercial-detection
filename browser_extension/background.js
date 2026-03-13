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

async function getTabVideoInfo(tab) {
  const results = await browser.tabs.executeScript(tab.id, { file: 'content_scripts/get_video_bounds.js' });
  const info = results[0];
  return info;
}

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
      resolve(canvas.toDataURL('image/jpeg', 0.6));
    };
    img.onerror = reject;
    img.src = dataUrl;
  });
}

async function screenshotTabAsBlob(tab, videoInfo) {
  // 2. screenshot (PNG so the crop step has lossless input before JPEG encoding)
  const dataUrl = await browser.tabs.captureTab(tab.tabId, { format: 'png' });

  // 3. crop to the video rect
  const finalUrl = await cropImage(dataUrl, videoInfo);
  blob = dataUrlToBlob(finalUrl);
  return blob;
}

function buildFormData(tab, tabState, blob) {
  const {isPaused, isSeeking, timestamp} = tabState;

  const form = new FormData();
  if (blob) form.append('image', blob, `frame_${timestamp}.jpg`);
  form.append('is_paused', isPaused ? 'true' : 'false');
  form.append('is_seeking', isSeeking ? 'true' : 'false');
  form.append('timestamp', timestamp);
  form.append('page_title', tab.title ?? '');
  form.append('page_url', tab.url ?? '');
  return form;
}

// ── immediate video-state POST ─────────────────────────────────────────────

function videoStateUrl(endpointUrl) {
  const u = new URL(endpointUrl);
  const parts = u.pathname.split('/');
  parts[parts.length - 1] = 'video-state';
  u.pathname = parts.join('/');
  return u.toString();
}

async function postVideoState(isPaused, isSeeking) {
  // if (!captureState.running) return;

  let config = {};
  try {
    ({ config = {} } = await browser.storage.local.get('config'));
  } catch (e) {
    bgLog('Could not read config for state update: ' + e.message, 'err');
    return;
  }

  const endpoints = config.endpoints ?? [];
  if (endpoints.length === 0) return;

  let tab;
  try {
    tab = await browser.tabs.get(captureState.tabId);
  } catch (e) {
    return;
  }

  const form = new FormData();
  form.append('is_paused', isPaused ? 'true' : 'false');
  form.append('is_seeking', isSeeking ? 'true' : 'false');
  form.append('page_title', tab.title ?? '');
  form.append('page_url', tab.url ?? '');

  bgLog('video state change → ' + JSON.stringify({ isPaused, isSeeking }), 'debug');

  await Promise.all(endpoints.map(async url => {
    const stateUrl = videoStateUrl(url);
    try {
      const res = await fetch(stateUrl, { method: 'POST', body: form });
      const note = isPaused ? 'paused' : isSeeking ? 'seeking' : 'resumed';
      if (res.ok) {
        bgLog(`State → ${stateUrl} ${res.status} (${note})`, 'ok');
      } else {
        bgLog(`State → ${stateUrl} ${res.status} ${res.statusText}`, 'err');
      }
    } catch (e) {
      bgLog(`State POST ${stateUrl} failed: ${e.message}`, 'err');
    }
  }));
}

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
    const videoInfo = await getTabVideoInfo(tab);

    // skip this tick if there's no video at all
    if (!videoInfo) { bgLog('No video found — skipping.', ''); return; }

    const isPaused = !videoInfo.playing;
    const isSeeking = videoInfo.seeking || videoInfo.recentlySeeked;
    const timestamp = new Date().toISOString();

    let screenshotBlob = null;
    if (!isPaused && !isSeeking) {
      screenshotBlob = await screenshotTabAsBlob(tab, videoInfo);
    }

    const tabState = {isPaused, isSeeking, timestamp};
    const form = buildFormData(tab, tabState, screenshotBlob);

    // 4. POST to each endpoint concurrently
    const posts = endpoints.map(async url => {
      try {
        const res = await fetch(url, { method: 'POST', body: form });
        if (res.ok) {
          const note = isPaused ? ' (paused)' : isSeeking ? ' (seeking — skipped)' : '';
          bgLog(`POST ${url} → ${res.status}${note}`, 'ok');
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

// ── message handler (used by popup) ─────────────────────────────────────────

browser.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  switch (msg.type) {
    // popup actions
    case 'getCaptureState':
      sendResponse({ running: captureState.running, log: captureState.log });
      break;
    case 'startCapture':
      startCapture();
      sendResponse({ ok: true });
      break;
    case 'stopCapture':
      stopCapture();
      sendResponse({ ok: true });
      break;
    case 'restartCaptureAlarm':
      restartAlarm(msg.interval);
      sendResponse({ ok: true });
      break;
    case 'videoStateChange':
      postVideoState(msg.isPaused, msg.isSeeking);
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