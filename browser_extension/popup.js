const DEFAULT_INTERVAL = 4; // seconds
const DEFAULT_URL = 'http://localhost:11434/receive';


const $interval = document.getElementById('interval');
const $list = document.getElementById('endpoints-list');
const $btnAdd = document.getElementById('btn-add');
const $btnSave = document.getElementById('btn-save');
const $btnToggle = document.getElementById('btn-toggle');
const $dot = document.getElementById('status-dot');
const $log = document.getElementById('log');

// ── helpers ────────────────────────────────────────────────────────────────

function addEndpointRow(url = '') {
  const row = document.createElement('div');
  row.className = 'endpoint-row d-flex gap-1 align-items-center';
  row.innerHTML = `
    <input type="text" class="form-control form-control-sm" placeholder="${DEFAULT_URL}" value="${url}">
    <button class="btn-icon btn btn-sm btn-outline-danger p-0" title="Remove" style="width:32px;height:32px;">×</button>
  `;
  row.querySelector('.btn-icon').addEventListener('click', () => row.remove());
  $list.appendChild(row);
  row.querySelector('input').focus();
}

function getEndpointInputs() {
  return Array.from($list.querySelectorAll('input[type="text"]'))
    .map(i => i.value.trim())
    .filter(Boolean);
}

function isValidUrl(s) {
  try { return /^https?:\/\/.+/.test(new URL(s).href); } catch { return false; }
}

function syncUI(running) {
  $btnToggle.textContent = running ? 'Stop' : 'Start';
  $btnToggle.classList.toggle('running', running);
  $dot.classList.toggle('active', running);
}

function appendLog(msg, type = '') {
  const line = document.createElement('div');
  line.className = `entry ${type}`;
  const t = new Date().toLocaleTimeString([], { hour12: false });
  line.textContent = `[${t}] ${msg}`;
  $log.appendChild(line);
  $log.scrollTop = $log.scrollHeight;
  // keep last 60 lines
  while ($log.children.length > 61) $log.removeChild($log.children[1]);
}

// ── init ───────────────────────────────────────────────────────────────────

const background = {
  async getCaptureState() {
    return browser.runtime.sendMessage({ type: 'getCaptureState' });
  },
  async startCapture() {
    return browser.runtime.sendMessage({ type: 'startCapture' });
  },
  async stopCapture() {
    return browser.runtime.sendMessage({ type: 'stopCapture' });
  },
  async restartAlarm(interval) {
    return browser.runtime.sendMessage({ type: 'restartCaptureAlarm', interval });
  }
}

async function init() {
  const { config = {} } = await browser.storage.local.get('config');
  $interval.value = config.interval ?? DEFAULT_INTERVAL;
  ($list.querySelectorAll('.endpoint-row') || []).forEach(r => r.remove());
  (config.endpoints ?? [DEFAULT_URL]).forEach(addEndpointRow);

  const state = await background.getCaptureState();
  syncUI(state.running);

  // replay background log into popup
  for (const entry of state.log ?? []) {
    appendLog(entry.msg, entry.type);
  }
}

init();

// ── events ─────────────────────────────────────────────────────────────────

$btnAdd.addEventListener('click', () => addEndpointRow());

$btnSave.addEventListener('click', async () => {
  const interval = Math.max(1, parseInt($interval.value) || DEFAULT_INTERVAL);
  const endpoints = getEndpointInputs();

  // validate
  let valid = true;
  $list.querySelectorAll('input[type="text"]').forEach(inp => {
    const ok = isValidUrl(inp.value.trim()) || inp.value.trim() === '';
    inp.classList.toggle('is-invalid', inp.value.trim() !== '' && !ok);
    if (!ok && inp.value.trim() !== '') valid = false;
  });
  if (!valid) { appendLog('Invalid URL — fix highlighted fields.', 'err'); return; }

  await browser.storage.local.set({ config: { interval, endpoints } });
  appendLog('Config saved.', 'ok');

  // if running, restart the alarm with new interval
  const state = await background.getCaptureState();
  if (state.running) {
    await background.restartAlarm(interval);
    appendLog(`Interval updated → ${interval}s`, '');
  }
});

$btnToggle.addEventListener('click', async () => {
  const state = await background.getCaptureState();
  if (state.running) {
    await background.stopCapture();
    syncUI(false);
    appendLog('Capture stopped.', '');
  } else {
    const { config = {} } = await browser.storage.local.get('config');
    const endpoints = config.endpoints ?? [];
    if (endpoints.length === 0) {
      appendLog('Add at least one endpoint first.', 'err');
      return;
    }
    await background.startCapture();
    syncUI(true);
    appendLog(`Capture started (every ${config.interval ?? DEFAULT_INTERVAL}s → ${endpoints.length} endpoint(s)).`, 'ok');
  }
});

// listen for log messages pushed from the background page
window.addEventListener('capture-log', e => {
  appendLog(e.detail.msg, e.detail.type);
});