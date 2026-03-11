const bg = browser.extension.getBackgroundPage();

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
  row.className = 'endpoint-row';
  row.innerHTML = `
    <input type="text" placeholder="http://localhost:11434/save" value="${url}">
    <button class="btn-icon" title="Remove">×</button>
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

async function init() {
  const { config = {} } = await browser.storage.local.get('config');
  $interval.value = config.interval ?? 10;
  ($list.querySelectorAll('.endpoint-row') || []).forEach(r => r.remove());
  (config.endpoints ?? ['http://localhost:11434/save']).forEach(addEndpointRow);

  const running = bg?.captureState?.running;
  syncUI(running);

  // replay background log into popup
  for (const entry of bg?.captureState?.log ?? []) {
    appendLog(entry.msg, entry.type);
  }
}

init();

// ── events ─────────────────────────────────────────────────────────────────

$btnAdd.addEventListener('click', () => addEndpointRow());

$btnSave.addEventListener('click', async () => {
  const interval = Math.max(1, parseInt($interval.value) || 10);
  const endpoints = getEndpointInputs();

  // validate
  let valid = true;
  $list.querySelectorAll('input[type="text"]').forEach(inp => {
    const ok = isValidUrl(inp.value.trim()) || inp.value.trim() === '';
    inp.classList.toggle('error', inp.value.trim() !== '' && !ok);
    if (!ok && inp.value.trim() !== '') valid = false;
  });
  if (!valid) { appendLog('Invalid URL — fix highlighted fields.', 'err'); return; }

  await browser.storage.local.set({ config: { interval, endpoints } });
  appendLog('Config saved.', 'ok');

  // if running, restart the alarm with new interval
  if (bg?.captureState?.running) {
    bg.restartAlarm(interval);
    appendLog(`Interval updated → ${interval}s`, '');
  }
});

$btnToggle.addEventListener('click', async () => {
  if (bg?.captureState?.running) {
    bg.stopCapture();
    syncUI(false);
    appendLog('Capture stopped.', '');
  } else {
    const { config = {} } = await browser.storage.local.get('config');
    const endpoints = config.endpoints ?? [];
    if (endpoints.length === 0) {
      appendLog('Add at least one endpoint first.', 'err');
      return;
    }
    bg.startCapture();
    syncUI(true);
    appendLog(`Capture started (every ${config.interval ?? 10}s → ${endpoints.length} endpoint(s)).`, 'ok');
  }
});

// listen for log messages pushed from the background page
window.addEventListener('capture-log', e => {
  appendLog(e.detail.msg, e.detail.type);
});