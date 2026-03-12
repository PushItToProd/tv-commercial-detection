from flask import Flask, request, jsonify, send_from_directory, Response
from datetime import datetime
from pathlib import Path
import json
import os
import shutil
import tempfile
import threading
import urllib.request
from classify import classify_image

app = Flask(__name__)

# In-memory state for the most recent /receive classification
_state = {
    "classification": None, # None | "ad" | "content" | "unknown"
    "paused": True,  # Whether the video is currently paused (based on last /receive request)
    "auto_switch": True,  # Whether to automatically apply matrix settings on classification change
}

SAVE_DIR = Path(os.environ.get("SAVE_DIR", "frames"))
SAVE_DIR.mkdir(parents=True, exist_ok=True)
LABELS_FILE = SAVE_DIR / "labels.json"

INCORRECT_DIR = Path(os.environ.get("INCORRECT_DIR", "incorrect_frames"))
INCORRECT_DIR.mkdir(parents=True, exist_ok=True)

_LAST_IMAGE_PATH = Path(tempfile.gettempdir()) / "tv_detector_last_frame.png"

CONFIG_FILE = Path(os.environ.get("CONFIG_FILE", "config.json"))


def load_config() -> dict:
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            return json.load(f)
    return {}


def apply_matrix_settings(classification: str) -> None:
    """Send HDMI matrix switch commands for the given classification in a background thread."""
    config = load_config()
    matrix_url = config.get("matrix_url", "http://localhost:5000")
    key = "ad_output_setting" if classification == "ad" else "race_output_setting"
    settings: dict = config.get(key, {})
    if not settings:
        return

    def _send():
        for output, input_num in settings.items():
            payload = json.dumps({"output": output, "input": int(input_num)}).encode()
            req = urllib.request.Request(
                f"{matrix_url}/set-output-input",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=5) as resp:
                    print(f"Matrix: output {output} → input {input_num}  ({resp.status})")
            except Exception as e:
                print(f"Matrix error (output {output} → input {input_num}): {e}")

    threading.Thread(target=_send, daemon=True).start()


def load_labels() -> dict:
    if LABELS_FILE.exists():
        with open(LABELS_FILE) as f:
            return json.load(f)
    return {}


def save_labels(labels: dict) -> None:
    with open(LABELS_FILE, "w") as f:
        json.dump(labels, f, indent=2)


@app.route("/save", methods=["POST"])
def save():
    if "image" not in request.files:
        return jsonify({"error": "No image field in request"}), 400

    image = request.files["image"]

    # Use the extension's timestamp if provided, otherwise use server time
    raw_ts = request.form.get("timestamp")
    try:
        dt = datetime.fromisoformat(raw_ts.replace("Z", "+00:00")) if raw_ts else datetime.now()
    except ValueError:
        dt = datetime.now()

    filename = dt.strftime("%Y-%m-%d_%H-%M-%S") + ".png"
    save_path = SAVE_DIR / filename
    image.save(save_path)

    print(f"Saved: {save_path}  |  page: {request.form.get('page_title', '?')}")
    return jsonify({"saved": filename}), 200


@app.route("/frames/<filename>")
def serve_frame(filename):
    # send_from_directory prevents path traversal
    return send_from_directory(SAVE_DIR.resolve(), filename)


@app.route("/classify", methods=["POST"])
def classify():
    data = request.get_json()
    if not data or "filename" not in data or "label" not in data:
        return jsonify({"error": "Missing filename or label"}), 400
    label = data["label"]
    if label not in ("ad", "content"):
        return jsonify({"error": "label must be 'ad' or 'content'"}), 400
    filename = data["filename"]
    # Guard against path traversal: filename must be a plain basename ending in .png
    if Path(filename).name != filename or not filename.endswith(".png"):
        return jsonify({"error": "Invalid filename"}), 400
    labels = load_labels()
    labels[filename] = label
    save_labels(labels)
    return jsonify({"classified": filename, "label": label}), 200


@app.route("/review")
def review():
    labels = load_labels()
    images = sorted(p.name for p in SAVE_DIR.glob("*.png"))
    image_data = json.dumps([{"filename": f, "label": labels.get(f)} for f in images])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Frame Review</title>
<style>
  body {{ font-family: sans-serif; background: #1a1a1a; color: #eee; margin: 0; padding: 1rem; }}
  h1 {{ margin-bottom: 0.5rem; }}
  #controls {{ margin-bottom: 1rem; display: flex; align-items: center; gap: 1.2rem; flex-wrap: wrap; }}
  #counter {{ opacity: 0.6; font-size: 0.9rem; }}
  .grid {{ display: flex; flex-wrap: wrap; gap: 1rem; }}
  .card {{
    background: #2a2a2a; border-radius: 8px; overflow: hidden;
    width: 260px; display: flex; flex-direction: column; border: 2px solid transparent;
  }}
  .card.ad      {{ border-color: #e05; }}
  .card.content {{ border-color: #0a5; }}
  .card img {{ width: 100%; display: block; cursor: zoom-in; }}
  .card .info {{ padding: 0.4rem 0.5rem; font-size: 0.72rem; opacity: 0.6; word-break: break-all; }}
  .badge {{
    display: inline-block; margin: 0 0.5rem 0.5rem;
    padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.75rem; font-weight: bold;
  }}
  .badge.ad      {{ background: #e05; color: #fff; }}
  .badge.content {{ background: #0a5; color: #fff; }}
  .badge.none    {{ background: #555; color: #ccc; }}
  .actions {{ display: flex; gap: 0.5rem; padding: 0.5rem; }}
  .btn {{
    flex: 1; padding: 0.4rem; border: none; border-radius: 4px;
    cursor: pointer; font-weight: bold; font-size: 0.85rem; transition: opacity .15s;
  }}
  .btn:hover {{ opacity: 0.75; }}
  .btn-ad      {{ background: #e05; color: #fff; }}
  .btn-content {{ background: #0a5; color: #fff; }}
  /* Lightbox */
  #lightbox {{
    display: none; position: fixed; inset: 0; background: rgba(0,0,0,.85);
    align-items: center; justify-content: center; z-index: 100;
  }}
  #lightbox.open {{ display: flex; }}
  #lightbox img {{ max-width: 95vw; max-height: 92vh; border-radius: 6px; }}
  #lightbox-close {{
    position: fixed; top: 1rem; right: 1.4rem; font-size: 2rem; cursor: pointer;
    color: #fff; line-height: 1; user-select: none;
  }}
</style>
</head>
<body>
<h1>Frame Review</h1>
<div id="controls">
  <label><input type="checkbox" id="hideClassified"> Hide already classified</label>
  <span id="counter"></span>
</div>
<div class="grid" id="grid"></div>

<div id="lightbox">
  <span id="lightbox-close" title="Close">&times;</span>
  <img id="lightbox-img" src="" alt="">
</div>

<script>
const IMAGES = {image_data};

function updateCounter() {{
  const cards = document.querySelectorAll('.card');
  const visible = [...cards].filter(c => c.style.display !== 'none').length;
  document.getElementById('counter').textContent = `${{visible}} / ${{cards.length}} shown`;
}}

function applyFilter() {{
  const hide = document.getElementById('hideClassified').checked;
  document.querySelectorAll('.card').forEach(card => {{
    const classified = card.dataset.label && card.dataset.label !== 'null';
    card.style.display = (hide && classified) ? 'none' : '';
  }});
  updateCounter();
}}

function classifyImage(filename, label, card) {{
  fetch('/classify', {{
    method: 'POST',
    headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify({{filename, label}})
  }})
  .then(r => r.json())
  .then(data => {{
    if (data.error) {{ alert(data.error); return; }}
    card.dataset.label = label;
    card.className = 'card ' + label;
    const badge = card.querySelector('.badge');
    badge.className = 'badge ' + label;
    badge.textContent = label.toUpperCase();
    applyFilter();
  }});
}}

function openLightbox(src) {{
  document.getElementById('lightbox-img').src = src;
  document.getElementById('lightbox').classList.add('open');
}}

document.getElementById('lightbox').addEventListener('click', e => {{
  if (e.target === e.currentTarget || e.target.id === 'lightbox-close')
    e.currentTarget.classList.remove('open');
}});

function buildGrid() {{
  const grid = document.getElementById('grid');
  IMAGES.forEach(({{filename, label}}) => {{
    const card = document.createElement('div');
    card.className = 'card' + (label ? ' ' + label : '');
    card.dataset.label = label;

    const imgSrc = '/frames/' + encodeURIComponent(filename);

    const img = document.createElement('img');
    img.src = imgSrc;
    img.loading = 'lazy';
    img.title = 'Click to enlarge';
    img.addEventListener('click', () => openLightbox(imgSrc));

    const info = document.createElement('div');
    info.className = 'info';
    info.textContent = filename;

    const badge = document.createElement('span');
    badge.className = 'badge ' + (label || 'none');
    badge.textContent = label ? label.toUpperCase() : 'UNCLASSIFIED';

    const actions = document.createElement('div');
    actions.className = 'actions';
    ['ad', 'content'].forEach(lbl => {{
      const btn = document.createElement('button');
      btn.className = 'btn btn-' + lbl;
      btn.textContent = lbl.charAt(0).toUpperCase() + lbl.slice(1);
      btn.addEventListener('click', () => classifyImage(filename, lbl, card));
      actions.appendChild(btn);
    }});

    card.append(img, info, badge, actions);
    grid.appendChild(card);
  }});
  updateCounter();
}}

document.getElementById('hideClassified').addEventListener('change', applyFilter);
buildGrid();
</script>
</body>
</html>"""

    return Response(html, mimetype="text/html")


@app.route("/receive", methods=["POST"])
def receive():
    is_paused = request.form.get("is_paused", "").lower() in ("true", "1", "yes")
    _state["paused"] = is_paused

    if "image" not in request.files:
        if is_paused:
            print(f"Paused (no image)  |  page: {request.form.get('page_title', '?')}")
            return jsonify({"classification": _state["classification"], "paused": True}), 200
        return jsonify({"error": "No image field in request"}), 400

    image = request.files["image"]

    # Write to a temp file so classify_image (which expects a path) can read it
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        image.save(tmp.name)
        tmp_path = tmp.name

    try:
        result = classify_image(tmp_path)
        shutil.copy2(tmp_path, _LAST_IMAGE_PATH)
    finally:
        os.unlink(tmp_path)

    previous = _state["classification"]
    _state["classification"] = result
    print(f"Received image → classified as: {result}  |  page: {request.form.get('page_title', '?')}")
    if result != previous and result in ("ad", "content") and _state["auto_switch"]:
        apply_matrix_settings(result)
    return jsonify({"classification": result, "paused": is_paused}), 200


@app.route("/report_wrong", methods=["POST"])
def report_wrong():
    data = request.get_json()
    if not data or "correct_label" not in data:
        return jsonify({"error": "Missing correct_label"}), 400
    correct_label = data["correct_label"]
    if correct_label not in ("ad", "content"):
        return jsonify({"error": "correct_label must be 'ad' or 'content'"}), 400
    if not _LAST_IMAGE_PATH.exists():
        return jsonify({"error": "No image available"}), 400

    filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".png"
    dest = INCORRECT_DIR / filename
    shutil.copy2(_LAST_IMAGE_PATH, dest)

    labels_file = INCORRECT_DIR / "labels.json"
    labels = {}
    if labels_file.exists():
        with open(labels_file) as f:
            labels = json.load(f)
    labels[filename] = {"correct_label": correct_label, "classified_as": _state["classification"]}
    with open(labels_file, "w") as f:
        json.dump(labels, f, indent=2)

    print(f"Correction saved: {dest}  |  classified as: {_state['classification']}, correct: {correct_label}")
    return jsonify({"saved": filename, "correct_label": correct_label}), 200


@app.route("/auto_switch", methods=["POST"])
def set_auto_switch():
    data = request.get_json()
    if not data or "enabled" not in data:
        return jsonify({"error": "Missing enabled"}), 400
    _state["auto_switch"] = bool(data["enabled"])
    return jsonify({"auto_switch": _state["auto_switch"]}), 200


@app.route("/trigger_matrix", methods=["POST"])
def trigger_matrix():
    data = request.get_json()
    if not data or "classification" not in data:
        return jsonify({"error": "Missing classification"}), 400
    classification = data["classification"]
    if classification not in ("ad", "content"):
        return jsonify({"error": "classification must be 'ad' or 'content'"}), 400
    apply_matrix_settings(classification)
    return jsonify({"triggered": classification}), 200


@app.route("/is_ad/status")
def is_ad_status():
    config = load_config()
    return jsonify({
        "classification": _state["classification"],
        "paused": _state["paused"],
        "auto_switch": _state["auto_switch"],
        "ad_input_a": config.get("ad_output_setting", {}).get("A", "?"),
        "race_input_a": config.get("race_output_setting", {}).get("A", "?"),
    })


@app.route("/is_ad")
def is_ad():
    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ad Detector</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    height: 100vh; font-family: sans-serif;
    background: #333; transition: background 0.4s;
  }
  #label {
    font-size: 20vw; font-weight: bold; color: #fff;
    text-shadow: 0 4px 24px rgba(0,0,0,0.5);
    letter-spacing: 0.05em;
  }
  #paused-indicator {
    font-size: 4vw; font-weight: bold; color: rgba(255,255,255,0.65);
    letter-spacing: 0.1em; margin-top: 0.5rem; min-height: 1.2em;
  }
  #buttons {
    position: fixed; bottom: 2rem;
    display: flex; gap: 1rem;
  }
  .btn-correction {
    padding: 1rem 2.4rem; font-size: 1.6rem; font-weight: bold;
    border: none; border-radius: 12px; cursor: pointer;
    background: rgba(255,255,255,0.2); color: #fff;
    transition: background 0.2s;
  }
  .btn-correction:hover { background: rgba(255,255,255,0.38); }
  #top-controls {
    position: fixed; top: 0; left: 0; right: 0;
    display: flex; align-items: center; gap: 1.5rem; padding: 1rem 1.6rem;
    background: rgba(0,0,0,0.4); z-index: 10; flex-wrap: wrap;
  }
  .ctrl-label { font-size: 1.3rem; color: rgba(255,255,255,0.8); white-space: nowrap; }
  .toggle-wrap { display: flex; align-items: center; gap: 0.75rem; }
  .toggle { position: relative; display: inline-block; width: 64px; height: 36px; }
  .toggle input { opacity: 0; width: 0; height: 0; }
  .toggle-slider {
    position: absolute; inset: 0; cursor: pointer;
    background: #555; border-radius: 36px; transition: background 0.2s;
  }
  .toggle-slider:before {
    content: ''; position: absolute;
    width: 26px; height: 26px; left: 5px; top: 5px;
    background: #fff; border-radius: 50%; transition: transform 0.2s;
  }
  .toggle input:checked + .toggle-slider { background: #0c6; }
  .toggle input:checked + .toggle-slider:before { transform: translateX(28px); }
  .btn-matrix {
    padding: 0.75rem 1.6rem; font-size: 1.3rem; font-weight: bold;
    border: 2px solid rgba(255,255,255,0.3); border-radius: 10px; cursor: pointer;
    background: rgba(255,255,255,0.12); color: #fff; transition: background 0.15s;
  }
  .btn-matrix:hover { background: rgba(255,255,255,0.28); }
</style>
</head>
<body>
<div id="top-controls">
  <div class="toggle-wrap">
    <label class="toggle"><input type="checkbox" id="auto-switch-toggle" checked><span class="toggle-slider"></span></label>
    <span class="ctrl-label">Auto-switch</span>
  </div>
  <button class="btn-matrix" id="btn-ad-input">Input ?</button>
  <button class="btn-matrix" id="btn-race-input">Input ?</button>
</div>
<div id="label">...</div>
<div id="paused-indicator"></div>
<div id="buttons"></div>
<script>
function reportWrong(correctLabel) {
  fetch('/report_wrong', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({correct_label: correctLabel})
  })
  .then(r => r.json())
  .then(data => { if (data.error) alert(data.error); })
  .catch(() => {});
}

function updateButtons(c) {
  const div = document.getElementById('buttons');
  div.innerHTML = '';
  if (c === 'ad' || c === 'content') {
    const btn = document.createElement('button');
    btn.className = 'btn-correction';
    btn.textContent = 'Wrong!';
    btn.onclick = () => reportWrong(c === 'ad' ? 'content' : 'ad');
    div.appendChild(btn);
  } else if (c !== null && c !== undefined) {
    [['ad', 'Ad'], ['content', 'Racing']].forEach(([lbl, display]) => {
      const btn = document.createElement('button');
      btn.className = 'btn-correction';
      btn.textContent = display;
      btn.onclick = () => reportWrong(lbl);
      div.appendChild(btn);
    });
  }
}

function triggerMatrix(classification) {
  fetch('/trigger_matrix', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({classification})
  }).catch(() => {});
}

document.getElementById('auto-switch-toggle').addEventListener('change', function() {
  fetch('/auto_switch', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({enabled: this.checked})
  }).catch(() => {});
});

document.getElementById('btn-ad-input').addEventListener('click', () => triggerMatrix('ad'));
document.getElementById('btn-race-input').addEventListener('click', () => triggerMatrix('content'));

function update() {
  fetch('/is_ad/status')
    .then(r => r.json())
    .then(data => {
      const c = data.classification;
      const paused = data.paused;
      const el = document.getElementById('label');
      const pausedEl = document.getElementById('paused-indicator');
      if (c === 'ad') {
        document.body.style.background = '#cc0033';
        el.textContent = 'AD';
      } else if (c === 'content') {
        document.body.style.background = '#00882b';
        el.textContent = 'RACING';
      } else {
        document.body.style.background = '#333';
        el.textContent = c === null ? '...' : '?';
      }
      pausedEl.textContent = paused ? '⏸ PAUSED' : '';
      updateButtons(c);
      document.getElementById('auto-switch-toggle').checked = data.auto_switch;
      if (data.ad_input_a !== undefined)
        document.getElementById('btn-ad-input').textContent = 'Input ' + data.ad_input_a;
      if (data.race_input_a !== undefined)
        document.getElementById('btn-race-input').textContent = 'Input ' + data.race_input_a;
    })
    .catch(() => {});
}

update();
setInterval(update, 2000);
</script>
</body>
</html>"""
    return Response(html, mimetype="text/html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=11434, debug=True)
