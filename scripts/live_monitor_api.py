from pathlib import Path
import csv
import json
import sys

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import OUTPUT_BASE_DIR, ROOM

OUTPUT_ROOT = Path(OUTPUT_BASE_DIR)
LIVE_ROOT = OUTPUT_ROOT / "live"

app = FastAPI(title="Surgery Live Monitor")


def _run_dirs():
    if not LIVE_ROOT.exists():
        return []
    return sorted(
        [path for path in LIVE_ROOT.glob("run_*") if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def _resolve_live_dir(run_id: str | None = None) -> Path | None:
    if run_id:
        if Path(run_id).name != run_id:
            raise HTTPException(status_code=400, detail="invalid run_id")
        path = LIVE_ROOT / run_id
        if not path.is_dir():
            raise HTTPException(status_code=404, detail=f"run_id not found: {run_id}")
        return path

    runs = _run_dirs()
    if runs:
        return runs[0]
    return LIVE_ROOT if LIVE_ROOT.is_dir() else None


def _run_payload(path: Path) -> dict:
    status_path = path / "status.json"
    updated_at = None
    dataset = None
    if status_path.exists():
        try:
            payload = json.loads(status_path.read_text(encoding="utf-8"))
            updated_at = payload.get("updated_at")
            dataset = payload.get("dataset")
        except Exception:
            pass
    return {
        "run_id": path.name,
        "path": str(path),
        "updated_at": updated_at,
        "dataset": dataset,
    }


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse("""
<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Surgery Live Monitor</title>
  <style>
    :root { color-scheme: dark; }
    body { margin: 0; font-family: system-ui, -apple-system, Segoe UI, sans-serif; background: #191b1f; color: #eef1f4; }
    header { display: flex; align-items: center; justify-content: space-between; padding: 12px 16px; border-bottom: 1px solid #343941; background: #202329; }
    h1 { margin: 0; font-size: 18px; font-weight: 650; }
    main { display: grid; grid-template-columns: 1fr 360px; gap: 14px; padding: 14px; }
    .images { display: grid; grid-template-columns: repeat(2, minmax(260px, 1fr)); gap: 14px; align-items: start; }
    figure { margin: 0; background: #111317; border: 1px solid #343941; border-radius: 6px; overflow: hidden; }
    figcaption { padding: 8px 10px; color: #c9d1d9; font-size: 13px; border-bottom: 1px solid #343941; }
    img { width: 100%; display: block; background: #08090b; }
    aside { background: #111317; border: 1px solid #343941; border-radius: 6px; padding: 12px; }
    .state { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
    .metric { padding: 8px; border: 1px solid #30363d; border-radius: 5px; background: #181b20; }
    .metric b { display: block; color: #9fb3c8; font-size: 12px; font-weight: 550; }
    .metric span { display: block; margin-top: 4px; font-size: 18px; }
    table { width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 13px; }
    caption { text-align: left; color: #9fb3c8; padding: 6px 0; font-weight: 650; }
    th, td { border-bottom: 1px solid #30363d; padding: 6px 4px; text-align: left; }
    th { color: #9fb3c8; font-weight: 600; }
    pre { overflow: auto; max-height: 280px; padding: 10px; background: #08090b; border: 1px solid #30363d; border-radius: 5px; color: #d7dde5; }
    .ok { color: #60d394; }
    .bad { color: #ff7b72; }
    select { background: #111317; color: #eef1f4; border: 1px solid #343941; border-radius: 5px; padding: 5px 8px; }
    @media (max-width: 1050px) { main { grid-template-columns: 1fr; } .images { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <header>
    <h1>Surgery Live Monitor</h1>
    <div><select id="runSelect"><option value="">latest</option></select> <span id="updated">waiting...</span></div>
  </header>
  <main>
    <section class="images">
      <figure><figcaption>Door full frame</figcaption><img id="doorFrame" /></figure>
      <figure><figcaption>Room frame</figcaption><img id="roomFrame" /></figure>
      <figure><figcaption>Door crop + ROI</figcaption><img id="doorRoi" /></figure>
      <figure><figcaption>Door crop</figcaption><img id="doorCrop" /></figure>
    </section>
    <aside>
      <div class="state">
        <div class="metric"><b>Door</b><span id="doorOpen">-</span></div>
        <div class="metric"><b>Surgery Date</b><span id="surgeryDate">-</span></div>
        <div class="metric"><b>Score</b><span id="score">-</span></div>
        <div class="metric"><b>Ratio</b><span id="ratio">-</span></div>
        <div class="metric"><b>AI Status</b><span id="aiStatus">-</span></div>
        <div class="metric"><b>Voted</b><span id="voted">-</span></div>
        <div class="metric"><b>Real Time</b><span id="realTime">-</span></div>
        <div class="metric"><b>Video Time</b><span id="videoTime">-</span></div>
      </div>
      <table>
        <caption>Realtime Events</caption>
        <thead><tr><th>Date</th><th>No</th><th>Type</th><th>Time</th></tr></thead>
        <tbody id="eventsBody"><tr><td colspan="4">waiting...</td></tr></tbody>
      </table>
      <pre id="json">{}</pre>
    </aside>
  </main>
<script>
let selectedRun = new URLSearchParams(window.location.search).get("run_id") || "";

function query() {
  return selectedRun ? `run_id=${encodeURIComponent(selectedRun)}&` : "";
}

async function refreshRuns() {
  try {
    const res = await fetch(`/api/runs?t=${Date.now()}`);
    const data = await res.json();
    const select = document.getElementById('runSelect');
    const current = selectedRun || select.value || "";
    select.innerHTML = '<option value="">latest</option>' + (data.runs || []).map(run => {
      const label = `${run.run_id}${run.dataset ? ' / ' + run.dataset : ''}${run.updated_at ? ' / ' + run.updated_at : ''}`;
      return `<option value="${run.run_id}">${label}</option>`;
    }).join('');
    select.value = current;
  } catch (err) {}
}

async function tick() {
  const t = Date.now();
  const q = query();
  document.getElementById('doorFrame').src = `/frame/door?${q}t=${t}`;
  document.getElementById('doorCrop').src = `/frame/door_crop?${q}t=${t}`;
  document.getElementById('doorRoi').src = `/frame/door_roi?${q}t=${t}`;
  document.getElementById('roomFrame').src = `/frame/room?${q}t=${t}`;
  try {
    const res = await fetch(`/api/status?${q}t=${t}`);
    const data = await res.json();
    const eventRes = await fetch(`/api/events?${q}t=${t}`);
    const eventData = await eventRes.json();
    document.getElementById('json').textContent = JSON.stringify(data, null, 2);
    document.getElementById('updated').textContent = data.run_id ? `${data.run_id} / ${data.updated_at || 'waiting...'}` : (data.updated_at || 'waiting...');
    setText('doorOpen', data.door_open ? 'OPEN' : 'CLOSE', data.door_open);
    setText('surgeryDate', data.surgery_date ?? '-');
    setText('score', data.door_score ?? '-');
    setText('ratio', data.door_ratio ?? '-');
    setText('aiStatus', data.status ?? '-');
    setText('voted', data.voted_status ?? '-');
    setText('realTime', data.real_time ?? '-');
    setText('videoTime', data.video_time ?? '-');
    renderEvents(eventData.events || []);
  } catch (err) {
    document.getElementById('updated').textContent = 'not ready';
  }
}
function setText(id, value, state) {
  const el = document.getElementById(id);
  el.textContent = value;
  el.className = state === true ? 'ok' : state === false ? 'bad' : '';
}
function renderEvents(events) {
  const body = document.getElementById('eventsBody');
  if (!events.length) {
    body.innerHTML = '<tr><td colspan="4">no events yet</td></tr>';
    return;
  }
  body.innerHTML = events.map(e => `
    <tr>
      <td>${e.Surgery_Date ?? ''}</td>
      <td>${e.Surgery_No ?? ''}</td>
      <td>${e.Type ?? ''}</td>
      <td>${e.Real_Time ?? ''}</td>
    </tr>
  `).join('');
}
document.getElementById('runSelect').addEventListener('change', event => {
  selectedRun = event.target.value;
  const url = new URL(window.location.href);
  if (selectedRun) url.searchParams.set('run_id', selectedRun);
  else url.searchParams.delete('run_id');
  window.history.replaceState({}, '', url);
  tick();
});
setInterval(refreshRuns, 3000);
setInterval(tick, 500);
refreshRuns();
tick();
</script>
</body>
</html>
""")


@app.get("/api/runs")
def runs():
    return JSONResponse({"runs": [_run_payload(path) for path in _run_dirs()]})


@app.get("/api/status")
def status(run_id: str | None = None):
    live_dir = _resolve_live_dir(run_id)
    if live_dir is None:
        return JSONResponse({"ready": False, "message": "main_parallel.py has not written live status yet"})
    path = live_dir / "status.json"
    if not path.exists():
        return JSONResponse({"ready": False, "run_id": live_dir.name, "message": "main_parallel.py has not written live status yet"})
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["run_id"] = live_dir.name
    return JSONResponse(payload)


def _file_response(filename: str, run_id: str | None = None):
    live_dir = _resolve_live_dir(run_id)
    if live_dir is None:
        raise HTTPException(status_code=404, detail="live directory not found")
    path = live_dir / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{filename} not found")
    return FileResponse(path, media_type="image/jpeg")


@app.get("/api/events")
def events(run_id: str | None = None):
    live_dir = _resolve_live_dir(run_id)
    if live_dir is None:
        return JSONResponse({"events": [], "path": None})

    status_path = live_dir / "status.json"
    if not status_path.exists():
        return JSONResponse({"events": [], "path": None})

    status_payload = json.loads(status_path.read_text(encoding="utf-8"))
    dataset = status_payload.get("dataset")
    if not dataset:
        return JSONResponse({"events": [], "path": None})

    result_dir = OUTPUT_ROOT / dataset / "result" / ROOM
    paths = sorted(result_dir.glob("Realtime_Events_Surgery_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not paths:
        return JSONResponse({"events": [], "path": str(result_dir)})

    latest = paths[0]
    rows = []
    with latest.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "Surgery_Date": row.get("Surgery_Date", ""),
                "Surgery_No": row.get("Surgery_No", ""),
                "Type": row.get("Type", ""),
                "Real_Time": row.get("Real_Time", ""),
            })
    return JSONResponse({"events": rows[-8:], "path": str(latest)})


@app.get("/frame/door")
def door_frame(run_id: str | None = None):
    return _file_response("door_frame.jpg", run_id)


@app.get("/frame/door_crop")
def door_crop(run_id: str | None = None):
    return _file_response("door_crop.jpg", run_id)


@app.get("/frame/door_roi")
def door_roi(run_id: str | None = None):
    return _file_response("door_crop_roi.jpg", run_id)


@app.get("/frame/room")
def room_frame(run_id: str | None = None):
    return _file_response("room_frame.jpg", run_id)
