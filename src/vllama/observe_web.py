"""Web UI for browsing observed API requests and responses."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

from vllama.observe import _record_to_dict, list_records, load_record


def create_observe_app(observe_dir: Path, model: str | None = None) -> FastAPI:
    app = FastAPI(title="vllama observe")

    @app.get("/api/records")
    async def get_records() -> JSONResponse:
        records = list_records(observe_dir, model)
        items = []
        for _path, record in records:
            items.append(
                {
                    "id": record.id,
                    "timestamp": record.timestamp,
                    "model": record.model,
                    "endpoint": record.endpoint,
                    "method": record.method,
                    "response_status": record.response_status,
                    "tokens_prompt": record.tokens_prompt,
                    "tokens_completion": record.tokens_completion,
                    "tokens_per_second": record.tokens_per_second,
                    "ttft_seconds": record.ttft_seconds,
                    "duration_seconds": record.duration_seconds,
                }
            )
        return JSONResponse(items)

    @app.get("/api/records/{record_id}")
    async def get_record(record_id: str) -> JSONResponse:
        records = list_records(observe_dir, model)
        for path, record in records:
            if record.id == record_id:
                try:
                    full = load_record(path)
                except Exception:
                    full = record
                return JSONResponse(_record_to_dict(full))
        return JSONResponse({"error": "not found"}, status_code=404)

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        filter_label = f" (model: {model})" if model else ""
        return _INDEX_HTML.replace("{{FILTER_LABEL}}", filter_label)

    return app


_INDEX_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>vllama observe</title>
<style>
  :root {
    --bg: #1e1e1e; --bg2: #252526; --bg3: #2d2d2d; --border: #3e3e3e;
    --text: #d4d4d4; --text2: #888; --accent: #569cd6; --accent2: #4ec9b0;
    --warn: #ce9178; --err: #f44747; --ok: #6a9955;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
         background: var(--bg); color: var(--text); display: flex;
         flex-direction: column; height: 100vh; }
  header { background: var(--bg2); padding: 10px 20px; border-bottom: 1px solid var(--border);
           display: flex; align-items: center; gap: 16px; flex-shrink: 0; }
  header h1 { font-size: 16px; font-weight: 600; color: var(--accent); }
  header .filter { font-size: 13px; color: var(--text2); }
  header button { background: var(--bg3); color: var(--text); border: 1px solid var(--border);
                  padding: 4px 12px; border-radius: 4px; cursor: pointer; font-size: 13px; }
  header button:hover { background: var(--border); }
  .main { display: flex; flex: 1; overflow: hidden; }
  .list-pane { width: 420px; min-width: 300px; border-right: 1px solid var(--border);
               overflow-y: auto; flex-shrink: 0; }
  .detail-pane { flex: 1; overflow-y: auto; padding: 16px 20px; }
  .record { padding: 8px 14px; cursor: pointer; border-bottom: 1px solid var(--border);
            transition: background 0.1s; }
  .record:hover { background: var(--bg3); }
  .record.selected { background: #264f78; }
  .record .top { display: flex; justify-content: space-between; align-items: center; }
  .record .method { font-weight: 600; font-size: 12px; color: var(--accent2); }
  .record .time { font-size: 12px; color: var(--text2); }
  .record .path { font-size: 13px; margin-top: 2px; color: var(--text); }
  .record .meta { font-size: 12px; color: var(--text2); margin-top: 2px;
                  display: flex; gap: 12px; }
  .record .status-ok { color: var(--ok); }
  .record .status-err { color: var(--err); }
  .detail-placeholder { color: var(--text2); font-style: italic;
                        padding: 40px; text-align: center; }
  .detail h2 { font-size: 15px; font-weight: 600; margin-bottom: 8px; color: var(--accent); }
  .detail .metrics { display: flex; flex-wrap: wrap; gap: 16px; margin-bottom: 16px;
                     padding: 10px 14px; background: var(--bg2); border-radius: 6px; }
  .detail .metric { font-size: 13px; }
  .detail .metric .label { color: var(--text2); font-size: 11px; }
  .detail .metric .value { color: var(--text); font-weight: 500; }
  .section { margin-bottom: 16px; }
  .section-header { font-size: 13px; font-weight: 600; color: var(--text2); cursor: pointer;
                    padding: 6px 0; user-select: none;
                    display: flex; align-items: center; gap: 6px; }
  .section-header:hover { color: var(--text); }
  .section-header .arrow { font-size: 10px; transition: transform 0.15s; }
  .section-header .arrow.open { transform: rotate(90deg); }
  .section-body { background: var(--bg2); border-radius: 4px; overflow: hidden; }
  .section-body pre { padding: 12px 14px; font-family: 'Fira Code', 'Cascadia Code', monospace;
                      font-size: 12px; line-height: 1.5; white-space: pre-wrap;
                      word-break: break-word; overflow-x: auto; margin: 0; }
  .empty { color: var(--text2); font-style: italic; padding: 12px 14px; font-size: 13px; }
  /* Syntax highlighting */
  .hl-key { color: #9cdcfe; }
  .hl-str { color: #ce9178; }
  .hl-num { color: #b5cea8; }
  .hl-bool { color: #569cd6; }
  .hl-null { color: #569cd6; }
  .hl-brace { color: #d4d4d4; }
  .hl-hdr-key { color: #9cdcfe; }
  .hl-hdr-val { color: #ce9178; }
</style>
</head>
<body>
<header>
  <h1>vllama observe</h1>
  <span class="filter">{{FILTER_LABEL}}</span>
  <button onclick="loadRecords()">Refresh</button>
</header>
<div class="main">
  <div class="list-pane" id="list"></div>
  <div class="detail-pane" id="detail">
    <div class="detail-placeholder">Select a request from the list to inspect.</div>
  </div>
</div>
<script>
let records = [];
let selectedId = null;

async function loadRecords() {
  const resp = await fetch('/api/records');
  records = await resp.json();
  renderList();
}

function renderList() {
  const el = document.getElementById('list');
  el.innerHTML = records.map(r => {
    const time = r.timestamp.substring(11, 19);
    const model = r.model || '?';
    const statusCls = r.response_status < 400 ? 'status-ok' : 'status-err';
    const tokens = (r.tokens_prompt || r.tokens_completion)
      ? `${r.tokens_prompt}+${r.tokens_completion}`
      : '-';
    const speed = r.tokens_per_second ? r.tokens_per_second.toFixed(1) + ' tok/s' : '';
    const sel = r.id === selectedId ? ' selected' : '';
    return `<div class="record${sel}" onclick="selectRecord('${r.id}')">
      <div class="top">
        <span class="method">${r.method}</span>
        <span class="time">${time}</span>
      </div>
      <div class="path">/${r.endpoint}</div>
      <div class="meta">
        <span>${model}</span>
        <span class="${statusCls}">${r.response_status}</span>
        <span>${tokens}</span>
        <span>${speed}</span>
      </div>
    </div>`;
  }).join('');
}

async function selectRecord(id) {
  selectedId = id;
  renderList();
  const resp = await fetch('/api/records/' + id);
  const r = await resp.json();
  renderDetail(r);
}

function renderDetail(r) {
  const el = document.getElementById('detail');
  const metrics = [];
  metrics.push(mk('Status', r.response_status));
  metrics.push(mk('Duration', r.duration_seconds.toFixed(3) + 's'));
  if (r.ttft_seconds != null) metrics.push(mk('TTFT', r.ttft_seconds.toFixed(3) + 's'));
  if (r.tokens_prompt || r.tokens_completion)
    metrics.push(mk('Tokens', `${r.tokens_prompt} + ${r.tokens_completion}`));
  if (r.tokens_per_second != null)
    metrics.push(mk('Speed', r.tokens_per_second.toFixed(1) + ' tok/s'));

  el.innerHTML = `<div class="detail">
    <h2>${r.method} /${r.endpoint}
      <span style="font-weight:400;color:var(--text2)">
      (${r.model || 'unknown'})</span></h2>
    <div class="metrics">${metrics.join('')}</div>
    ${section('Request Headers', fmtHeaders(r.request_headers), true)}
    ${section('Request Body', fmtJson(r.request_body), false)}
    ${section('Response Headers', fmtHeaders(r.response_headers), true)}
    ${section('Response', fmtJson(r.response_body), false)}
    ${r.response_body_raw && r.response_body_raw !== r.response_body
      ? section('Raw Response', esc(r.response_body_raw), true)
      : ''}
  </div>`;
}

function mk(label, value) {
  return `<div class="metric"><div class="label">${label}</div>` +
    `<div class="value">${value}</div></div>`;
}

let sectionCounter = 0;
function section(title, content, collapsed) {
  const id = 'sec-' + (sectionCounter++);
  const display = collapsed ? 'none' : 'block';
  const arrowCls = collapsed ? 'arrow' : 'arrow open';
  return `<div class="section">
    <div class="section-header" onclick="toggleSection('${id}',this)">
      <span class="${arrowCls}">&#9654;</span> ${title}
    </div>
    <div class="section-body" id="${id}" style="display:${display}">
      ${content ? '<pre>' + content + '</pre>' : '<div class="empty">(empty)</div>'}
    </div>
  </div>`;
}

function toggleSection(id, header) {
  const body = document.getElementById(id);
  const arrow = header.querySelector('.arrow');
  if (body.style.display === 'none') {
    body.style.display = 'block';
    arrow.classList.add('open');
  } else {
    body.style.display = 'none';
    arrow.classList.remove('open');
  }
}

function esc(s) {
  if (!s) return '';
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

function fmtJson(s) {
  if (!s) return '';
  try {
    const obj = JSON.parse(s);
    return hlJson(JSON.stringify(obj, null, 2));
  } catch(e) { return esc(s); }
}

function hlJson(s) {
  var out = '';
  var i = 0;
  while (i < s.length) {
    var c = s[i];
    if (c === '"') {
      var j = i + 1;
      while (j < s.length && s[j] !== '"') {
        if (s[j] === '\\\\') j++;
        j++;
      }
      var tok = s.slice(i, j + 1);
      var rest = s.slice(j + 1).match(/^\\s*:/);
      if (rest) {
        out += '<span class="hl-key">' + esc(tok) + '</span>';
      } else {
        out += '<span class="hl-str">' + esc(tok) + '</span>';
      }
      i = j + 1;
    } else if (c === '-' || (c >= '0' && c <= '9')) {
      var m = s.slice(i).match(/^-?\\d+\\.?\\d*/);
      if (m) {
        out += '<span class="hl-num">' + m[0] + '</span>';
        i += m[0].length;
      } else { out += esc(c); i++; }
    } else if (s.slice(i, i+4) === 'true') {
      out += '<span class="hl-bool">true</span>'; i += 4;
    } else if (s.slice(i,i+5) === 'false') {
      out += '<span class="hl-bool">false</span>'; i += 5;
    } else if (s.slice(i, i+4) === 'null') {
      out += '<span class="hl-null">null</span>'; i += 4;
    } else if ('{}[],'.indexOf(c) >= 0) {
      out += '<span class="hl-brace">' + c + '</span>'; i++;
    } else { out += esc(c); i++; }
  }
  return out;
}

function fmtHeaders(h) {
  if (!h || Object.keys(h).length === 0) return '';
  return Object.entries(h).map(([k,v]) =>
    '<span class="hl-hdr-key">' + esc(k) + '</span>: ' +
    '<span class="hl-hdr-val">' + esc(v) + '</span>'
  ).join('\\n');
}

loadRecords();
</script>
</body>
</html>
"""
