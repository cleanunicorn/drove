"""Web UI for browsing observed API requests and responses."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

from vllama.observe import _record_to_dict, list_records, load_record, record_matches


def create_observe_app(observe_dir: Path, model: str | None = None) -> FastAPI:
    app = FastAPI(title="vllama observe")

    @app.get("/api/records")
    async def get_records(search: str = "") -> JSONResponse:
        records = list_records(observe_dir, model)
        items = []
        for _path, record in records:
            if search and not record_matches(record, search):
                continue
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
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
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
  header input.search { background: var(--bg3); color: var(--text);
                        border: 1px solid var(--border); padding: 4px 10px;
                        border-radius: 4px; font-size: 13px; min-width: 260px;
                        font-family: inherit; }
  header input.search:focus { outline: none; border-color: var(--accent); }
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
  .section-body .jt { padding: 12px 14px; font-family: 'Fira Code', 'Cascadia Code', monospace;
                      font-size: 12px; line-height: 1.6; }
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
  /* JSON tree */
  .jt-row { display: flex; align-items: flex-start; }
  .jt-toggle { width: 14px; cursor: pointer; color: var(--text2);
               user-select: none; flex-shrink: 0; text-align: center; }
  .jt-toggle:hover { color: var(--text); }
  .jt-children { padding-left: 20px; }
  .jt-children.collapsed { display: none; }
  .jt-ellipsis { color: var(--text2); cursor: pointer; }
  .jt-ellipsis:hover { color: var(--text); }
  .jt-comma { color: var(--text2); }
  /* Markdown */
  .md-toggle { background: var(--bg3); color: var(--accent); border: 1px solid var(--border);
                padding: 1px 6px; border-radius: 3px; cursor: pointer; font-size: 10px;
                margin-right: 8px; vertical-align: middle; }
  .md-toggle:hover { background: var(--border); }
  .md-render {
    margin-top: 6px; padding: 10px; background: var(--bg);
    border: 1px solid var(--border); border-radius: 4px;
    color: var(--text); line-height: 1.5; font-family: sans-serif;
  }
  .md-render h1, .md-render h2, .md-render h3 {
    margin: 8px 0 4px; font-size: 1.1em; color: var(--accent2);
  }
  .md-render p { margin-bottom: 8px; }
  .md-render code {
    background: var(--bg2); padding: 2px 4px;
    border-radius: 3px; font-family: monospace;
  }
  .md-render pre { background: var(--bg2); padding: 8px; border-radius: 4px; overflow-x: auto; }
  .md-render pre code { background: none; padding: 0; }
  .md-render ul, .md-render ol { margin-left: 20px; margin-bottom: 8px; }

</style>
</head>
<body>
<header>
  <h1>vllama observe</h1>
  <span class="filter">{{FILTER_LABEL}}</span>
  <input id="search" class="search" type="text" placeholder="Search anything…"
         oninput="onSearchInput()" autocomplete="off">
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
let searchDebounce = null;

function onSearchInput() {
  if (searchDebounce) clearTimeout(searchDebounce);
  searchDebounce = setTimeout(loadRecords, 150);
}

async function loadRecords() {
  const q = (document.getElementById('search') || {}).value || '';
  const url = q ? '/api/records?search=' + encodeURIComponent(q) : '/api/records';
  const resp = await fetch(url);
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
    ${section('Response Headers', fmtHeaders(r.response_headers), true)}
    ${r.response_body_raw && r.response_body_raw !== r.response_body
      ? section('Raw Response', esc(r.response_body_raw), true)
      : ''}
  </div>`;
  mountJsonTree('Request Body', r.request_body, false);
  mountJsonTree('Response', r.response_body, false);
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

function fmtHeaders(h) {
  if (!h || Object.keys(h).length === 0) return '';
  return Object.entries(h).map(([k,v]) =>
    '<span class="hl-hdr-key">' + esc(k) + '</span>: ' +
    '<span class="hl-hdr-val">' + esc(v) + '</span>'
  ).join('\\n');
}

function mountJsonTree(title, raw, collapsed) {
  var det = document.querySelector('.detail');
  if (!det) return;
  var rawSec = det.querySelector('.section:last-of-type');
  var div = document.createElement('div');
  div.className = 'section';
  var id = 'jt-' + (sectionCounter++);
  div.innerHTML =
    '<div class="section-header" onclick="toggleSection(\\'' +
    id + '\\',this)">' +
    '<span class="' + (collapsed ? 'arrow' : 'arrow open') +
    '">&#9654;</span> ' + title + '</div>' +
    '<div class="section-body" id="' + id + '" style="display:' +
    (collapsed ? 'none' : 'block') + '"></div>';
  if (rawSec) det.insertBefore(div, rawSec);
  else det.appendChild(div);
  var body = div.querySelector('.section-body');
  if (!raw) { body.innerHTML = '<div class="empty">(empty)</div>'; return; }
  try {
    var obj = JSON.parse(raw);
    var container = document.createElement('div');
    container.className = 'jt';
    container.appendChild(buildTree(obj));
    body.appendChild(container);
  } catch(e) {
    body.innerHTML = '<pre>' + esc(raw) + '</pre>';
  }
}

function buildTree(val, key) {
  var frag = document.createDocumentFragment();
  if (val === null) {
    frag.appendChild(mkLeaf(key, '<span class="hl-null">null</span>'));
  } else if (typeof val === 'boolean') {
    frag.appendChild(mkLeaf(key, '<span class="hl-bool">' + val + '</span>'));
  } else if (typeof val === 'number') {
    frag.appendChild(mkLeaf(key, '<span class="hl-num">' + val + '</span>'));
  } else if (typeof val === 'string') {
    frag.appendChild(mkLeaf(key, '<span class="hl-str">' +
      esc(JSON.stringify(val)) + '</span>', val));
  } else if (Array.isArray(val)) {
    frag.appendChild(mkNode(key, val, '[', ']'));
  } else if (typeof val === 'object') {
    frag.appendChild(mkNode(key, val, '{', '}'));
  }
  return frag;
}

function mkLeaf(key, html, rawVal) {
  var row = document.createElement('div');
  row.className = 'jt-row';
  var sp = document.createElement('span');
  sp.style.width = '14px';
  sp.style.flexShrink = '0';
  row.appendChild(sp);
  var content = document.createElement('span');
  var pre = key !== undefined
    ? '<span class="hl-key">' + esc(JSON.stringify(key)) +
      '</span>: '
    : '';
  content.innerHTML = pre + html;
  if ((key === "content" || key === "reasoning_content") && typeof rawVal === "string") {
    var btn = document.createElement('button');
    btn.className = 'md-toggle';
    btn.textContent = 'MD';
    btn.onclick = function() { toggleMd(btn, rawVal, content); };
    content.prepend(btn);
  }
  row.appendChild(content);
  return row;
}

function toggleMd(btn, raw, parent) {
  var render = parent.querySelector('.md-render');
  var str = parent.querySelector('.hl-str');
  if (render) {
    render.remove();
    str.style.display = 'inline';
    btn.textContent = 'MD';
  } else {
    render = document.createElement('div');
    render.className = 'md-render';
    render.innerHTML = marked.parse(raw);
    parent.appendChild(render);
    str.style.display = 'none';
    btn.textContent = 'RAW';
  }
}

function mkNode(key, val, open, close) {
  var isArr = Array.isArray(val);
  var entries = isArr ? val.map(function(v,i){return [i,v];})
    : Object.entries(val);
  var wrap = document.createElement('div');
  var row = document.createElement('div');
  row.className = 'jt-row';
  var toggle = document.createElement('span');
  toggle.className = 'jt-toggle';
  toggle.textContent = entries.length ? '\\u25BC' : ' ';
  row.appendChild(toggle);
  var head = document.createElement('span');
  var pre = key !== undefined
    ? '<span class="hl-key">' + esc(JSON.stringify(key)) +
      '</span>: '
    : '';
  head.innerHTML = pre + '<span class="hl-brace">' + open + '</span>';
  row.appendChild(head);
  var ellipsis = document.createElement('span');
  ellipsis.className = 'jt-ellipsis';
  ellipsis.style.display = 'none';
  ellipsis.textContent = '...';
  row.appendChild(ellipsis);
  wrap.appendChild(row);
  var children = document.createElement('div');
  children.className = 'jt-children';
  for (var ci = 0; ci < entries.length; ci++) {
    var ek = isArr ? undefined : entries[ci][0];
    var ev = entries[ci][1];
    var child = buildTree(ev, ek);
    if (ci < entries.length - 1) {
      appendComma(child);
    }
    children.appendChild(child);
  }
  wrap.appendChild(children);
  var closeRow = document.createElement('div');
  closeRow.innerHTML = '<span style="width:14px;display:inline-block">' +
    '</span><span class="hl-brace">' + close + '</span>';
  wrap.appendChild(closeRow);
  if (entries.length) {
    toggle.onclick = function() {
      var open = !children.classList.contains('collapsed');
      children.classList.toggle('collapsed');
      closeRow.style.display = open ? 'none' : '';
      ellipsis.style.display = open ? 'inline' : 'none';
      toggle.textContent = open ? '\\u25B6' : '\\u25BC';
    };
    ellipsis.onclick = toggle.onclick;
  }
  return wrap;
}

function appendComma(frag) {
  var last = frag.lastChild || frag;
  if (last.nodeType === 1) {
    var c = document.createElement('span');
    c.className = 'jt-comma';
    c.textContent = ',';
    last.appendChild(c);
  }
}

loadRecords();
</script>
</body>
</html>
"""
