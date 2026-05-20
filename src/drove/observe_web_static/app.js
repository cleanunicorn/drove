"use strict";

const state = {
  records: [],
  total: 0,
  offset: 0,
  pageSize: 100,
  isLoadingMore: false,
  selectedId: null,
  searchDebounce: null,
  sectionCounter: 0,
};

const api = {
  async list(query, offset, limit) {
    const params = new URLSearchParams();
    if (query) params.set("search", query);
    params.set("offset", String(offset));
    params.set("limit", String(limit));
    const url = "/api/records?" + params.toString();
    const resp = await fetch(url);
    return resp.json();
  },
  async get(id) {
    const resp = await fetch("/api/records/" + id);
    return resp.json();
  },
};

function esc(s) {
  if (!s) return "";
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}

function el(tag, attrs, children) {
  const node = document.createElement(tag);
  if (attrs) {
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "class") node.className = v;
      else if (k === "style") node.style.cssText = v;
      else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
      else node.setAttribute(k, v);
    }
  }
  if (children) {
    for (const c of [].concat(children)) {
      if (c == null) continue;
      node.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
    }
  }
  return node;
}

async function loadRecords() {
  const q = (document.getElementById("search") || {}).value || "";
  state.offset = 0;
  const data = await api.list(q, 0, state.pageSize);
  state.records = data.items;
  state.total = data.total;
  renderList();
}

async function loadMoreRecords() {
  if (state.isLoadingMore) return;
  state.isLoadingMore = true;
  renderLoadMoreState();
  const q = (document.getElementById("search") || {}).value || "";
  try {
    const nextOffset = state.records.length;
    const data = await api.list(q, nextOffset, state.pageSize);
    state.records = state.records.concat(data.items);
    state.total = data.total;
    renderList();
  } finally {
    state.isLoadingMore = false;
    renderLoadMoreState();
  }
}

function onSearchInput() {
  if (state.searchDebounce) clearTimeout(state.searchDebounce);
  state.searchDebounce = setTimeout(loadRecords, 150);
}

function recordSummaryHtml(r) {
  const time = r.timestamp.substring(11, 19);
  const model = r.model || "?";
  const statusCls = r.response_status < 400 ? "status-ok" : "status-err";
  const tokens = r.tokens_prompt || r.tokens_completion
    ? `${r.tokens_prompt}+${r.tokens_completion}`
    : "-";
  const speed = r.tokens_per_second ? r.tokens_per_second.toFixed(1) + " tok/s" : "";
  const sel = r.id === state.selectedId ? " selected" : "";
  return `<div class="record${sel}" data-id="${r.id}">
    <div class="top">
      <span class="method">${esc(r.method)}</span>
      <span class="time">${time}</span>
    </div>
    <div class="path">/${esc(r.endpoint)}</div>
    <div class="meta">
      <span>${esc(model)}</span>
      <span class="${statusCls}">${r.response_status}</span>
      <span>${tokens}</span>
      <span>${speed}</span>
    </div>
  </div>`;
}

function renderList() {
  const list = document.getElementById("list");
  list.innerHTML = state.records.map(recordSummaryHtml).join("");
  list.querySelectorAll(".record").forEach((node) => {
    node.addEventListener("click", () => selectRecord(node.dataset.id));
  });
  const loadMoreBtn = document.getElementById("load-more-btn");
  loadMoreBtn.style.display = state.records.length < state.total ? "inline-block" : "none";
  renderLoadMoreState();
}

function renderLoadMoreState() {
  const loadMoreBtn = document.getElementById("load-more-btn");
  if (!loadMoreBtn) return;
  loadMoreBtn.disabled = state.isLoadingMore;
}

async function selectRecord(id) {
  state.selectedId = id;
  renderList();
  const record = await api.get(id);
  renderDetail(record);
}

function metric(label, value) {
  return `<div class="metric"><div class="label">${label}</div>` +
    `<div class="value">${value}</div></div>`;
}

function buildMetrics(r) {
  const parts = [metric("Status", r.response_status)];
  parts.push(metric("Duration", r.duration_seconds.toFixed(3) + "s"));
  if (r.ttft_seconds != null) parts.push(metric("TTFT", r.ttft_seconds.toFixed(3) + "s"));
  if (r.tokens_prompt || r.tokens_completion)
    parts.push(metric("Tokens", `${r.tokens_prompt} + ${r.tokens_completion}`));
  if (r.tokens_per_second != null)
    parts.push(metric("Speed", r.tokens_per_second.toFixed(1) + " tok/s"));
  return parts.join("");
}

function fmtHeaders(h) {
  if (!h || Object.keys(h).length === 0) return "";
  return Object.entries(h)
    .map(([k, v]) =>
      '<span class="hl-hdr-key">' + esc(k) + '</span>: ' +
      '<span class="hl-hdr-val">' + esc(v) + '</span>'
    )
    .join("\n");
}

function makeSection(title, contentHtml, collapsed) {
  const id = "sec-" + state.sectionCounter++;
  const arrowCls = collapsed ? "arrow" : "arrow open";
  const display = collapsed ? "none" : "block";
  const inner = contentHtml
    ? `<pre>${contentHtml}</pre>`
    : `<div class="empty">(empty)</div>`;
  const node = el("div", { class: "section" });
  node.innerHTML = `
    <div class="section-header">
      <span class="${arrowCls}">&#9654;</span> ${title}
    </div>
    <div class="section-body" id="${id}" style="display:${display}">${inner}</div>`;
  const header = node.querySelector(".section-header");
  header.addEventListener("click", () => toggleSection(node));
  return node;
}

function toggleSection(sectionNode) {
  const body = sectionNode.querySelector(".section-body");
  const arrow = sectionNode.querySelector(".arrow");
  const isOpen = body.style.display !== "none";
  body.style.display = isOpen ? "none" : "block";
  arrow.classList.toggle("open", !isOpen);
}

function renderDetail(r) {
  const root = document.getElementById("detail");
  root.innerHTML = "";
  const detail = el("div", { class: "detail" });
  detail.innerHTML = `
    <h2>${esc(r.method)} /${esc(r.endpoint)}
      <span style="font-weight:400;color:var(--text2)">
      (${esc(r.model || "unknown")})</span></h2>
    <div class="metrics">${buildMetrics(r)}</div>`;
  root.appendChild(detail);

  detail.appendChild(makeJsonSection("Request Body", r.request_body));
  detail.appendChild(makeSection("Request Headers", fmtHeaders(r.request_headers), true));
  detail.appendChild(makeJsonSection("Response", r.response_body));
  detail.appendChild(makeSection("Response Headers", fmtHeaders(r.response_headers), true));
  if (r.response_body_raw && r.response_body_raw !== r.response_body) {
    detail.appendChild(makeSection("Raw Response", esc(r.response_body_raw), true));
  }
}

function makeJsonSection(title, raw) {
  const id = "jt-" + state.sectionCounter++;
  const node = el("div", { class: "section" });
  node.innerHTML = `
    <div class="section-header">
      <span class="arrow open">&#9654;</span> ${title}
    </div>
    <div class="section-body" id="${id}" style="display:block"></div>`;
  node.querySelector(".section-header").addEventListener("click", () => toggleSection(node));
  const body = node.querySelector(".section-body");
  if (!raw) {
    body.innerHTML = '<div class="empty">(empty)</div>';
    return node;
  }
  try {
    const obj = JSON.parse(raw);
    const container = el("div", { class: "jt" });
    container.appendChild(buildTree(obj));
    body.appendChild(container);
  } catch {
    body.innerHTML = `<pre>${esc(raw)}</pre>`;
  }
  return node;
}

function buildTree(val, key) {
  const frag = document.createDocumentFragment();
  if (val === null) {
    frag.appendChild(makeLeaf(key, '<span class="hl-null">null</span>'));
  } else if (typeof val === "boolean") {
    frag.appendChild(makeLeaf(key, `<span class="hl-bool">${val}</span>`));
  } else if (typeof val === "number") {
    frag.appendChild(makeLeaf(key, `<span class="hl-num">${val}</span>`));
  } else if (typeof val === "string") {
    frag.appendChild(
      makeLeaf(key, `<span class="hl-str">${esc(JSON.stringify(val))}</span>`, val)
    );
  } else if (Array.isArray(val)) {
    frag.appendChild(makeNode(key, val, "[", "]"));
  } else if (typeof val === "object") {
    frag.appendChild(makeNode(key, val, "{", "}"));
  }
  return frag;
}

function keyPrefix(key) {
  return key !== undefined
    ? `<span class="hl-key">${esc(JSON.stringify(key))}</span>: `
    : "";
}

const MD_KEYS = new Set(["content", "reasoning_content"]);

function makeLeaf(key, html, rawVal) {
  const row = el("div", { class: "jt-row" });
  row.appendChild(el("span", { style: "width:14px;flex-shrink:0" }));
  const content = el("span");
  if (MD_KEYS.has(key) && typeof rawVal === "string") {
    const btn = el("button", { class: "md-toggle md-toggle-left" }, "MD");
    btn.addEventListener("click", () => toggleMd(btn, rawVal, content));
    content.appendChild(btn);
  }
  content.insertAdjacentHTML("beforeend", keyPrefix(key) + html);
  row.appendChild(content);
  return row;
}

function toggleMd(btn, raw, parent) {
  const existing = parent.querySelector(".md-render");
  const str = parent.querySelector(".hl-str");
  if (existing) {
    existing.remove();
    if (str) str.style.display = "inline";
    btn.textContent = "MD";
    return;
  }
  const render = el("div", { class: "md-render" });
  render.innerHTML = marked.parse(raw);
  parent.appendChild(render);
  if (str) str.style.display = "none";
  btn.textContent = "RAW";
}

function makeNode(key, val, open, close) {
  const isArr = Array.isArray(val);
  const entries = isArr ? val.map((v, i) => [i, v]) : Object.entries(val);
  const wrap = el("div");

  const row = el("div", { class: "jt-row" });
  const toggle = el("span", { class: "jt-toggle" }, entries.length ? "▼" : " ");
  row.appendChild(toggle);
  const head = el("span");
  head.innerHTML = keyPrefix(key) + `<span class="hl-brace">${open}</span>`;
  row.appendChild(head);
  const ellipsis = el("span", { class: "jt-ellipsis", style: "display:none" }, "...");
  row.appendChild(ellipsis);
  wrap.appendChild(row);

  const children = el("div", { class: "jt-children" });
  for (let i = 0; i < entries.length; i++) {
    const [k, v] = entries[i];
    const child = buildTree(v, isArr ? undefined : k);
    if (i < entries.length - 1) appendComma(child);
    children.appendChild(child);
  }
  wrap.appendChild(children);

  const closeRow = el("div");
  closeRow.innerHTML =
    '<span style="width:14px;display:inline-block"></span>' +
    `<span class="hl-brace">${close}</span>`;
  wrap.appendChild(closeRow);

  if (entries.length) {
    const onToggle = () => {
      const wasOpen = !children.classList.contains("collapsed");
      children.classList.toggle("collapsed");
      closeRow.style.display = wasOpen ? "none" : "";
      ellipsis.style.display = wasOpen ? "inline" : "none";
      toggle.textContent = wasOpen ? "▶" : "▼";
    };
    toggle.addEventListener("click", onToggle);
    ellipsis.addEventListener("click", onToggle);
  }
  return wrap;
}

function appendComma(frag) {
  const last = frag.lastChild || frag;
  if (last.nodeType === 1) {
    last.appendChild(el("span", { class: "jt-comma" }, ","));
  }
}

document.getElementById("search").addEventListener("input", onSearchInput);
document.getElementById("refresh-btn").addEventListener("click", loadRecords);
document.getElementById("load-more-btn").addEventListener("click", loadMoreRecords);
loadRecords();
