"use strict";

const state = {
  records: [], total: 0, offset: 0, pageSize: 100, query: "",
  listVersion: 0, listController: null, listStatus: "idle", listError: "",
  isLoadingMore: false, moreError: "", exhausted: false,
  selectedId: null, detailVersion: 0, detailController: null,
  searchDebounce: null, sectionCounter: 0, focusedId: null,
};

async function checkedJson(url, signal) {
  const response = await fetch(url, { signal });
  if (!response.ok) throw new Error(`Request failed (HTTP ${response.status}).`);
  return response.json();
}

const api = {
  list(query, offset, limit, signal) {
    const params = new URLSearchParams({ offset: String(offset), limit: String(limit) });
    if (query) params.set("search", query);
    return checkedJson("/api/records?" + params, signal);
  },
  get(id, signal) {
    return checkedJson("/api/records/" + encodeURIComponent(id), signal);
  },
};

function esc(s) {
  if (s == null) return "";
  const d = document.createElement("div");
  d.textContent = String(s);
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
  for (const child of children == null ? [] : [].concat(children)) {
    if (child != null) node.appendChild(typeof child === "string" ? document.createTextNode(child) : child);
  }
  return node;
}

function resetList() {
  clearTimeout(state.searchDebounce);
  state.searchDebounce = null;
  state.listController?.abort();
  state.listController = new AbortController();
  state.listVersion += 1;
  state.query = document.getElementById("search").value;
  state.records = [];
  state.total = 0;
  state.offset = 0;
  state.listStatus = "loading";
  state.listError = "";
  state.moreError = "";
  state.isLoadingMore = false;
  state.exhausted = false;
  renderList();
  return state.listVersion;
}

async function fetchPage(version, append, trigger = document.activeElement) {
  if (version !== state.listVersion) return;
  const query = state.query;
  const offset = state.offset;
  const previousCount = state.records.length;
  const footerFocus = ["load-more-btn", "list-retry"].includes(trigger?.id) ? trigger : null;
  const restoreFooterFocus = () => footerFocus && (document.activeElement === footerFocus || document.activeElement === document.body);
  if (append) { state.isLoadingMore = true; state.moreError = ""; }
  renderListStatus();
  try {
    const data = await api.list(query, offset, state.pageSize, state.listController.signal);
    if (version !== state.listVersion) return;
    if (!Array.isArray(data.items) || !Number.isInteger(data.total) || data.total < 0) {
      throw new Error("Invalid record list.");
    }
    const seen = new Set(state.records.map(record => record.id));
    const fresh = data.items.filter(record => {
      if (seen.has(record.id)) return false;
      seen.add(record.id);
      return true;
    });
    state.records = state.records.concat(fresh);
    state.total = data.total;
    state.offset = offset + data.items.length;
    state.exhausted = data.items.length === 0 || state.offset >= state.total;
    state.listStatus = "success";
    renderList();
    if (restoreFooterFocus()) {
      const target = document.querySelectorAll(".record")[Math.min(previousCount, state.records.length - 1)];
      if (target) target.focus();
    }
  } catch (error) {
    if (version !== state.listVersion || error.name === "AbortError") return;
    if (append) state.moreError = "Could not load more requests. Try again.";
    else {
      state.listStatus = "error";
      state.listError = query ? "Could not search requests. Try again." : "Could not load requests. Try again.";
    }
  } finally {
    if (version === state.listVersion) {
      state.isLoadingMore = false;
      renderListStatus();
      if (restoreFooterFocus()) {
        if (state.listError || state.moreError) document.getElementById("list-retry").focus();
        else if (!state.records.length) document.getElementById("refresh-btn").focus();
      }
    }
  }
}

function loadRecords() {
  const trigger = document.activeElement;
  return fetchPage(resetList(), false, trigger);
}

function loadMoreRecords() {
  if (state.listStatus !== "success" || state.isLoadingMore || state.exhausted) return;
  return fetchPage(state.listVersion, true);
}

function onSearchInput() {
  const version = resetList(); // Invalidate old responses before the debounce delay.
  state.searchDebounce = setTimeout(() => fetchPage(version, false), 150);
}

function recordTime(timestamp) {
  const value = String(timestamp || "Unknown time");
  if (!/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}/.test(value)) return value;
  const now = new Date();
  const today = [now.getFullYear(), String(now.getMonth() + 1).padStart(2, "0"), String(now.getDate()).padStart(2, "0")].join("-");
  return value.slice(0, 10) === today ? value.slice(11, 19) : value.slice(0, 10) + " " + value.slice(11, 19);
}

function recordButton(record) {
  const button = el("button", { type: "button", class: "record", "aria-pressed": "false", tabindex: "-1" });
  button.dataset.id = record.id;
  const top = el("span", { class: "top" }, [
    el("span", { class: "method" }, String(record.method)),
    el("span", { class: "time", title: String(record.timestamp) }, recordTime(record.timestamp)),
  ]);
  const tokens = record.tokens_prompt || record.tokens_completion ? `${record.tokens_prompt}+${record.tokens_completion}` : "-";
  const speed = record.tokens_per_second ? record.tokens_per_second.toFixed(1) + " tok/s" : "";
  const meta = el("span", { class: "meta" }, [
    el("span", null, record.model || "?"),
    el("span", { class: record.response_status < 400 ? "status-ok" : "status-err" }, String(record.response_status)),
    el("span", null, tokens), el("span", null, speed),
  ]);
  button.append(top, el("span", { class: "path" }, "/" + record.endpoint), meta);
  button.addEventListener("click", () => selectRecord(record.id));
  button.addEventListener("focus", () => {
    state.focusedId = record.id;
    document.querySelectorAll(".record").forEach(row => { row.tabIndex = row === button ? 0 : -1; });
  });
  button.addEventListener("keydown", event => {
    const keys = ["ArrowUp", "ArrowDown", "Home", "End"];
    if (!keys.includes(event.key)) return;
    event.preventDefault();
    const rows = [...document.querySelectorAll(".record")];
    const current = rows.indexOf(button);
    const index = event.key === "Home" ? 0 : event.key === "End" ? rows.length - 1 :
      Math.max(0, Math.min(rows.length - 1, current + (event.key === "ArrowDown" ? 1 : -1)));
    rows[index].focus();
  });
  return button;
}

function updateSelection() {
  document.querySelectorAll(".record").forEach(row => {
    const selected = row.dataset.id === state.selectedId;
    row.classList.toggle("selected", selected);
    row.setAttribute("aria-pressed", String(selected));
  });
}

function renderList() {
  const activeId = document.activeElement?.classList.contains("record") ? document.activeElement.dataset.id : null;
  const list = document.getElementById("list");
  list.replaceChildren(...state.records.map(recordButton));
  const rows = [...list.children];
  const tabRow = rows.find(row => row.dataset.id === (activeId || state.focusedId)) || rows[0];
  if (tabRow) {
    tabRow.tabIndex = 0;
    if (activeId) tabRow.focus();
  }
  updateSelection();
  renderListStatus();
}

function renderListStatus() {
  const busy = state.listStatus === "loading" || state.isLoadingMore;
  document.getElementById("list-pane").setAttribute("aria-busy", String(busy));
  const status = document.getElementById("list-status");
  const count = document.getElementById("record-count");
  const retry = document.getElementById("list-retry");
  const more = document.getElementById("load-more-btn");
  let message = "";
  if (state.listStatus === "loading") message = state.query ? "Searching requests…" : "Loading requests…";
  else if (state.listStatus === "error") message = state.listError;
  else if (state.isLoadingMore) message = "Loading more requests…";
  else if (state.moreError) message = state.moreError;
  else if (state.total === 0) message = state.query ? `No matches for “${state.query}”.` : "No requests recorded yet. Enable logging with drove config observe true.";
  else if (state.records.length === 0) message = "No requests available on this page. Refresh to check again.";
  status.textContent = message;
  status.classList.toggle("error", !!state.listError || !!state.moreError);
  count.hidden = state.listStatus !== "success";
  const shown = state.records.length.toLocaleString();
  const total = state.total.toLocaleString();
  count.textContent = state.query && state.records.length === state.total ? `${total} matches` : `${shown} of ${total} ${state.query ? "matches" : "requests"}`;
  retry.hidden = !state.listError && !state.moreError;
  more.hidden = state.listStatus !== "success" || state.exhausted || !!state.moreError;
  more.disabled = state.isLoadingMore;
  more.textContent = state.isLoadingMore ? "Loading…" : "Load more";
}

function detailNotice(message, retry) {
  const notice = el("div", { class: "detail-placeholder", role: "status" }, el("p", null, message));
  if (retry) notice.appendChild(el("button", { type: "button", class: "retry", onclick: retry }, "Retry"));
  document.getElementById("detail").replaceChildren(notice);
}

async function showRecord(id) {
  state.detailController?.abort();
  state.detailController = new AbortController();
  const version = ++state.detailVersion;
  state.selectedId = id;
  updateSelection();
  const root = document.getElementById("detail");
  const retryFocus = root.contains(document.activeElement) && document.activeElement.classList.contains("retry") ? document.activeElement : null;
  const restoreFocus = () => retryFocus && (document.activeElement === retryFocus || document.activeElement === document.body);
  root.setAttribute("aria-busy", String(!!id));
  if (!id) { detailNotice("Select a request from the list to inspect."); return; }
  detailNotice("Loading request details…");
  try {
    const record = await api.get(id, state.detailController.signal);
    if (version !== state.detailVersion) return;
    if (!record || record.id !== id) throw new Error("Invalid record.");
    renderDetail(record);
    if (restoreFocus()) root.querySelector(".section-header")?.focus();
  } catch (error) {
    if (version !== state.detailVersion || error.name === "AbortError") return;
    detailNotice("Could not load request details. The request may no longer exist.", () => showRecord(id));
    if (restoreFocus()) root.querySelector(".retry")?.focus();
  } finally {
    if (version === state.detailVersion) root.setAttribute("aria-busy", "false");
  }
}

function selectRecord(id) {
  const hash = "#record=" + encodeURIComponent(id);
  if (location.hash !== hash) history.pushState(null, "", hash);
  return showRecord(id);
}

function selectionFromHash() {
  if (!location.hash) return showRecord(null);
  try {
    if (!location.hash.startsWith("#record=")) throw new Error("Invalid selection");
    const id = decodeURIComponent(location.hash.slice(8));
    if (!id) throw new Error("Empty selection");
    return showRecord(id);
  } catch {
    showRecord(null);
    detailNotice("This request link is invalid. Clear the selection and choose a request.", () => {
      history.replaceState(null, "", location.pathname + location.search);
      showRecord(null);
    });
  }
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
    <button type="button" class="section-header" aria-controls="${id}" aria-expanded="${!collapsed}">
      <span class="${arrowCls}">&#9654;</span> ${title}
    </button>
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
  sectionNode.querySelector(".section-header").setAttribute("aria-expanded", String(!isOpen));
}

function renderDetail(r) {
  const root = document.getElementById("detail");
  root.innerHTML = "";
  const detail = el("div", { class: "detail" });
  detail.innerHTML = `
    <h2>${esc(r.method)} /${esc(r.endpoint)}
      <span style="font-weight:400;color:var(--text2)">
      (${esc(r.model || "unknown")})</span></h2>
    <p class="timestamp"></p>
    <div class="metrics">${buildMetrics(r)}</div>`;
  detail.querySelector(".timestamp").textContent = String(r.timestamp || "Unknown time");
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
    <button type="button" class="section-header" aria-controls="${id}" aria-expanded="true">
      <span class="arrow open">&#9654;</span> ${title}
    </button>
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
  const toggle = entries.length
    ? el("button", { type: "button", class: "jt-toggle", "aria-expanded": "true", "aria-label": "Toggle " + (key === undefined ? (isArr ? "array" : "object") : String(key)) }, "▼")
    : el("span", { class: "jt-toggle", "aria-hidden": "true" }, " ");
  row.appendChild(toggle);
  const head = el("span");
  head.innerHTML = keyPrefix(key) + `<span class="hl-brace">${open}</span>`;
  row.appendChild(head);
  const ellipsis = el("span", { class: "jt-ellipsis", style: "display:none" }, "...");
  row.appendChild(ellipsis);
  wrap.appendChild(row);

  const children = el("div", { class: "jt-children", id: "tree-" + state.sectionCounter++ });
  if (entries.length) toggle.setAttribute("aria-controls", children.id);
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
      toggle.setAttribute("aria-expanded", String(!wasOpen));
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
document.getElementById("list-retry").addEventListener("click", () => state.moreError ? loadMoreRecords() : loadRecords());
window.addEventListener("hashchange", selectionFromHash);
loadRecords();
selectionFromHash();
