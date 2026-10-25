// BetterWright globals plus BASE_URL/BASELINE supplied by the Python runner.
const note = "Verify packaged observe Markdown safety, escaping and offline rendering";
let assertions = 0;
function check(value, message) {
  assertions++;
  if (!value) throw new Error(message);
}
const seed = {
  id: "security-fixture", timestamp: "2026-10-25T12:00:00", model: "fixture",
  endpoint: "v1/chat/completions", method: "POST", response_status: 200,
  request_headers: {}, response_headers: {}, request_body: "{}", response_body: "{}",
  response_body_raw: "", tokens_prompt: 0, tokens_completion: 0,
  tokens_per_second: 0, ttft_seconds: 0, duration_seconds: 0,
};
await page.addInitScript(() => {
  window.__executed = 0;
  window.__errors = [];
  window.addEventListener("error", e => { if (e.message) window.__errors.push(e.message); });
  window.alert = () => { window.__executed++; };
});
await page.goto(BASE_URL);
await snapshot({interactive: true});
await snapshot();

async function selectFixture(record) {
  await page.evaluate(async value => {
    await fetch("/__fixture", {
      method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(value),
    });
    await loadRecords();
  }, record);
  await page.evaluate(() => document.getElementById("detail").replaceChildren());
  await snapshot();
  await page.locator(".record").click();
  await page.locator(".detail").waitFor({state: "visible"});
  await snapshot({diff: true});
}
async function clickMd() {
  const view = await snapshot({interactive: true});
  const match = view.match(/button "MD"[^\n]*\[ref=(e\d+)\]/);
  check(match, "MD button must be present in the inspected view: " + view);
  await page.locator(`aria-ref=${match[1]}`).click();
  await snapshot({diff: true});
}
async function clickRaw() {
  const view = await snapshot({interactive: true});
  const match = view.match(/button "RAW"[^\n]*\[ref=(e\d+)\]/);
  check(match, "RAW button must be present in the inspected view: " + view);
  await page.locator(`aria-ref=${match[1]}`).click();
  await snapshot({diff: true});
}
async function assertSafe() {
  const result = await page.evaluate(() => {
    const roots = [...document.querySelectorAll(".md-render")];
    const tags = new Set(["H1","H2","H3","H4","H5","H6","P","BR","HR","STRONG","EM",
      "DEL","BLOCKQUOTE","UL","OL","LI","PRE","CODE","A","TABLE","THEAD","TBODY","TR","TH","TD"]);
    const bad = [];
    for (const root of roots) for (const node of root.querySelectorAll("*")) {
      if (!tags.has(node.tagName)) bad.push(node.outerHTML);
      for (const attr of node.attributes) {
        if (!["href", "title", "start"].includes(attr.name)) bad.push(attr.name);
        if (attr.name === "href" && /^(javascript|vbscript|data):/i.test(attr.value.replace(/[\s\u0000-\u0020]/g, ""))) bad.push(attr.value);
      }
      node.dispatchEvent(new MouseEvent("mouseover", {bubbles: true}));
      // Avoid following safe external links, but exercise any surviving event handlers.
      if (node.tagName !== "A") node.click();
    }
    return {executed: window.__executed, bad, count: roots.length, errors: window.__errors};
  });
  check(result.count > 0, "Markdown preview should exist");
  check(result.executed === 0, "Logged HTML executed: " + JSON.stringify(result));
  check(result.bad.length === 0, "Unsafe DOM reached preview: " + JSON.stringify(result.bad));
  check(result.errors.length === 0, "Unexpected browser error: " + JSON.stringify(result.errors));
}

const attacks = [
  '<img src=x onerror="window.__executed++">',
  '<script>window.__executed++</script>',
  '<p onclick="window.__executed++" onmouseover="window.__executed++">click</p>',
  '<svg onload="window.__executed++"><a href="javascript:window.__executed++">bad</a></svg>',
  '[bad](javascript:window.__executed++)',
  '<a href="jav&#x61;script:window.__executed++">bad</a>',
  '<a href="JaVaScRiPt:window.__executed++">bad</a>',
  '<a href="java&#x09;script:window.__executed++">bad</a>',
  '<a href="data:text/html,<script>alert(1)</script>">bad</a>',
  '<iframe srcdoc="<script>parent.__executed++</script>"></iframe>',
  '<form id="marked"><input name="parse"></form><p style="display:none" class="record" id="search" data-id="x">text</p>',
];
for (const payload of attacks) {
  const body = JSON.stringify({content: payload, reasoning_content: payload});
  await selectFixture({...seed, request_body: body, response_body: body});
  for (let index = 0; index < 4; index++) {
    await clickMd();
    await assertSafe();
  }
  for (let index = 0; index < 4; index++) await clickRaw();
  check(await page.locator(".hl-str").evaluateAll(nodes => nodes.every(n => n.style.display !== "none" && n.textContent.length > 0)), "RAW text must return");
  check(await page.locator(".hl-str").first().textContent() === JSON.stringify(payload), "RAW must preserve exact source");
}

const hostile = '\"><img src=x onerror="window.__executed++"><span data-injected="yes">';
const badId = 'record" onclick="window.__executed++';
const fields = {...seed, id: badId, model: hostile, method: hostile, endpoint: hostile,
  timestamp: "12345678901<img src", response_status: hostile, tokens_prompt: hostile,
  request_headers: {[hostile]: hostile}, response_headers: {[hostile]: hostile},
  request_body: JSON.stringify({[hostile]: hostile}), response_body: hostile, response_body_raw: hostile};
await selectFixture(fields);
const escaped = await page.evaluate(() => ({
  id: document.querySelector(".record").dataset.id,
  injected: document.querySelectorAll("img, [onclick], [data-injected]").length,
  text: document.querySelector("#detail").textContent,
  executed: window.__executed,
}));
check(escaped.id === badId, "ID must round-trip without breaking its attribute");
check(escaped.injected === 0 && escaped.executed === 0, "Record fields injected DOM/events");
check(escaped.text.includes(hostile), "Hostile record text must remain literal");
check(await page.evaluate(() => esc(0) === "0" && esc(false) === "false" && esc(null) === ""), "Escaping must preserve zero/false");

const markdown = "# Safe heading\n\n- First\n- Second\n\n**Bold** and *emphasis*.\n\n```js\nconst value = '<img onerror=alert(1)>';\n```\n\n[Safe link](https://example.com/docs)\n\n| A | B |\n| - | - |\n| 1 | 2 |";
await selectFixture({...seed, request_body: JSON.stringify({content: markdown}), response_body: JSON.stringify({reasoning_content: markdown})});
await clickMd();
await clickMd();
await assertSafe();
const formatted = await page.evaluate(() => ({
  headings: document.querySelectorAll(".md-render h1").length,
  lists: document.querySelectorAll(".md-render ul").length,
  code: [...document.querySelectorAll(".md-render pre code")].map(n => n.textContent),
  links: [...document.querySelectorAll(".md-render a")].map(n => n.getAttribute("href")),
  tables: document.querySelectorAll(".md-render table").length,
  resources: performance.getEntriesByType("resource").map(r => new URL(r.name).origin),
  zero: document.querySelector(".metrics").textContent,
}));
check(formatted.headings === 2 && formatted.lists === 2 && formatted.tables === 2, "Normal Markdown structure must render in both body sections");
check(formatted.code.length === 2 && formatted.code.every(x => x.includes("<img onerror=alert(1)>")), "Code must retain literal source");
check(formatted.links.length === 2 && formatted.links.every(x => x === "https://example.com/docs"), "Safe links must survive in both body sections");
check(formatted.zero.includes("0.000s") && formatted.zero.includes("0.0 tok/s"), "Zero metrics must remain visible");
if (!BASELINE) {
  check(formatted.resources.every(origin => origin === BASE_URL), "Cold load requested an external resource");
  const policy = await page.evaluate(async () => (await fetch("/")).headers.get("content-security-policy"));
  check(policy.includes("default-src 'self'") && policy.includes("'unsafe-inline'"), "Fixture must block external network without masking inline-handler bugs");
}

// A missing/throwing dependency must leave the exact RAW view intact.
for (const dependency of ["marked", "DOMPurify"]) {
  await selectFixture({...seed, request_body: JSON.stringify({content: attacks[0]})});
  await page.evaluate(name => { window.__savedDependency = window[name]; window[name] = undefined; }, dependency);
  await clickMd();
  check(await page.locator(".md-render").count() === 0, "Missing dependency must not create a preview");
  check(await page.getByRole("button", {name: "MD", exact: true}).count() === 1, "Missing dependency must preserve toggle state");
  check(await page.locator(".hl-str").textContent() === JSON.stringify(attacks[0]), "Missing dependency must preserve RAW");
  await page.evaluate(name => { window[name] = window.__savedDependency; }, dependency);
}
await selectFixture({...seed, request_body: JSON.stringify({content: markdown})});
await page.evaluate(() => {
  window.__savedSanitize = DOMPurify.sanitize;
  DOMPurify.sanitize = () => { throw new Error("synthetic sanitizer failure"); };
});
await clickMd();
check(await page.locator(".md-render").count() === 0, "Throwing sanitizer must fail closed");
await page.evaluate(() => { DOMPurify.sanitize = window.__savedSanitize; });

// Once records/assets are loaded, rendering needs no network at all.
await page.evaluate(() => {
  const meta = document.createElement("meta");
  meta.httpEquiv = "Content-Security-Policy";
  meta.content = "default-src 'none'; script-src 'unsafe-inline' 'unsafe-eval'; style-src 'unsafe-inline'";
  document.head.appendChild(meta);
});
await clickMd();
await assertSafe();
check(await page.locator(".md-render h1").textContent() === "Safe heading", "Offline toggle must render formatted content");
await clickRaw();
await clickMd();
await snapshot();
const proof = await screenshot({kind: "proof"});
return {note, assertions, payloads: attacks.length, keysAndSections: 4, offline: !BASELINE, package: "wheel assets byte-verified", proof};
