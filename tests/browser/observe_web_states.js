const findings = [];
const allContrast = [];
async function auditContrast(label) {
  const pairs = await contrastAudit();
  check(pairs.length > 0, label + " has measurable text");
  check(pairs.every(p => p.ratio >= 4.5), label + " contrast: " + JSON.stringify(pairs.filter(p => p.ratio < 4.5)));
  allContrast.push(...pairs.map(p => ({...p,state:label})));
}
let assertions = 0;
function check(value, message) {
  assertions++;
  if (!value) throw new Error(message);
}
async function config(value) {
  return page.evaluate(async value => (await fetch('/__test/config', {method:'POST', headers:{'Content-Type':'application/json'},body:JSON.stringify(value)})).json(), value);
}
async function release(name) {
  return page.evaluate(async name => (await fetch('/__test/release', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name})})).json(), name);
}
async function text(selector) { return page.locator(selector).first().textContent(); }
async function waitText(selector, expected) {
  await page.waitForFunction(({selector, expected}) => document.querySelector(selector)?.textContent.includes(expected), {selector, expected});
}
async function inspect() { await snapshot({interactive:true}); await snapshot(); }
async function act(selector, method='click', value) {
  await inspect();
  if (method === 'fill') await page.locator(selector).first().fill(value);
  else if (method === 'press') await page.locator(selector).first().press(value);
  else await page.locator(selector).first().click();
  await snapshot({diff:true});
}
await page.setViewportSize({width:1280,height:900});
await page.goto(base);
await inspect();
if (baseline) {
  check(await page.locator('#record-count').count() === 1, 'baseline missing visible API total and request-state region');
  return {assertions};
}
await waitText('#record-count', '100 of 4,213');
await page.evaluate(()=>{ window.observeErrors=[]; addEventListener('error', event=>window.observeErrors.push(event.message)); addEventListener('unhandledrejection', event=>window.observeErrors.push(String(event.reason))); });
check(await page.locator('.record').count() === 100, 'first page has 100 rows');
check((await text('.record .time')).length === 8, 'today shows time');
check((await page.locator('.record .time').nth(1).textContent()).includes('2025-01-02'), 'old records show date');
// Actual backend smoke, using a seeded on-disk record instead of intercepted responses.
await config({mode:'live'});
await act('#refresh-btn');
await waitText('#record-count','1 of 1');
await act('.record');
await waitText('.timestamp','T');
check(await page.locator('.section-header').count() >= 4, 'live API opens actual saved record');
await config({mode:'synthetic'});
// Every operation has visible loading and retry; malformed JSON/network errors included.
for (const failure of [{status:503},{malformed:true},{disconnect:true}]) {
  const hold = 'initial-' + assertions;
  await config({rules:[{path:'/api/records',hold,...failure}]});
  await act('#refresh-btn');
  await waitText('#list-status','Loading requests');
  check(await page.locator('#list-pane').getAttribute('aria-busy') === 'true','initial busy');
  await auditContrast('loading');
  await release(hold);
  await waitText('#list-status','Could not load');
  check(await page.locator('#list-retry').isVisible(),'initial retry visible');
  await auditContrast('list error');
  await act('#list-retry','press','Enter');
  await waitText('#record-count','100 of 4,213');
  check(await page.evaluate(()=>document.activeElement.classList.contains('record')),'initial retry returns focus to records');
}
await config({rules:[{path:'/api/records',query:'empty',hold:'search-fail',status:500}]});
await act('#search','fill','empty');
await waitText('#list-status','Searching requests');
await release('search-fail');
await waitText('#list-status','Could not search');
await act('#list-retry','press','Enter');
await waitText('#list-status','No matches');
check((await text('#record-count')) === '0 matches','empty search count');
await config({rules:[{path:'/api/records',query:'',json:{items:[],total:0}}]});
await act('#search','fill','');
await waitText('#list-status','No requests recorded yet');
check((await text('#record-count')) === '0 of 0 requests','empty records count');
await act('#search','fill','short');
await waitText('#record-count','37 matches');
await act('#search','fill','');
await waitText('#record-count','100 of 4,213');
await config({rules:[{path:'/api/records',offset:100,hold:'page-fail',status:500}]});
await act('#load-more-btn','press','Enter');
await waitText('#list-status','Loading more');
check(await page.locator('.record').count() ===100,'append keeps current rows');
await auditContrast('disabled pagination');
await release('page-fail');
await waitText('#list-status','Could not load more');
check(await page.evaluate(()=>document.activeElement.id)==='list-retry','page failure moves keyboard focus to retry');
await act('#list-retry','press','Enter');
await waitText('#record-count','130 of 4,213');
await act('#load-more-btn','press','Enter');
await page.waitForFunction(()=>document.querySelector('#load-more-btn').hidden);
check(await page.locator('.record').count() ===130,'empty final page retains rows and stops pagination');
// A finished request must not steal focus after the user moved elsewhere.
await act('#refresh-btn');
await waitText('#record-count','100 of 4,213');
await config({rules:[{path:'/api/records',offset:100,hold:'focus-page'}]});
await act('#load-more-btn','press','Enter');
await act('#search');
await release('focus-page');
await waitText('#record-count','130 of 4,213');
check(await page.evaluate(()=>document.activeElement.id)==='search','page success does not steal moved focus');
// Duplicate responses advance the server offset, not the deduplicated row count.
await act('#refresh-btn');
await waitText('#record-count','100 of 4,213');
const duplicate = await page.evaluate(async()=> (await (await fetch('/api/records?offset=0&limit=1')).json()).items[0]);
const nextRecord = await page.evaluate(async()=> (await (await fetch('/api/records?offset=101&limit=1')).json()).items[0]);
await config({rules:[{path:'/api/records',offset:100,json:{items:[duplicate],total:102}},{path:'/api/records',offset:101,json:{items:[nextRecord],total:102}}]});
await act('#load-more-btn','press','Enter');
await waitText('#record-count','100 of 102');
await act('#load-more-btn','press','Enter');
await waitText('#record-count','101 of 102');
check(await page.locator('#load-more-btn').isHidden(),'raw offset reaches total despite duplicate display');
await config({rules:[{path:'/api/records',query:'"<img src=x>"',json:{items:[],total:0}}]});
await act('#search','fill','"<img src=x>"');
await waitText('#list-status','No matches');
check(await page.locator('#list-status img').count()===0,'query is literal text');
check((await text('#list-status')).includes('<img src=x>'),'query preserved exactly');
await act('#search','fill','');
await waitText('#record-count','100 of 4,213');
// Detail failures and retry, last-clicked selection, early search invalidation and stale page.
for (const failure of [{status:404},{malformed:true},{disconnect:true}]) {
  const hold='detail-'+assertions;
  await config({rules:[{path:'/api/records/record-1',hold,...failure}]});
  await act('.record[data-id="record-1"]');
  await waitText('#detail','Loading request details');
  check(await page.locator('#detail').getAttribute('aria-busy')==='true','detail busy');
  await release(hold);
  await waitText('#detail','Could not load request details');
  await act('#detail .retry','press','Enter');
  await waitText('.timestamp','2025-01-02T12:34:56');
}
// Keep the actual HTTP response alive despite cancellation to exercise the version
// fallback. Observe showRecord completion, including its render/catch/finally work.
const raceRecords = await page.evaluate(async () => {
  const records = await Promise.all([1, 2].map(async id =>
    (await fetch('/api/records/record-' + id)).json()));
  records[0].model = 'older-detail-marker';
  records[1].model = 'latest-detail-marker';
  window.raceCompleted = [];
  window.raceOriginalFetch = window.fetch;
  window.raceOriginalShowRecord = showRecord;
  window.fetch = (input, options) => window.raceOriginalFetch(input,
    String(input).includes('/api/records/record-1') ? {...options, signal:undefined} : options);
  showRecord = async id => {
    try { return await window.raceOriginalShowRecord(id); }
    finally { window.raceCompleted.push(id); }
  };
  return records;
});
await config({seen:[],rules:[
  {path:'/api/records/record-1',hold:'old-detail',json:raceRecords[0]},
  {path:'/api/records/record-2',json:raceRecords[1]}
]});
await act('.record[data-id="record-1"]');
await waitText('#detail','Loading request details');
await page.waitForFunction(async()=> (await (await fetch('/__test/seen')).json()).some(r=>r.path==='/api/records/record-1'));
await act('.record[data-id="record-2"]');
await waitText('#detail h2','latest-detail-marker');
await page.waitForFunction(()=>window.raceCompleted.includes('record-2'));
await release('old-detail');
await page.waitForFunction(()=>window.raceCompleted.includes('record-1'));
check((await text('#detail h2')).includes('latest-detail-marker'), 'late non-aborting response cannot overwrite latest detail content');
check(!(await text('#detail')).includes('older-detail-marker'), 'older detail marker remains absent after stale continuation settles');
check(await page.locator('.record[data-id="record-2"]').getAttribute('aria-pressed')==='true','new detail stays selected');
check((await page.evaluate(()=>location.hash))==='#record=record-2','selection in URL');
await page.evaluate(()=>{
  window.fetch = window.raceOriginalFetch;
  showRecord = window.raceOriginalShowRecord;
});
await auditContrast('selected');
await page.locator('.record[data-id="record-2"]').hover();
await auditContrast('selected hover');
await page.locator('#refresh-btn').hover();
await auditContrast('button hover');
await config({rules:[{path:'/api/records',query:'old',hold:'old-search',status:500}]});
await act('#search','fill','old');
await page.waitForFunction(async()=> (await (await fetch('/__test/seen')).json()).some(r=>r.query==='old'));
await act('#search','fill','short');
await release('old-search');
await waitText('#record-count','37 matches');
check(!(await text('#list-status')).includes('Could not'),'stale failure ignored');
await act('#search','fill','');
await waitText('#record-count','100 of 4,213');
await config({rules:[{path:'/api/records',offset:100,hold:'old-page'}]});
await act('#load-more-btn','press','Enter');
await act('#search','fill','short');
await release('old-page');
await waitText('#record-count','37 matches');
check(await page.locator('.record').count()===37,'old page cannot append to search');
// Keyboard flows, roving focus and details disclosure.
await act('#search','press','Tab'); // Refresh
await page.keyboard.press('Tab'); // one row
check(await page.evaluate(()=>document.activeElement.classList.contains('record')),'Tab enters list');
await page.keyboard.press('Home');
await page.keyboard.press('ArrowDown');
check(await page.evaluate(()=>document.activeElement.dataset.id)==='record-1','arrow moves focus');
await page.keyboard.press('Enter');
await waitText('.timestamp','2025-01-02');
check(await page.evaluate(()=>document.activeElement.dataset.id)==='record-1','selection preserves focus');
await page.keyboard.press('Tab');
check(await page.evaluate(()=>document.activeElement.classList.contains('section-header')),'Tab exits roving list to details');
await page.keyboard.press('Space');
check(await page.locator('.section-header').first().getAttribute('aria-expanded')==='false','keyboard section collapses');
check(await page.evaluate(()=>getComputedStyle(document.activeElement).outlineStyle)==='solid','keyboard focus outline visible');
await page.keyboard.press('Enter');
await page.keyboard.press('Tab');
check(await page.evaluate(()=>document.activeElement.classList.contains('jt-toggle')),'JSON control keyboard reachable');
await page.keyboard.press('Enter');
check(await page.locator('.jt-toggle').first().getAttribute('aria-expanded')==='false','JSON collapses');
await page.keyboard.press('Space');
check((await page.evaluate(()=>window.observeErrors)).length===0,'no unhandled application errors during failure/race/keyboard cases');
await act('.section-header:has-text("Request Headers")','press','Enter');
check(await page.locator('.section-header:has-text("Request Headers")').getAttribute('aria-expanded')==='true','header section keyboard opens');
// Hash links outside list, reload, history, clear and invalid input.
await page.evaluate(()=>{location.hash='#record=record-777';});
await waitText('.timestamp','invalid time');
await page.reload();
await waitText('.timestamp','invalid time');
check((await text('.detail h2')).includes('model-long'),'off-page link restored');
await page.evaluate(()=>{location.hash='#record=record-2';});
await waitText('.timestamp','2025-01-02');
await page.goBack();
await waitText('.timestamp','invalid time');
await page.goForward();
await waitText('.timestamp','2025-01-02');
await page.evaluate(()=>{location.hash='';});
await waitText('#detail','Select a request');
await page.evaluate(()=>{location.hash='#record=%ZZ';});
await waitText('#detail','link is invalid');
await act('#detail .retry','press','Enter');
check((await page.evaluate(()=>location.hash))==='','invalid link can be cleared');
await page.evaluate(()=>{location.hash='#record=missing';});
await waitText('#detail','Could not load');
// Desktop and mobile rendering/contrast, including controls, syntax and benign markdown.
await page.evaluate(()=>{location.hash='#record=record-777';});
await waitText('.timestamp','invalid time');
await page.waitForFunction(()=>typeof marked!=='undefined');
await act('.md-toggle','press','Enter');
await page.waitForFunction(()=>document.querySelector('.md-render h1'));
check(await page.locator('.md-render h1').count()===1,'normal markdown heading visible');
async function contrastAudit() {
  return page.evaluate(()=>{
    const rgb=s=>(s.match(/[\d.]+/g)||[]).map(Number);
    const luminance=c=>c.slice(0,3).map(x=>x/255).map(x=>x<=0.04045?x/12.92:((x+0.055)/1.055)**2.4).reduce((sum,x,i)=>sum+x*[0.2126,0.7152,0.0722][i],0);
    const pairs=[];
    for(const node of document.querySelectorAll('body *')) {
      if(![...node.childNodes].some(n=>n.nodeType===Node.TEXT_NODE&&n.textContent.trim())) continue;
      const rect=node.getBoundingClientRect();if(!rect.width||!rect.height)continue;
      const style=getComputedStyle(node);if(style.visibility==='hidden')continue;
      let bgNode=node,bg;
      while(bgNode){bg=rgb(getComputedStyle(bgNode).backgroundColor);if(bg.length===3||bg[3]>0)break;bgNode=bgNode.parentElement;}
      if(!bgNode) bg=[26,23,20];
      const fg=rgb(style.color), a=luminance(fg),b=luminance(bg),ratio=(Math.max(a,b)+0.05)/(Math.min(a,b)+0.05);
      pairs.push({element:node.tagName+'.'+node.className,ratio,foreground:style.color,background:bg});
    }
    const input=document.querySelector('#search'),placeholder=getComputedStyle(input,'::placeholder');
    const a=luminance(rgb(placeholder.color)),b=luminance(rgb(getComputedStyle(input).backgroundColor));
    pairs.push({element:'search placeholder',ratio:(Math.max(a,b)+0.05)/(Math.min(a,b)+0.05)});
    return pairs;
  });
}
const desktopPairs=await contrastAudit();
check(desktopPairs.length>30,'audited actual rendered text classes');
check(desktopPairs.every(p=>p.ratio>=4.5),'desktop contrast failures: '+JSON.stringify(desktopPairs.filter(p=>p.ratio<4.5)));
await page.setViewportSize({width:375,height:812});
await inspect();
check(await page.evaluate(()=>document.documentElement.scrollWidth<=document.documentElement.clientWidth),'no mobile page overflow');
check(await page.locator('#search').isVisible(),'mobile search reachable');
check(await page.locator('.detail h2').isVisible(),'mobile details reachable');
const mobilePairs=await contrastAudit();
check(mobilePairs.every(p=>p.ratio>=4.5),'mobile contrast failures: '+JSON.stringify(mobilePairs.filter(p=>p.ratio<4.5)));
const longMobileProof=await screenshot({kind:'proof'});
await page.evaluate(()=>{location.hash='#record=record-2';});
await waitText('.timestamp','2025-01-02');
await act('.md-toggle','press','Enter');
await page.waitForFunction(()=>document.querySelector('.md-render h1'));
await page.locator('.md-render').first().scrollIntoViewIfNeeded();
await inspect();
check(await page.evaluate(()=>document.documentElement.scrollWidth<=document.documentElement.clientWidth),'normal mobile markdown has no page overflow');
const mobileProof=await screenshot({kind:'proof'});
await page.setViewportSize({width:1280,height:900});
await inspect();
await page.evaluate(()=>{document.getElementById('detail').scrollTop=0;});
await inspect();
const desktopProof=await screenshot({kind:'proof'});
allContrast.push(...desktopPairs,...mobilePairs);
return {assertions,minContrast:Math.min(...allContrast.map(p=>p.ratio)),contrastSamples:allContrast.length,mobileWidth:375,longMobileProof,mobileProof,desktopProof,findings};
