// Replay/state checks without a browser; visual layout still needs browser inspection.
'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const html = fs.readFileSync(path.join(__dirname, '../docs/router-replay.html'), 'utf8');
const payload = html.match(/<script id="replay-data" type="application\/json">([\s\S]*?)<\/script>/)[1];
const data = JSON.parse(payload);
const scripts = [...html.matchAll(/<script(?: [^>]*)?>([\s\S]*?)<\/script>/g)];
const script = scripts.at(-1)[1];
const elements = new Map();
for (const [, id] of html.matchAll(/\bid="([^"]+)"/g)) {
  assert(!elements.has(id), `Duplicate ID ${id}`);
  elements.set(id, {textContent:'',innerHTML:'',disabled:false,listeners:{},
    addEventListener(event, handler) { this.listeners[event] = handler; }});
}
elements.get('replay-data').textContent = payload;
elements.get('milestone').options = [...html.matchAll(/<option value="(\d+)">/g)].map((m,index)=>({value:m[1],index}));
const phases = [0,1,2,3].map(n=>({dataset:{phase:String(n)},attributes:{},listeners:{},
  setAttribute(k,v) { this.attributes[k]=v; },
  addEventListener(event,handler) { this.listeners[event]=handler; }}));
const root = elements.get('router-walkthrough');
root.querySelector = selector => {
  assert(selector.startsWith('#'));
  const element = elements.get(selector.slice(1));
  assert(element, `Missing element ${selector}`);
  return element;
};
root.querySelectorAll = selector => { assert.equal(selector,'[data-phase]'); return phases; };
const document = {getElementById:id=>elements.get(id)};
const context = vm.createContext({document,console});
vm.runInContext(script,context,{filename:'router-walkthrough.html'});
const element=id=>elements.get(id);
const click=id=>element(id).listeners.click();
const seek=index=>element('candidate-slider').listeners.input({target:{value:String(index)}});
const phase=n=>phases[n].listeners.click();

assert.equal(data.records.length,24);
assert.equal(data.operators.length,54);
assert.equal(data.budgets.length,8);
assert.equal(element('candidate-slider').value,17);
assert.match(element('story-title').textContent,/successful edit/);
assert.match(element('candidate-origin').textContent,/B4.*B5/);
assert.match(element('edits').innerHTML,/fc/);
assert.match(element('ranking').innerHTML,/<td>1<\/td><td>E6<\/td><td>B5<\/td>/);
phase(2);
assert.match(element('story-text').textContent,/lower KL makes this a successful edit/);
phase(3);
assert.match(element('story-text').textContent,/stored as E7/);
assert.equal(element('memory-title').textContent,'Successful-edit memory · 7 recorded');

// Every state must render, expose one active phase, and keep eight budget rows.
for(let i=0;i<data.records.length;i++) {
  seek(i);
  assert.match(element('measurement').textContent,/appears at step 3/);
  for(let p=0;p<4;p++) {
    phase(p);
    assert.equal(phases.filter(b=>b.attributes['aria-current']==='step').length,1);
    assert.equal((element('budgets').innerHTML.match(/<tr/g)||[]).length,8);
    assert(element('story-title').textContent);
    assert(!/undefined|NaN/.test(element('story-text').textContent));
    assert(!/undefined|NaN/.test(element('gates').innerHTML));
  }
}

seek(6); phase(2);
assert.match(element('story-text').textContent,/Mistakes tie/);
phase(3);
assert.match(element('edit-memory').innerHTML,/E1/);
assert.match(element('budgets').innerHTML,/updated/);
seek(8);
assert.match(element('story-title').textContent,/fourth proposal/);
seek(12); phase(2);
assert.match(element('story-text').textContent,/96 of 100.*cutoff of 21/);
assert.match(element('measurement').textContent,/not a final accuracy/);
phase(3);
assert.match(element('story-text').textContent,/No incumbent/);
seek(18);
assert.match(element('ranking').innerHTML,/<td>1<\/td><td>E5<\/td><td>B2<\/td>/);
phase(2);
assert.match(element('story-text').textContent,/does not improve/);
phase(3);
assert.match(element('story-text').textContent,/remains available/);

// Independently recover final incumbents from recorded costs and metrics.
const inc = data.budgets.map(()=>null);
const key = p=>p.join(',');
const lex = (a,b)=>{for(let i=0;i<a.length;i++)if(a[i]!==b[i])return a[i]-b[i];return 0;};
for(const r of data.records.filter(r=>r.complete))data.budgets.forEach((b,i)=>{
  if(r.cost.some((v,k)=>v>b[k]+1e-9))return;
  const p=inc[i];
  if(!p||r.errors<p.errors||(r.errors===p.errors&&(r.mean_kl<p.mean_kl||(r.mean_kl===p.mean_kl&&lex(r.policy,p.policy)<0))))inc[i]=r;
});
assert.deepEqual(inc.map(r=>r.policy),data.incumbents);
seek(23); phase(3);
assert.equal(element('next').disabled,true);
assert.match(element('story-text').textContent,/allowance is now exhausted/);
for(const r of inc)assert(element('budgets').innerHTML.includes('C'+String(r.id+1).padStart(2,'0')));
seek(0);
assert.equal(element('previous').disabled,true);
click('next');
assert.equal(phases[1].attributes['aria-current'],'step');
click('previous');
assert.equal(phases[0].attributes['aria-current'],'step');
click('previous'); // Disabled controls stay harmless if called programmatically.
assert.equal(element('candidate-slider').value,0);
element('milestone').listeners.change({target:{value:'17'}});
assert.equal(element('candidate-slider').value,17);
assert.equal(key(data.records[17].parent),key(data.records.find(r=>r.id===11).policy));
console.log('Detailed replay: all 96 states and primary controls passed.');

// The main page tells one continuous story; its three optional demonstrations
// must preserve exact edit compatibility and both illustrative resource costs.
const storyHtml=fs.readFileSync(path.join(__dirname,'../docs/router-walkthrough.html'),'utf8');
const storyElements=new Map();
function mockElement(dataset={}) {
  return {dataset,attributes:{},listeners:{},textContent:'',innerHTML:'',hidden:false,className:'',
    setAttribute(k,v){this.attributes[k]=v;},
    addEventListener(event,handler){this.listeners[event]=handler;},
    click(){this.listeners.click();},
    querySelector(selector){assert.equal(selector,'.symbol');return this.symbol??=(mockElement());}};
}
for(const [,id] of storyHtml.matchAll(/\bid="([^"]+)"/g)) {
  assert(!storyElements.has(id),`Duplicate story ID ${id}`);
  storyElements.set(id,mockElement());
}
for(const [,ids] of storyHtml.matchAll(/\baria-labelledby="([^"]+)"/g)) {
  for(const id of ids.split(' '))assert(storyElements.has(id),`Missing accessible label ${id}`);
}
const networks=[];
for(const [tag] of storyHtml.matchAll(/<div\b[^>]*\bdata-policy="[^"]+"[^>]*>/g)) {
  const attrs=Object.fromEntries([...tag.matchAll(/\b([\w-]+)="([^"]*)"/g)].map(m=>[m[1],m[2]]));
  const dataset={policy:attrs['data-policy'],edited:attrs['data-edited'],different:attrs['data-different'],layerIds:attrs['data-layer-ids']};
  const e=attrs.id?storyElements.get(attrs.id):mockElement();e.dataset=dataset;networks.push(e);
}
const storyPayload=storyHtml.match(/<script id="story-data" type="application\/json">([\s\S]*?)<\/script>/)[1];
const storyData=JSON.parse(storyPayload);
const toyStorage=policy=>policy.reduce((total,w)=>total+w/4,0);
const toyCompute=policy=>policy.reduce((total,w,i)=>total+w/4*storyData.macs[i],0);
const toyPolicies=[storyData.A,storyData.B,storyData.C];
for(let i=0;i<3;i++) {
  assert(toyStorage(toyPolicies[i])<=storyData.budgets[i]);
  assert.equal(toyCompute(toyPolicies[i]),[13,14,15][i]);
  // Higher-accuracy seeds must not already fit and replace smaller-budget seeds.
  if(i>0)assert(toyStorage(toyPolicies[i])>storyData.budgets[i-1]);
}
storyElements.get('story-data').textContent=storyPayload;
storyElements.get('local-outcome').hidden=true;
storyElements.get('transfer-outcome').hidden=true;
storyElements.get('router-story').querySelectorAll=selector=>{assert.equal(selector,'.network[data-policy]');return networks;};
const storyScript=[...storyHtml.matchAll(/<script(?: [^>]*)?>([\s\S]*?)<\/script>/g)].at(-1)[1];
const storyContext=vm.createContext({document:{getElementById:id=>{
  assert(storyElements.has(id),`Missing story element ${id}`);return storyElements.get(id);
}},console:{assert:(condition,message)=>assert(condition,message)}});
vm.runInContext(storyScript,storyContext,{filename:'router-walkthrough.html'});
const s=id=>storyElements.get(id);
const press=id=>s(id).listeners.click();
assert.equal((storyHtml.match(/class="chapter"/g)||[]).length,10);
assert.equal((storyHtml.match(/<button /g)||[]).length,3);
assert(!storyHtml.includes('<table'));
for(const e of networks) {
  assert.equal((e.innerHTML.match(/class="node /g)||[]).length,e.dataset.policy.split(',').length);
  assert.equal(e.attributes.role,'img');
  assert(!/undefined|NaN/.test(e.innerHTML));
}
for(const [id,policy] of [['context-source',storyData.A],['context-b',storyData.B],['context-c',storyData.C]]) {
  const e=s(id);
  assert.deepEqual(e.dataset.policy.split(',').map(Number),[0,1,4,5].map(i=>policy[i]));
  assert.match(e.attributes['aria-label'],/^L1: .*L2: .*L5: .*L6:/);
  assert(!/L3:|L4:/.test(e.attributes['aria-label']));
}
assert.equal((s('context-b').innerHTML.match(/ different/g)||[]).length,1);
assert.equal((s('context-c').innerHTML.match(/ different/g)||[]).length,2);
press('reveal-local');
assert.equal(s('local-outcome').hidden,false);
assert.equal(s('reveal-local').attributes['aria-expanded'],'true');
press('reveal-local');assert.equal(s('local-outcome').hidden,true);
press('toggle-mismatch');
assert.match(s('gate-message').textContent,/Blocked/);
assert.match(s('match-status').textContent,/must start at W8/);
assert.match(s('cost-status').textContent,/No candidate/);
assert.equal(s('toggle-mismatch').attributes['aria-pressed'],'true');
press('toggle-mismatch');
assert.match(s('gate-message').textContent,/All gates pass/);
assert.match(s('cost-status').textContent,/Size 9 ≤ 9 · BitOps proxy: 14 \(reported\)/);
assert.equal(s('toggle-mismatch').attributes['aria-pressed'],'false');
const originalB=s('transfer-policy').innerHTML;
press('toggle-mismatch');
assert.match(s('gate-message').textContent,/Blocked/);
press('apply-transfer');
assert.match(s('gate-message').textContent,/All gates pass/);
assert.equal(s('transfer-outcome').hidden,false);
assert.match(s('transfer-policy').attributes['aria-label'],/L3: W4A8, L4: W8A8/);
assert.match(s('transfer-policy').attributes['aria-label'],/L6: W8A8/);
press('apply-transfer');
assert.equal(s('transfer-outcome').hidden,true);
assert.equal(s('transfer-policy').innerHTML,originalB);
console.log('Scrolling story: ten chapters, exact edit/cost checks, accessible labels and all three demonstrations passed.');
