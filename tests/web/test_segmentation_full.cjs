'use strict';
const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const web=path.resolve(__dirname,'../../tools/segmentation_review/full_web');
const ui=require(path.join(web,'app.js'));

const modules=['skin','face','hair','eyes','mouth','outfit'].map((id,i)=>({id,name:id,color:'#123456',m2f:[i+1],see:[i]}));
const groupSet=prefix=>Object.fromEntries([...modules.map(m=>m.id),'all'].map(id=>[id,`${prefix}-${id}.png`]));
const prediction=(prefix,n)=>({masks:Array.from({length:n},(_,i)=>`${prefix}-raw${i}.png`),groups:groupSet(prefix),groups_colored:prefix+'-colored.png'});
const sample={id:'sample',title:'衣服样本',source_sha256:'a'.repeat(64),width:80,height:120,original:'original.png',views:{},
  fusion:{groups:groupSet('fusion'),groups_colored:'fusion-colored.png',objects:'fusion-objects.png',unknown:'fusion-unknown.png',conflict:'fusion-conflict.png',report_fingerprint:'fixed-fusion'}};
for(const view of ['white','original']) sample.views[view]={input:`${view}.png`,m2f:{projections:{score:prediction(`m2f-${view}-score`,12),nearest:prediction(`m2f-${view}-nearest`,12)}},see:{thresholds:Object.fromEntries(['0','1','2'].map(t=>[t,prediction(`see-${view}-${t}`,19)]))}};
const data={schema_version:1,cases:[sample],modules,see_labels:Array.from({length:19},(_,id)=>({id,name:'see '+id,color:'#778899'})),m2f_labels:Array.from({length:12},(_,id)=>({id,name:'m2f '+id,color:'#445566'}))};
const state=changes=>({...ui.defaultState(),...changes});

test('defaults focus outfit, matching score projection and policy threshold',()=>{
  assert.equal(ui.defaultState().part,'outfit');assert.equal(ui.defaultState().projection,'score');assert.equal(ui.defaultState().threshold,'1');
});
test('raw input and threshold select See and M2F while fusion remains fixed',()=>{
  const s=state({view:'original',threshold:'2'});
  assert.equal(ui.selectAssets(sample,'see',s,data).assets[0].path,'see-original-2-outfit.png');
  assert.equal(ui.selectAssets(sample,'m2f',s,data).assets[0].path,'m2f-original-score-outfit.png');
  const fusion=ui.selectAssets(sample,'fusion',s,data);
  assert.equal(fusion.assets[0].path,'fusion-outfit.png');assert.match(fusion.detail,/固定/);
  assert.equal(ui.selectAssets(sample,'fusion',state({view:'white',threshold:'0',projection:'nearest'}),data).assets[0].path,'fusion-outfit.png');
});
test('nearest is a genuine separate projection, not an alias for score',()=>{
  assert.equal(ui.selectAssets(sample,'m2f',state({projection:'nearest'}),data).assets[0].path,'m2f-white-nearest-outfit.png');
});
test('group overview prefers true grouped colored images',()=>{
  for(const model of ['see','m2f','fusion'])assert.match(ui.selectAssets(sample,model,state({part:'all'}),data).assets[0].path,/-colored\.png$/);
});
test('See garment detail uses actual channel and other panes explicitly show broad reference',()=>{
  const s=state({detail:'1'});
  assert.equal(ui.selectAssets(sample,'see',s,data).assets[0].path,'see-white-1-raw1.png');
  const m=ui.selectAssets(sample,'m2f',s,data),f=ui.selectAssets(sample,'fusion',s,data);
  assert.equal(m.assets[0].path,'m2f-white-score-raw11.png');assert.match(m.detail,/参照/);assert.match(m.note,/不等价/);
  assert.equal(f.assets[0].path,'fusion-outfit.png');assert.match(f.note,/不等价/);
});
test('ambiguous arm and leg channels do not silently become skin or outfit',()=>{
  for(const detail of ['12','14']) {
    const selection=ui.selectAssets(sample,'see',state({detail}),data);
    assert.match(selection.note,/裸肤|裸臂|粗类/);
    assert.equal(selection.assets[0].path,`see-white-1-raw${detail}.png`);
  }
});
test('objects stay See raw18 and fusion actual object mask; M2F has no object class',()=>{
  const s=state({part:'objects'});
  assert.equal(ui.selectAssets(sample,'see',s,data).assets[0].path,'see-white-1-raw18.png');
  assert.equal(ui.selectAssets(sample,'fusion',s,data).assets[0].path,'fusion-objects.png');
  const m=ui.selectAssets(sample,'m2f',s,data);assert.equal(m.available,false);assert.equal(m.reason,'no-object-class');assert.deepEqual(m.assets,[]);
});
test('unknown and conflict masks belong only to actual fusion',()=>{
  for(const part of ['unknown','conflict']) {
    assert.equal(ui.selectAssets(sample,'fusion',state({part}),data).assets[0].path,`fusion-${part}.png`);
    assert.equal(ui.selectAssets(sample,'see',state({part}),data).available,false);
    assert.equal(ui.selectAssets(sample,'m2f',state({part}),data).available,false);
  }
});
test('raw channel selections remain independent and fusion exposes only business groups',()=>{
  const s=state({raw:true,rawIds:{see:[0,18],m2f:[0,10],fusion:['hair','outfit']}});
  assert.deepEqual(ui.selectAssets(sample,'see',s,data).assets.map(a=>a.path),['see-white-1-raw0.png','see-white-1-raw18.png']);
  assert.deepEqual(ui.selectAssets(sample,'m2f',s,data).assets.map(a=>a.path),['m2f-white-score-raw0.png','m2f-white-score-raw10.png']);
  assert.deepEqual(ui.selectAssets(sample,'fusion',s,data).assets.map(a=>a.path),['fusion-hair.png','fusion-outfit.png']);
});
test('difference mode uses exact alpha subtraction recipes without pre-rendered diffs',()=>{
  const s=state({difference:true});assert.equal(ui.differenceSupported(sample,s,data),true);
  assert.deepEqual(ui.selectAssets(sample,'see',s,data).assets.map(a=>[a.path,a.subtract]),[['see-white-1-outfit.png','m2f-white-score-outfit.png']]);
  assert.deepEqual(ui.selectAssets(sample,'m2f',s,data).assets.map(a=>[a.path,a.subtract]),[['m2f-white-score-outfit.png','see-white-1-outfit.png']]);
  assert.deepEqual(ui.selectAssets(sample,'fusion',s,data).assets.map(a=>[a.path,a.subtract]),[['fusion-outfit.png','m2f-white-score-outfit.png'],['m2f-white-score-outfit.png','fusion-outfit.png']]);
});
test('differences are unavailable for non-equivalent raw/detail/object selections',()=>{
  for(const s of [state({raw:true}),state({detail:'11'}),state({part:'objects'}),state({part:'unknown'})])assert.equal(ui.differenceSupported(sample,s,data),false);
});
test('review export binds actual image, fusion report and all display parameters',()=>{
  const normalized=ui.normalizeData(data);
  const opinions={sample:{modules:{outfit:{choice:'fusion',evidence_context:{view:'white',see_threshold:1,projection:'score'}}},note:'衣领边界'}};
  const exp=ui.reviewExport(normalized,opinions,{view:'white',see_threshold:1,projection:'score',part:'outfit',detail:'all',difference:false});
  assert.equal(exp.cases[0].source_sha256,sample.source_sha256);assert.equal(exp.cases[0].fusion_report_fingerprint,'fixed-fusion');
  assert.equal(exp.cases[0].modules.outfit.choice,'fusion');assert.equal(exp.display_context.projection,'score');assert.match(exp.purpose,/不自动/);
});
test('invalid or repeated case identity is rejected',()=>{
  assert.throws(()=>ui.normalizeData({...data,cases:[sample,sample]}));assert.throws(()=>ui.normalizeData({...data,cases:[]}));
});
test('reopening raw channels restores checked selections instead of showing mismatched group masks',()=>{
  const s=state({raw:false,rawPanel:false,rawIds:{see:[11],m2f:[10],fusion:[]}});
  assert.deepEqual(ui.rawPanelState(s,true),{rawPanel:true,raw:true});
  assert.deepEqual(s.rawIds,{see:[11],m2f:[10],fusion:[]});
  assert.deepEqual(ui.rawPanelState({...s,raw:true,rawPanel:true},false),{rawPanel:false,raw:true});
  assert.deepEqual(ui.rawPanelState(state(),true),{rawPanel:true,raw:false});
});
test('difference explanations retain per-module semantic limitations rather than replacing them',()=>{
  const annotated={...data,modules:modules.map(m=>({...m,note:m.id==='skin'?'See仅显示neck颈部，不是全部裸肤。':'当前部位映射说明'}))};
  const text=ui.partExplanation(state({part:'skin',difference:true}),annotated);
  assert.match(text,/neck颈部/);assert.match(text,/待确认255/);assert.match(text,/不全部归因于See/);
  assert.match(ui.partExplanation(state({part:'all',difference:true}),annotated),/范围不等价/);
});
test('fusion removed-label recipe excludes unknown instead of calling it confirmed removal',()=>{
  const selection=ui.selectAssets(sample,'fusion',state({difference:true}),data);
  assert.deepEqual(selection.assets[1].exclude,['fusion-unknown.png']);
});
test('old opinions import matches full image identity, keeps current decisions and preserves source context',()=>{
  const existing={sample:{modules:{outfit:{choice:'fusion'},hair:{choice:'pending'}},note:'新备注'}};
  const imported={cases:[{id:'sample',source_sha256:sample.source_sha256,modules:{outfit:{choice:'see'},hair:{choice:'m2f',evidence_context:{view:'original',see_threshold:2}},eyes:{choice:'complementary'}},note:'旧备注'},
    {id:'sample',source_sha256:'b'.repeat(64),modules:{face:{choice:'see'}}},{id:'not-this-dataset',source_sha256:sample.source_sha256}]};
  const result=ui.importReview(ui.normalizeData(data),existing,imported);
  assert.equal(result.opinions.sample.modules.outfit.choice,'fusion');assert.equal(result.opinions.sample.modules.hair.choice,'m2f');
  assert.equal(result.opinions.sample.modules.hair.evidence_context.view,'original');assert.equal(result.opinions.sample.modules.eyes.choice,'complementary');
  assert.equal(result.opinions.sample.note,'新备注');assert.equal(result.stats.skipped_cases,2);assert.equal(result.stats.kept_choices,1);
  assert.equal(existing.sample.modules.hair.choice,'pending');
});
test('invalid opinion files and unsupported choices do not change user decisions',()=>{
  assert.throws(()=>ui.importReview(data,{},{}));
  const result=ui.importReview(data,{}, {cases:[{id:sample.id,source_sha256:sample.source_sha256,modules:{outfit:{choice:'guess'}},note:'导入备注'}]});
  assert.deepEqual(result.opinions.sample.modules,{});assert.equal(result.opinions.sample.note,'导入备注');
});
test('static page IDs match JavaScript and offline drawing avoids pixel reads',()=>{
  const html=fs.readFileSync(path.join(web,'index.html'),'utf8');
  const script=fs.readFileSync(path.join(web,'app.js'),'utf8');
  const ids=[...html.matchAll(/\bid="([^"]+)"/g)].map(x=>x[1]);
  const refs=[...script.matchAll(/\$\('([^']+)'\)/g)].map(x=>x[1]);
  assert.equal(new Set(ids).size,ids.length);
  for(const id of refs)assert(ids.includes(id),'Missing DOM ID '+id);
  for(const forbidden of ['getImageData','toDataURL','fetch('])assert(!script.includes(forbidden),'Offline canvas must avoid '+forbidden);
});

if(process.argv[2])test('real manifest covers three models and alpha recipes at every supported selection',()=>{
  const manifest=path.resolve(process.argv[2]),root=path.dirname(manifest);
  const d=ui.normalizeData(JSON.parse(fs.readFileSync(manifest,'utf8')));assert.equal(d.cases.length,12);
  const paths=new Set(),headers=new Map();let selections=0;
  function pngHeader(file) {
    if(headers.has(file))return headers.get(file);
    const fd=fs.openSync(file,'r'),header=Buffer.alloc(33);
    try{assert.equal(fs.readSync(fd,header,0,33,0),33);}finally{fs.closeSync(fd);}
    headers.set(file,header);return header;
  }
  function check(selection,c) {
    assert.equal(selection.available,true);
    for(const a of selection.assets)for(const p of [a.path,a.subtract].filter(Boolean)) {
      const file=path.resolve(root,p);assert(file.startsWith(root+path.sep));paths.add(file);
      const png=pngHeader(file);assert.equal(png.toString('hex',0,8),'89504e470d0a1a0a');assert.equal(png.readUInt32BE(16),c.width);assert.equal(png.readUInt32BE(20),c.height);assert.equal(png[25],6,'Transparent display assets must have RGBA alpha');
    }
    selections++;
  }
  for(const c of d.cases)for(const view of ['white','original'])for(const projection of ['score','nearest'])for(const threshold of ['0','1','2']) {
    for(const part of ['all',...d.modules.map(m=>m.id)])for(const model of ['see','m2f','fusion']) {
      check(ui.selectAssets(c,model,state({view,projection,threshold,part}),d),c);
      check(ui.selectAssets(c,model,state({view,projection,threshold,part,difference:true}),d),c);
    }
    for(const detail of ['1','4','6','10','11','13','15','12','14'])for(const model of ['see','m2f','fusion'])check(ui.selectAssets(c,model,state({view,projection,threshold,detail}),d),c);
    for(const model of ['see','fusion'])check(ui.selectAssets(c,model,state({view,projection,threshold,part:'objects'}),d),c);
    assert.equal(ui.selectAssets(c,'m2f',state({view,projection,threshold,part:'objects'}),d).reason,'no-object-class');
    for(const part of ['unknown','conflict']) {
      check(ui.selectAssets(c,'fusion',state({view,projection,threshold,part}),d),c);
      for(const model of ['see','m2f'])assert.equal(ui.selectAssets(c,model,state({view,projection,threshold,part}),d).available,false);
    }
    const rawIds={see:d.see_labels.map(x=>x.id),m2f:d.m2f_labels.map(x=>x.id),fusion:d.modules.map(x=>x.id)};
    for(const model of ['see','m2f','fusion'])check(ui.selectAssets(c,model,state({view,projection,threshold,raw:true,rawIds}),d),c);
    const fixed=ui.selectAssets(c,'fusion',state({view,projection,threshold}),d);
    assert.equal(fixed.assets[0].path,c.fusion.groups.outfit);
  }
  const imported={cases:d.cases.map(c=>({id:c.id,source_sha256:c.source_sha256,modules:{outfit:{choice:'see',evidence_context:{view:'original',see_threshold:0}}},note:'旧工具备注'}))};
  imported.cases.push({id:d.cases[0].id,source_sha256:'0'.repeat(64),modules:{outfit:{choice:'m2f'}}});
  const merged=ui.importReview(d,{},imported);assert.equal(merged.stats.matched_cases,12);assert.equal(merged.stats.skipped_cases,1);
  const exported=ui.reviewExport(d,merged.opinions,{view:'original',projection:'score',see_threshold:2});
  for(const c of exported.cases){assert(c.fusion_report_fingerprint);assert.equal(c.modules.outfit.choice,'see');assert.equal(c.modules.outfit.evidence_context.see_threshold,0);}
  console.log(`REAL_CONTRACT_CHECKED ${selections} selections / ${paths.size} PNG assets`);
});
