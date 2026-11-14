const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {test} = require('node:test');
const ui = require('../../tools/segmentation_review/web/app.js');

const modules = ['skin','face','hair','eyes','mouth','outfit'].map((id,index)=>({
  id,name:id,m2f:[index+1],see:[index],color:'#123456'
}));
const labels = Array.from({length:19},(_,id)=>({id,name:'channel '+id,color:'#8899aa'}));
const sample = {id:'sample',width:20,height:30,original:'original.png',source_sha256:'a'.repeat(64),views:{}};
for (const view of ['white','original']) {
  sample.views[view] = {
    m2f:{masks:Array.from({length:12},(_,id)=>`${view}/m2f-${id}.png`),groups:{hair:`${view}/m2f-hair.png`},groups_colored:`${view}/m2f-groups.png`},
    see:{thresholds:Object.fromEntries(['0','1','2'].map(threshold=>[threshold,{
      masks:Array.from({length:19},(_,id)=>`${view}/${threshold}/see-${id}.png`),
      groups:{hair:`${view}/${threshold}/see-hair.png`},groups_colored:`${view}/${threshold}/see-groups.png`
    }]))}
  };
}

test('object shortcut selects only the independent See objects channel at the requested input and threshold',()=>{
  for (const view of ['white','original']) for (const threshold of ['0','1','2']) {
    const selected=ui.maskAssets(sample,'see',view,threshold,'objects',null,modules,labels);
    assert.equal(selected.available,true);
    assert.equal(selected.assets.length,1);
    assert.equal(selected.assets[0].path,sample.views[view].see.thresholds[threshold].masks[18]);
  }
});

test('Mask2Former has no independent object class and must not substitute background or outfit',()=>{
  const selected=ui.maskAssets(sample,'m2f','white',0,'objects',null,modules,labels);
  assert.equal(selected.available,false);
  assert.equal(selected.reason,'no-object-class');
  assert.deepEqual(selected.assets,[]);
  assert.match(selected.message,/无独立物体类/);
});

test('raw channel selection keeps its existing meaning even after visiting the object shortcut',()=>{
  const see=ui.maskAssets(sample,'see','original',2,'objects',[0,18],modules,labels);
  assert.deepEqual(see.assets.map(asset=>asset.path),[
    sample.views.original.see.thresholds['2'].masks[0],sample.views.original.see.thresholds['2'].masks[18]
  ]);
  const m2f=ui.maskAssets(sample,'m2f','white',0,'objects',[0],modules,labels);
  assert.equal(m2f.available,true);
  assert.equal(m2f.assets[0].path,sample.views.white.m2f.masks[0]);
});

test('six module mapping and original group summaries remain unchanged',()=>{
  const before=JSON.stringify(modules);
  for (const model of ['see','m2f']) {
    const selected=ui.maskAssets(sample,model,'white',1,'hair',null,modules,labels);
    const expected=model==='see'?sample.views.white.see.thresholds['1'].groups.hair:sample.views.white.m2f.groups.hair;
    assert.equal(selected.assets[0].path,expected);
    const all=ui.maskAssets(sample,model,'white',1,'all',null,modules,labels);
    assert.equal(all.assets[0].path,model==='see'?sample.views.white.see.thresholds['1'].groups_colored:sample.views.white.m2f.groups_colored);
  }
  assert.equal(modules.length,6);assert.equal(JSON.stringify(modules),before);
});

test('object display context is exportable without inventing an object preference module',()=>{
  const data=ui.normalizeData({cases:[sample],modules,see_labels:labels});
  const opinions={sample:{modules:Object.fromEntries(modules.map(m=>[m.id,{choice:'complementary'}])),note:'物体单独观察'}};
  const exported=ui.reviewExport(data,opinions,{part:'objects',view:'original',see_threshold:2});
  assert.equal(exported.display_context.part,'objects');
  assert.deepEqual(Object.keys(exported.cases[0].modules),modules.map(m=>m.id));
  assert(!Object.hasOwn(exported.cases[0].modules,'objects'));
});

if (process.argv[2]) test('real exported objects: both inputs and all three thresholds keep exact raw18 paths',()=>{
  const manifest=path.resolve(process.argv[2]);const root=path.dirname(manifest);
  const data=ui.normalizeData(JSON.parse(fs.readFileSync(manifest,'utf8')));
  assert(data.cases.length>0);assert.equal(data.modules.length,6);
  let checked=0;
  for (const c of data.cases) for (const view of ['white','original']) for (const threshold of ['0','1','2']) {
    const result=ui.maskAssets(c,'see',view,threshold,'objects',null,data.modules,data.see_labels);
    assert.equal(result.available,true);assert.equal(result.assets.length,1);
    assert.equal(result.assets[0].path,c.views[view].see.thresholds[threshold].masks[18]);
    const file=path.resolve(root,result.assets[0].path);assert(file.startsWith(root+path.sep));
    const png=fs.readFileSync(file);assert.equal(png.toString('hex',0,8),'89504e470d0a1a0a');
    assert.equal(png.readUInt32BE(16),c.width);assert.equal(png.readUInt32BE(20),c.height);assert.equal(png[25],6);
    const m2f=ui.maskAssets(c,'m2f',view,threshold,'objects',null,data.modules,data.m2f_labels);
    assert.equal(m2f.reason,'no-object-class');assert.deepEqual(m2f.assets,[]);checked++;
  }
  assert.equal(checked,48);
});
