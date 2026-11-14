'use strict';
const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const web=path.resolve(__dirname,'../../tools/batch_acceptance/web');
const ui=require(path.join(web,'app.js'));

const modules=['skin','face','hair','eyes','mouth','outfit'].map((id,i)=>({id,name:id,color:'#4488cc',note:id+'说明'}));
const groups=p=>Object.fromEntries([...modules.map(x=>x.id),'all'].map(id=>[id,p+'-'+id+'.png']));
const item={id:'case_01',title:'样本',source_sha256:'a'.repeat(64),width:80,height:120,
  original:'assets/case/original.png',thumbnail:'assets/case/thumb.jpg',
  methods:Object.fromEntries(['l0','esrgan-l0'].map(m=>[m,{final:m+'-final.png',candidate:m+'-float-preview.png',restored:m+'-restored-preview.png',report:'../runs/'+m+'/report.json',report_fingerprint:m+'-fingerprint'}])),
  shared:{white:'white.png',alpha:'alpha.png',colored:'labels-colored.png',groups:groups('shared'),allowed:'allowed.png',protected:'protected.png',weight:'weight-alpha.png',objects:'objects.png',unknown:'unknown.png',conflict:'conflict.png',source_colored:'source-colored.png',evidence:{}},
  differences:{'1':'diff-1.png','4':'diff-4.png','8':'diff-8.png'},
  checks:{same_masks:true,same_weight:true,source_unchanged:true,protected_unchanged:true},
  trace:[{stage:'final',title:'最终结果',status:'complete',detail:'已保存',links:[{label:'报告',path:'../runs/report.json'}]}]};
for(const v of ['white','original'])item.shared.evidence[v]={m2f:{groups:groups('m2f-'+v),groups_colored:'m2f-'+v+'-colored.png'},see:{groups:groups('see-'+v),groups_colored:'see-'+v+'-colored.png',objects:'see-'+v+'-objects.png'}};
const data={schema_version:1,run_id:'run-one',parameters:{lambda:.0005,downscale:.6,sigma_at_2048:12,strength:1},modules,cases:[item]};
const state=change=>({...ui.defaultState(),...change});

test('default is three final-image columns, with optional difference hidden',()=>{
  const s=ui.defaultState();assert.equal(s.stage,'final');assert.equal(s.visible.difference,false);
  for(const id of ['original','l0','esrgan-l0'])assert.equal(s.visible[id],true);
});
test('switching away from a difference-only stage always keeps an original reference visible',()=>{
  const s=state({stage:'evidence',visible:{original:false,l0:false,'esrgan-l0':false,difference:true}});
  assert.deepEqual(ui.visibleColumns(s),['original']);
});
test('original peek is explicitly labelled and does not masquerade as a method result',()=>{
  const a=ui.selectPanel(item,'esrgan-l0',state({peekOriginal:true}),data);
  assert.equal(a.path,item.original);assert.match(a.title,/原图暂看/);
});
test('final uses saved method outputs and preserves original reference',()=>{
  for(const id of ['l0','esrgan-l0']){
    const a=ui.selectPanel(item,id,state(),data);assert.equal(a.path,item.methods[id].final);assert.equal(a.kind,'image');assert.match(a.note,/最终/);
  }
  assert.equal(ui.selectPanel(item,'original',state(),data).path,item.original);
});
test('float candidate and Gaussian preview are labelled as unfinished candidates',()=>{
  for(const variant of ['candidate','restored']){
    const a=ui.selectPanel(item,'l0',state({stage:'candidate',candidate:variant}),data);
    assert.equal(a.path,item.methods.l0[variant]);assert.match(a.note,/候选|混合/);
  }
});
test('evidence columns identify models rather than pretending to be denoising methods',()=>{
  const s=state({stage:'evidence',evidenceView:'original',region:'hair'});
  const m=ui.selectPanel(item,'l0',s,data),v=ui.selectPanel(item,'esrgan-l0',s,data);
  assert.equal(m.path,item.shared.evidence.original.m2f.groups.hair);assert.match(m.title,/Mask2Former/);
  assert.equal(v.path,item.shared.evidence.original.see.groups.hair);assert.match(v.title,/See-through/);
});
test('shared region is the same preserved mask for the two methods',()=>{
  const s=state({stage:'regions',region:'outfit'});
  for(const id of ['l0','esrgan-l0'])assert.equal(ui.selectPanel(item,id,s,data).path,item.shared.groups.outfit);
});
test('weight is a display of true shared weights, not a noise-confidence score',()=>{
  const a=ui.selectPanel(item,'l0',state({stage:'weight'}),data);
  assert.equal(a.path,item.shared.weight);assert.equal(a.kind,'mask');assert.match(a.note,/权重/);
  assert.equal(ui.selectPanel(item,'esrgan-l0',state({stage:'weight'}),data).path,item.shared.protected);
  const exact={...item,shared:{...item.shared,zero_weight:'zero-weight.png'}};
  assert.equal(ui.selectPanel(exact,'esrgan-l0',state({stage:'weight'}),data).path,'zero-weight.png');
});
test('absolute final difference selects the declared magnification, never fakes a gain',()=>{
  const a=ui.selectPanel(item,'difference',state({diffGain:'4'}),data);
  assert.equal(a.path,item.differences['4']);assert.match(a.title,/4/);assert.match(a.note,/差异/);
  assert.equal(ui.selectPanel(item,'difference',state({stage:'candidate'}),data).available,false);
});
test('M2F does not acquire a fake object channel in evidence view',()=>{
  const s=state({stage:'evidence',region:'objects'});
  assert.equal(ui.selectPanel(item,'l0',s,data).available,false);
  assert.equal(ui.selectPanel(item,'esrgan-l0',s,data).path,item.shared.evidence.white.see.objects);
});
test('check table distinguishes failures and absent information from passed checks',()=>{
  const rows=ui.checkRows({same_masks:true,same_weight:false,extra:{passed:false,detail:'不一致'},missing:null});
  assert.equal(rows.find(x=>x.key==='same_masks').status,'pass');
  assert.equal(rows.find(x=>x.key==='same_weight').status,'fail');
  assert.equal(rows.find(x=>x.key==='extra').status,'fail');
  assert.equal(rows.find(x=>x.key==='missing').status,'unknown');
  assert.equal(ui.checkRows({alpha:{passed:true,status:'not_present',detail:'原图无alpha'}})[0].status,'na');
});
test('review export binds run, image and both exact report fingerprints without approval',()=>{
  const opinions={case_01:{segmentation:'needs-fix',preference:'esrgan-l0',tags:['edge'],note:'脸部边缘漏选',context:{stage:'regions',zoom:1,center:[40,60]}}};
  const out=ui.reviewExport(data,opinions,{stage:'final',zoom:1});
  assert.equal(out.run_id,'run-one');assert.equal(out.cases[0].source_sha256,item.source_sha256);
  assert.equal(out.cases[0].report_fingerprints.l0,'l0-fingerprint');assert.equal(out.cases[0].review.note,opinions.case_01.note);
  assert.equal(out.automatic_quality_approval,false);
  assert.deepEqual(out.cases[0].review.context,opinions.case_01.context);
  opinions.case_01.context.center[0]=0;assert.equal(out.cases[0].review.context.center[0],40);
});
test('local-storage identity changes with run or input identity',()=>{
  assert.notEqual(ui.storageKey(data),ui.storageKey({...data,run_id:'run-two'}));
  assert.notEqual(ui.storageKey(data),ui.storageKey({...data,cases:[{...item,source_sha256:'b'.repeat(64)}]}));
});

function extensionFixture(){
  const hash=n=>n.toString(16).padStart(64,'0');
  const cases=Array.from({length:9},(_,index)=>({...item,id:'batch_'+String(index+1).padStart(4,'0'),
    source_sha256:hash(index+1),methods:{l0:{...item.methods.l0,report_fingerprint:hash(index+101)},
      'esrgan-l0':{...item.methods['esrgan-l0'],report_fingerprint:hash(index+201)}}}));
  const old={...data,cases:cases.slice(0,6)},saved=Object.fromEntries(old.cases.map(c=>[c.id,
    {segmentation:'ok',preference:'esrgan-l0',tags:['edge'],note:'旧意见 '+c.id,context:{stage:'final',center:[40,60]}}]));
  const descriptor={run_id:old.run_id,cases:old.cases.map(c=>({id:c.id,source_sha256:c.source_sha256,
    report_fingerprints:Object.fromEntries(['l0','esrgan-l0'].map(m=>[m,c.methods[m].report_fingerprint]))}))};
  return {next:{...data,cases,previous_reviews:[descriptor]},old,saved,descriptor};
}
function migrate(data,current,reader){assert.equal(typeof ui.migrateReviews,'function','Review migration is not implemented');return ui.migrateReviews(data,current,reader);}

test('adding three cases inherits only the six exact old identities without mutating saved opinions',()=>{
  const f=extensionFixture(),before=JSON.stringify(f.saved),reads=[];
  const result=migrate(f.next,{},key=>{reads.push(key);return JSON.stringify(f.saved);});
  assert.deepEqual(reads,[ui.storageKey(f.old)]);assert.notEqual(reads[0],ui.storageKey(f.next));
  assert.equal(result.stats.migrated_cases,6);assert.equal(Object.keys(result.opinions).length,6);
  assert.equal(result.opinions.batch_0001.note,f.saved.batch_0001.note);
  assert.equal(result.opinions.batch_0007,undefined);assert.equal(JSON.stringify(f.saved),before);
  result.opinions.batch_0001.context.center[0]=0;assert.equal(f.saved.batch_0001.context.center[0],40);
});
test('explicit old storage key takes precedence and current new-key decisions are never overwritten',()=>{
  const f=extensionFixture();f.next.previous_reviews=[{...f.descriptor,run_id:'not-the-original-run',storage_key:'exact-old-key'}];
  const current={batch_0001:{segmentation:'pending',preference:'pending',tags:[],note:'',context:{stage:'regions'}}};
  const snapshot=JSON.stringify(current),reads=[];
  const result=migrate(f.next,current,key=>{reads.push(key);return JSON.stringify(f.saved);});
  assert.deepEqual(reads,['exact-old-key']);assert.equal(result.stats.migrated_cases,5);
  assert.deepEqual(result.opinions.batch_0001,current.batch_0001);assert.equal(JSON.stringify(current),snapshot);
});
test('case ID, complete source SHA and both complete report fingerprints must all match',()=>{
  const f=extensionFixture();
  f.descriptor.cases[0].id='same-file-name-other-case';
  f.descriptor.cases[1].source_sha256=f.descriptor.cases[1].source_sha256.slice(0,12);
  f.descriptor.cases[2].report_fingerprints.l0='f'.repeat(64);
  delete f.descriptor.cases[3].report_fingerprints['esrgan-l0'];
  f.descriptor.cases[4].source_sha256='e'.repeat(64);
  const result=migrate(f.next,{},()=>JSON.stringify(f.saved));
  assert.deepEqual(Object.keys(result.opinions),['batch_0006']);assert.equal(result.stats.migrated_cases,1);
});
test('malformed or unavailable old stores do not discard current opinions or abort initialization',()=>{
  const f=extensionFixture(),current={batch_0009:{segmentation:'needs-fix',preference:'l0',tags:[],note:'新图'}};
  for(const reader of [()=>'{bad json',()=>{throw new Error('storage disabled');},()=>null]){
    const result=migrate(f.next,current,reader);assert.deepEqual(result.opinions,current);assert.equal(result.stats.migrated_cases,0);
  }
});
test('a descriptor for the current storage key is ignored rather than self-migrated',()=>{
  const f=extensionFixture();f.next.previous_reviews=[{storage_key:ui.storageKey(f.next),cases:f.descriptor.cases}];
  let reads=0;const result=migrate(f.next,{},()=>{reads++;return JSON.stringify(f.saved);});
  assert.equal(reads,0);assert.equal(result.stats.migrated_cases,0);
});
test('malformed prior descriptors do not interrupt a valid new nine-case manifest',()=>{
  const f=extensionFixture();f.next.previous_reviews=[null,{run_id:'bad',cases:[null]},
    {run_id:'bad',cases:[{id:'batch_0001'}]},...f.next.previous_reviews];
  const result=migrate(f.next,{},()=>JSON.stringify(f.saved));assert.equal(result.stats.migrated_cases,6);
});
test('an appended batch explains split producer records and links the aggregate audit',()=>{
  assert.equal(typeof ui.aggregationInfo,'function');
  const f=extensionFixture();f.next.aggregation={initial_cases:6,added_cases:3,
    manifest:'aggregate-manifest.json',validation:'aggregate-validation.json',original_review:'manifest.original-6.json',
    extension_manifest:'extensions/added-3/manifest.json',extension_validation:'extensions/added-3/validation.json'};
  const info=ui.aggregationInfo(f.next);assert.match(info.text,/9/);assert.match(info.text,/6/);assert.match(info.text,/3/);
  assert.equal(info.warning,false);assert.equal(info.links.length,5);
  assert(info.links.some(l=>l.path==='aggregate-validation.json'));
  assert.equal(ui.aggregationInfo(data),null);
});
test('case identity and grid errors are rejected',()=>{
  assert.throws(()=>ui.normalizeData({...data,cases:[item,item]}));
  assert.throws(()=>ui.normalizeData({...data,cases:[{...item,width:0}]}));
});
test('local trace links are supported while protocol execution is rejected',()=>{
  assert.equal(ui.localPath('../runs/report.json'),true);assert.equal(ui.localPath('javascript:alert(1)'),false);
  assert.equal(ui.localPath('https://example.org'),false);assert.equal(ui.localPath('//example.org/a'),false);
});
test('stage titles describe every processing stage',()=>{
  assert.deepEqual(ui.STAGES.map(x=>x.id),['input','evidence','regions','weight','candidate','final','checks']);
});
test('page control IDs match the renderer and offline drawing never reads local canvas pixels',()=>{
  const html=fs.readFileSync(path.join(web,'index.html'),'utf8'),js=fs.readFileSync(path.join(web,'app.js'),'utf8');
  const ids=new Set(Array.from(html.matchAll(/id="([^"]+)"/g),x=>x[1]));
  for(const [,id] of js.matchAll(/\$\('([^']+)'\)/g))assert(ids.has(id),id);
  assert(!/\b(?:fetch|getImageData|toDataURL)\s*\(/.test(js));
});

if(process.argv[2])test('real batch contract resolves all stages and saved method assets',()=>{
  const manifest=path.resolve(process.argv[2]),root=path.dirname(manifest);
  const d=ui.normalizeData(JSON.parse(fs.readFileSync(manifest,'utf8')));assert(d.cases.length>0);
  if(typeof d.summary.cases==='number')assert.equal(d.cases.length,d.summary.cases);
  const paths=new Set(),headers=new Map();let selections=0,links=0;
  function check(selection,c){
    if(!selection.available || !selection.path)return;
    const file=path.resolve(root,selection.path);assert(fs.existsSync(file),file);paths.add(file);
    let png=headers.get(file);
    if(!png){const fd=fs.openSync(file,'r');png=Buffer.alloc(33);try{assert.equal(fs.readSync(fd,png,0,33,0),33);}finally{fs.closeSync(fd);}headers.set(file,png);}
    assert.equal(png.toString('hex',0,8),'89504e470d0a1a0a');assert.equal(png.readUInt32BE(16),c.width);assert.equal(png.readUInt32BE(20),c.height);
    if(selection.kind==='mask')assert.equal(png[25],6,'Display masks need actual RGBA alpha');
  }
  for(const c of d.cases){
    assert.equal(c.checks.same_masks,true);assert.equal(c.checks.same_weight,true);
    for(const stage of ui.STAGES.map(x=>x.id))for(const column of ['original','l0','esrgan-l0']){
      for(const region of ['colored','pre-sam','post-sam','allowed','active',...d.modules.map(x=>x.id),'objects','unknown','conflict','protected','source','weight']){
        const a=ui.selectPanel(c,column,state({stage,region}),d);
        check(a,c);
        selections++;
      }
    }
    for(const candidate of ['candidate','restored'])for(const column of ['l0','esrgan-l0']){const a=ui.selectPanel(c,column,state({stage:'candidate',candidate}),d);assert.equal(a.path,c.methods[column][candidate]);check(a,c);}
    for(const diffGain of ['1','4','8']){const a=ui.selectPanel(c,'difference',state({diffGain}),d);assert(a.available);check(a,c);}
    for(const evidenceView of ['white','original'])for(const region of ['colored',...d.modules.map(x=>x.id),'objects'])for(const column of ['l0','esrgan-l0'])check(ui.selectPanel(c,column,state({stage:'evidence',region,evidenceView}),d),c);
    for(const trace of c.trace || [])for(const link of trace.links || []){assert(ui.localPath(link.path));assert(fs.existsSync(path.resolve(root,link.path)),link.path);links++;}
    for(const m of ['l0','esrgan-l0']){assert(ui.localPath(c.methods[m].report));assert(fs.existsSync(path.resolve(root,c.methods[m].report)));}
  }
  if(Array.isArray(d.previous_reviews) && d.previous_reviews.length){
    const stores=new Map();
    for(const previous of d.previous_reviews){
      const key=previous.storage_key || ui.storageKey(previous);
      stores.set(key,JSON.stringify(Object.fromEntries(previous.cases.map(c=>[c.id,
        {segmentation:'ok',preference:'l0',tags:[],note:'原图验收意见 '+c.id}]))));
    }
    const expected=new Set(d.previous_reviews.flatMap(p=>p.cases.map(c=>c.id)));
    const migrated=ui.migrateReviews(d,{},key=>stores.get(key) || null);
    assert.equal(migrated.stats.migrated_cases,expected.size,'All preserved original case identities must remain migratable');
    for(const id of expected)assert.match(migrated.opinions[id].note,/原图验收意见/);
    const first=[...expected][0],existing={[first]:{segmentation:'needs-fix',preference:'esrgan-l0',tags:[],note:'新增后已有判断'}};
    const preserved=ui.migrateReviews(d,existing,key=>stores.get(key) || null);
    assert.equal(preserved.opinions[first].note,'新增后已有判断');
    console.log(`BATCH_PREVIOUS_REVIEWS_CHECKED ${expected.size} preserved case identities`);
  }
  const aggregation=ui.aggregationInfo(d);
  if(aggregation){assert.equal(aggregation.warning,false);for(const link of aggregation.links)assert(fs.existsSync(path.resolve(root,link.path)),link.path);}
  console.log(`BATCH_REAL_CONTRACT ${selections} selections / ${paths.size} image assets / ${links} trace links`);
});
