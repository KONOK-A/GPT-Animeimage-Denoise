(function(){
  'use strict';
  const COLUMNS=['original','l0','esrgan-l0','difference'];
  const NAMES={original:'原图',l0:'纯 L0','esrgan-l0':'ESRGAN + L0',difference:'最终算法差异'};
  const STAGES=[
    {id:'input',name:'输入',note:'核对方向、尺寸与实际分析输入。源文件保持原样，页面使用方向归一后的副本。'},
    {id:'evidence',name:'模型证据',note:'这里显示分区模型的独立证据，不是两种降噪算法的结果。See通道可重叠，皮肤映射仅含颈部。'},
    {id:'regions',name:'融合分区',note:'两种算法共用同一份融合标签与允许范围。模块归属不是鱼鳞噪点识别图，也不等于最终混合权重。'},
    {id:'weight',name:'实际权重',note:'实际使用float32 weight.npy。PNG只是显示预览；羽化和线条保护会使允许范围中的部分像素权重为0。'},
    {id:'candidate',name:'降噪候选',note:'候选与Gaussian恢复图都尚未按蒙版融合，属于整图阶段预览；最终仅由保存的权重决定写回范围。'},
    {id:'final',name:'最终出图',note:'三列为同尺寸原图和两种算法的最终结果。默认处理皮肤、面部、头发；眼睛、嘴巴、服饰和背景通常受保护。'},
    {id:'checks',name:'检查记录',note:'继续查看最终图，同时核对输入、选区与保护像素记录。工程检查通过不等于视觉质量已验收。'}
  ];
  const MASK_COLORS={allowed:'#37b89c',protected:'#a69acb',weight:'#4f9fe8',objects:'#f17d5b',unknown:'#edbe48',conflict:'#ee955d'};
  const CHECK_NAMES={same_masks:'两方法分区一致',same_weight:'两方法浮点权重一致',source_unchanged:'输入文件未改动',protected_unchanged:'保护像素未改变',size_preserved:'尺寸保持',alpha_preserved:'Alpha保持',icc_preserved:'ICC保持'};
  const TAGS=[['missing','漏选'],['extra','多选'],['edge','边缘'],['detail','细节损伤'],['weak','效果过弱'],['color','色偏'],['other','其他']];
  const DEFAULT_REVIEW={segmentation:'pending',preference:'pending',tags:[],note:''};
  function defaultState(){return {index:0,stage:'final',region:'colored',evidenceView:'white',candidate:'candidate',diffGain:'8',mode:'overlay',opacity:.55,peekOriginal:false,zoom:1,centerX:0,centerY:0,fit:true,hover:null,request:0,visible:{original:true,l0:true,'esrgan-l0':true,difference:false}};}
  function visibleColumns(s){const columns=COLUMNS.filter(id=>s.visible[id] && (id!=='difference' || ['final','checks'].includes(s.stage)));return columns.length?columns:['original'];}
  function normalizeData(data){
    if(!data || !Array.isArray(data.cases) || !data.cases.length || !data.run_id)throw new Error('缺少本次运行数据。请保持 manifest.js、assets 与页面的相对位置。');
    const ids=new Set();
    for(const c of data.cases){if(!c.id || ids.has(c.id) || !c.original || !c.source_sha256 || !(c.width>0) || !(c.height>0))throw new Error('样本身份或尺寸无效。');ids.add(c.id);}
    return {...data,modules:data.modules || [],parameters:data.parameters || {},summary:data.summary || {}};
  }
  function localPath(value){return typeof value==='string' && Boolean(value) && !/^[a-z][a-z0-9+.-]*:/i.test(value) && !value.startsWith('//') && !value.startsWith('\\\\');}
  function image(path,title,note,extra={}){return {available:Boolean(path),path,kind:'image',title,note,...extra};}
  function mask(path,title,note,color){return {available:Boolean(path),path,kind:'mask',title,note,color};}
  function moduleOf(data,id){return data.modules.find(x=>x.id===id);}
  function selectPanel(c,column,s,data){
    const shared=c.shared || {},method=c.methods && c.methods[column],part=s.region;
    if(s.peekOriginal)return image(c.original,'原图暂看 / '+NAMES[column],'当前临时显示输入原图；点击返回阶段结果继续比较。');
    if(column==='difference'){
      if(!['final','checks'].includes(s.stage))return {available:false,reason:'stage',title:'差异仅供最终图比较',note:'切换到最终出图或检查记录查看。'};
      const d=c.differences && c.differences[String(s.diffGain)];
      return image(typeof d==='string'?d:d && d.path,`两算法最终差异 ×${s.diffGain}`,'实际绝对差异；倍率仅帮助看清，不是噪点程度或质量分数。');
    }
    if(column==='original')return image(c.original,'原图','方向归一后的输入副本，不是降噪结果。');
    if(s.stage==='input')return column==='l0'?image(shared.white,'SkyTNT 白底分析图','实际送入白底分区分支的分析输入。'):
      image(shared.alpha || shared.input_original || c.original,shared.alpha?'SkyTNT 前景 alpha':'原图分析参照',shared.alpha?'软前景灰度预览，不直接等于最终人物选区。':'原图分支实际分析输入；原alpha可先合白。');
    if(s.stage==='evidence'){
      const model=column==='l0'?'m2f':'see',p=shared.evidence && shared.evidence[s.evidenceView] && shared.evidence[s.evidenceView][model];
      const title=model==='m2f'?'Mask2Former v4 证据':'See-through 证据';
      if(!p)return {available:false,title,note:'本次未导出该输入下的模型证据。'};
      if(part==='objects')return model==='m2f'?{available:false,title,note:'M2F没有独立物体类别，原图作为参照。'}:mask(p.objects,title+' / 原始物体','独立原始候选，不等于融合后的确认外物层。',MASK_COLORS.objects);
      if(part==='colored')return mask(p.groups_colored,title+' / 分组总览',p.note || '独立模型的映射分组；没有在此视图中混合两种算法。',null);
      const m=moduleOf(data,part);
      return mask(p.groups && p.groups[part],title+' / '+(m && m.name || part),(m && m.note || '')+' '+(p.note || ''),m && m.color);
    }
    if(s.stage==='regions'){
      const names={colored:'融合标签总览',allowed:'默认允许范围',active:'实际W>0作用范围',protected:'语义保护范围',source:'决策来源',weight:'实际权重预览',objects:'最终物体排除',unknown:'待确认区域',conflict:'分歧记录','pre-sam':'SAM精修前分区','post-sam':'SAM精修后分区'};
      const m=moduleOf(data,part),path=part==='colored'?shared.colored:part==='source'?shared.source_colored:part==='pre-sam'?shared.pre_sam_colored:part==='post-sam'?shared.post_sam_colored:shared.groups && shared.groups[part] || shared[part];
      const alreadyColored=['colored','source','pre-sam','post-sam'].includes(part);
      const note=part==='pre-sam'?'本图真实留痕：已完成双模型融合和按类修正，但尚未SAM精修。':part==='post-sam'?'本图真实留痕：SAM精修后的分区；不是重新推测的中间结果。':'两种降噪方法共用这份保存的分区。'+(m && m.note || '');
      return mask(path,names[part] || (m && m.name) || part,note,alreadyColored?null:(m && m.color || MASK_COLORS[part] || '#37b89c'));
    }
    if(s.stage==='weight')return column==='l0'?mask(shared.weight,'共用的实际混合权重','float32 weight.npy的显示预览；颜色深浅是混合权重，不是噪点置信度。',MASK_COLORS.weight):
      mask(shared.zero_weight || shared.protected,shared.zero_weight?'实际W=0范围':'共用的语义保护范围',shared.zero_weight?'包含语义保护以及线条保护／羽化产生的零权重像素，最终应逐像素保持原图。':'这里只显示语义保护；线条和羽化还可能产生额外W=0像素，请以实际权重为准。',MASK_COLORS.protected);
    if(s.stage==='candidate')return image(method && method[s.candidate],NAMES[column]+(s.candidate==='restored'?' / Gaussian恢复候选':' / 原始降噪候选'),s.candidate==='restored'?'恢复大范围渐变后的候选，尚未按权重融合。':'从精确浮点缓存导出的候选预览，尚未Gaussian恢复和蒙版混合。');
    return image(method && method.final,NAMES[column]+' / 最终出图','已应用共用权重与保护规则的最终结果。');
  }
  function checkRows(checks,prefix=''){
    const rows=[];
    for(const [key,value] of Object.entries(checks || {})){
      const full=prefix?prefix+'.'+key:key;
      if(value && typeof value==='object' && !Array.isArray(value)){
        if(typeof value.passed==='boolean')rows.push({key:full,name:CHECK_NAMES[key] || full,status:value.status==='not_present'?'na':value.passed?'pass':'fail',detail:value.detail || value.status || ''});
        else rows.push(...checkRows(value,full));
      }else rows.push({key:full,name:CHECK_NAMES[key] || full,status:typeof value==='boolean'?(value?'pass':'fail'):value==null?'unknown':'info',detail:value==null?'未提供':typeof value==='boolean'?'':String(value)});
    }
    return rows;
  }
  function cleanReview(value){const v=value && typeof value==='object'?value:{};const result={segmentation:['pending','ok','needs-fix'].includes(v.segmentation)?v.segmentation:'pending',preference:['pending','original','l0','esrgan-l0','adjust'].includes(v.preference)?v.preference:'pending',tags:Array.isArray(v.tags)?v.tags.filter(t=>TAGS.some(([id])=>id===t)):[],note:typeof v.note==='string'?v.note:''};if(v.context && typeof v.context==='object')result.context=JSON.parse(JSON.stringify(v.context));return result;}
  function storageKey(data){return 'batch-acceptance-v1:'+data.run_id+':'+data.cases.map(c=>c.source_sha256).join(':');}
  function aggregationInfo(data){
    const a=data.aggregation;if(!a || typeof a!=='object')return null;
    const count=data.cases.length,valid=Number.isInteger(a.initial_cases) && Number.isInteger(a.added_cases) && a.initial_cases>=0 && a.added_cases>=0 && a.initial_cases+a.added_cases===count;
    const text=valid?`本页共 ${count} 张：原 ${a.initial_cases} 张保留，另追加 ${a.added_cases} 张。原批次与追加批次各保留独立过程记录，全部图片的核验见汇总。`:
      `本页共 ${count} 张。原批次与追加批次保留独立记录，汇总数量信息需核对。`;
    const links=[['全部图片清单',a.manifest],['全部图片核验',a.validation],['原批次原始清单',a.original_review],['追加批次清单',a.extension_manifest],['追加批次核验',a.extension_validation]]
      .filter(([,p])=>localPath(p)).map(([label,path])=>({label,path}));
    return {text,links,warning:!valid};
  }
  function migrateReviews(data,current,readKey){
    const opinions=JSON.parse(JSON.stringify(current && typeof current==='object' && !Array.isArray(current)?current:{}));
    const stats={migrated_cases:0,kept_current_cases:0,identity_mismatches:0,invalid_stores:0,unavailable_stores:0};
    const cases=new Map(data.cases.map(c=>[c.id,c])),currentIds=new Set(Object.keys(opinions));
    const fullHash=value=>typeof value==='string' && /^[a-f0-9]{64}$/i.test(value);
    for(const previous of Array.isArray(data.previous_reviews)?data.previous_reviews:[]){
      if(!previous || typeof previous!=='object' || !Array.isArray(previous.cases))continue;
      const key=typeof previous.storage_key==='string'?previous.storage_key:
        typeof previous.run_id==='string' && previous.cases.every(c=>c && typeof c.source_sha256==='string')?storageKey(previous):null;
      if(!key || key===storageKey(data))continue;
      let raw,saved;
      try{raw=readKey(key);}catch(error){stats.unavailable_stores++;continue;}
      if(raw==null)continue;
      try{saved=JSON.parse(raw);}catch(error){stats.invalid_stores++;continue;}
      if(!saved || typeof saved!=='object' || Array.isArray(saved)){stats.invalid_stores++;continue;}
      for(const old of previous.cases){
        const c=old && cases.get(old.id);
        const exact=c && fullHash(old.source_sha256) && old.source_sha256===c.source_sha256 &&
          ['l0','esrgan-l0'].every(method=>{
            const fingerprint=c.methods && c.methods[method] && (c.methods[method].report_fingerprint || c.methods[method].fingerprint);
            const recorded=old.report_fingerprints && old.report_fingerprints[method];
            return fullHash(fingerprint) && fullHash(recorded) && recorded===fingerprint;
          });
        if(!exact){stats.identity_mismatches++;continue;}
        // Keep the current key, including an intentionally cleared review.
        // Previous descriptors are ordered newest first; the first match wins.
        if(Object.hasOwn(opinions,c.id)){if(currentIds.has(c.id))stats.kept_current_cases++;continue;}
        if(!Object.hasOwn(saved,c.id) || !saved[c.id] || typeof saved[c.id]!=='object' || Array.isArray(saved[c.id]))continue;
        opinions[c.id]=cleanReview(saved[c.id]);stats.migrated_cases++;
      }
    }
    return {opinions,stats};
  }
  function reviewExport(data,opinions,current){return {schema_version:1,kind:'batch-denoise-acceptance',run_id:data.run_id,created_at:new Date().toISOString(),automatic_quality_approval:false,note:'人工验收意见，不是真值标注，不自动修改模型或处理策略。',display_context:JSON.parse(JSON.stringify(current)),cases:data.cases.map(c=>({id:c.id,title:c.title || c.id,source_sha256:c.source_sha256,size:[c.width,c.height],report_fingerprints:Object.fromEntries(['l0','esrgan-l0'].map(m=>[m,c.methods && c.methods[m] && (c.methods[m].report_fingerprint || c.methods[m].fingerprint) || null])),review:cleanReview(opinions[c.id])}))};}
  const API={STAGES,defaultState,visibleColumns,normalizeData,localPath,selectPanel,checkRows,storageKey,migrateReviews,aggregationInfo,reviewExport};
  if(typeof module!=='undefined' && module.exports)module.exports=API;
  if(typeof window==='undefined' || typeof document==='undefined')return;
  window.BATCH_ACCEPTANCE_API=API;
  if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',start);else start();

  function start(){
    const $=id=>document.getElementById(id);let data;
    try{data=normalizeData(window.BATCH_ACCEPTANCE_DATA);}catch(e){$('page-error').textContent=e.message;$('page-error').hidden=false;return;}
    const state=defaultState(),panes={},imageCache=new Map(),tintCache=new Map();let opinions={};
    try{const saved=JSON.parse(localStorage.getItem(storageKey(data)) || '{}');if(saved && typeof saved==='object' && !Array.isArray(saved))opinions=saved;}catch(e){$('save-status').textContent='浏览器不允许本地保存，请导出JSON留存';}
    const migration=migrateReviews(data,opinions,key=>localStorage.getItem(key));opinions=migration.opinions;
    if(migration.stats.migrated_cases){
      try{localStorage.setItem(storageKey(data),JSON.stringify(opinions));$('save-status').textContent='已恢复 '+migration.stats.migrated_cases+' 张原有图片的验收意见；当前已有判断保留';}
      catch(e){$('save-status').textContent='已在本页恢复原有验收意见，但浏览器未允许保存，请导出JSON留存';}
    }
    const current=()=>data.cases[state.index],activeColumns=()=>visibleColumns(state);
    for(const id of COLUMNS){const canvas=$(id+'-canvas');panes[id]={canvas,ctx:canvas.getContext('2d'),base:null,layer:null,cutout:null,selection:null};}
    function safeLink(parent,text,path){if(!localPath(path))return;const a=document.createElement('a');a.textContent=text;a.href=path;a.target='_blank';a.rel='noopener';parent.append(a);}
    function loadImage(path){if(!localPath(path))return Promise.reject(new Error('无效的本地资源路径：'+path));if(imageCache.has(path))return imageCache.get(path);const promise=new Promise((resolve,reject)=>{const img=new Image();img.onload=()=>resolve(img);img.onerror=()=>{imageCache.delete(path);reject(new Error('无法加载：'+path));};img.src=path;});imageCache.set(path,promise);while(imageCache.size>12)imageCache.delete(imageCache.keys().next().value);return promise;}
    async function tinted(path,color){const img=await loadImage(path);if(!color)return img;const key=path+'|'+color;if(tintCache.has(key))return tintCache.get(key);const c=document.createElement('canvas');c.width=img.naturalWidth;c.height=img.naturalHeight;const ctx=c.getContext('2d');ctx.drawImage(img,0,0);ctx.globalCompositeOperation='source-in';ctx.fillStyle=color;ctx.fillRect(0,0,c.width,c.height);tintCache.set(key,c);while(tintCache.size>4)tintCache.delete(tintCache.keys().next().value);return c;}
    function save(){try{localStorage.setItem(storageKey(data),JSON.stringify(opinions));$('save-status').textContent='验收意见已保存在当前浏览器';}catch(e){$('save-status').textContent='本地保存不可用，请导出JSON留存';}progress();}
    function progress(){let count=0;for(const c of data.cases){const r=cleanReview(opinions[c.id]);if(r.segmentation!=='pending' && r.preference!=='pending')count++;const b=document.querySelector(`.sample-button[data-id="${c.id}"] .sample-review`);if(b)b.textContent=r.segmentation==='pending' && r.preference==='pending'?'待验收':r.segmentation!=='pending' && r.preference!=='pending'?'已记录':'部分记录';}$('review-count').textContent=count+' / '+data.cases.length+' 已记录';}
    function stageControls(){
      $('region-control').hidden=!['evidence','regions'].includes(state.stage);$('evidence-control').hidden=state.stage!=='evidence';$('candidate-control').hidden=state.stage!=='candidate';
      for(const option of $('region-view').options){
        const value=option.value;
        option.disabled=state.stage==='evidence' && !['colored','objects',...data.modules.map(m=>m.id)].includes(value);
        if(value==='pre-sam')option.disabled ||= !current().shared?.pre_sam_colored;
        if(value==='post-sam')option.disabled ||= !current().shared?.post_sam_colored;
      }
      if(Array.from($('region-view').options).some(o=>o.value===state.region && o.disabled)){state.region='colored';$('region-view').value='colored';}
      $('mask-display-controls').hidden=!['evidence','regions','weight'].includes(state.stage);$('stage-note').textContent=STAGES.find(x=>x.id===state.stage).note;
      document.querySelectorAll('.stage-button').forEach(b=>b.setAttribute('aria-current',String(b.dataset.stage===state.stage)));
      $('show-difference').disabled=!['final','checks'].includes(state.stage);$('difference-gain').disabled=!['final','checks'].includes(state.stage);
      const active=activeColumns();$('comparison-grid').style.setProperty('--panes',active.length);
      if(active.length===1 && active[0]==='original')state.visible.original=true;
      for(const id of COLUMNS){$(id+'-pane').hidden=!active.includes(id);$('show-'+id).checked=state.visible[id];}
      $('peek-original').setAttribute('aria-pressed',String(state.peekOriginal));$('peek-original').textContent=state.peekOriginal?'返回阶段结果':'暂看原图';
      $('opacity-value').value=Math.round(state.opacity*100)+'%';$('opacity').disabled=state.mode!=='overlay';
      $('source-legend').hidden=!(state.stage==='regions' && state.region==='source');
      $('current-stage').textContent=STAGES.find(x=>x.id===state.stage).name;
    }
    function build(){
      $('sample-count').textContent=data.cases.length+' 张';$('run-id').textContent=data.run_id;
      const aggregation=aggregationInfo(data);
      if(aggregation){$('aggregation-note').hidden=false;$('aggregation-text').textContent=aggregation.text;$('aggregation-note').classList.toggle('warning',aggregation.warning);for(const link of aggregation.links)safeLink($('aggregation-links'),link.label,link.path);}
      data.cases.forEach((c,index)=>{const button=document.createElement('button');button.type='button';button.className='sample-button';button.dataset.id=c.id;const im=document.createElement('img');im.src=c.thumbnail || c.original;im.alt='';im.loading='lazy';const text=document.createElement('span'),title=document.createElement('strong'),sub=document.createElement('small'),review=document.createElement('small');title.textContent=c.title || c.id;sub.textContent=c.id;review.className='sample-review';text.append(title,sub,review);button.append(im,text);button.addEventListener('click',()=>choose(index));$('sample-list').append(button);});
      STAGES.forEach((stage,index)=>{const b=document.createElement('button');b.type='button';b.className='stage-button';b.dataset.stage=stage.id;const n=document.createElement('span');n.textContent=index+1;const title=document.createElement('span');title.textContent=stage.name;b.append(n,title);b.addEventListener('click',()=>{state.stage=stage.id;stageControls();load();});$('stage-list').append(b);});
      const regions=[['colored','分组总览'],['pre-sam','SAM精修前'],['post-sam','SAM精修后'],['allowed','默认允许范围'],['active','实际作用范围 W>0'],...data.modules.map(m=>[m.id,m.name]),['objects','最终物体排除'],['unknown','待确认'],['conflict','分歧'],['protected','语义保护范围'],['source','决策来源'],['weight','实际权重']];
      for(const [value,name] of regions){const option=document.createElement('option');option.value=value;option.textContent=name;$('region-view').append(option);}
      for(const [value,name] of TAGS){const label=document.createElement('label'),input=document.createElement('input');input.type='checkbox';input.value=value;input.addEventListener('change',updateReview);label.append(input,document.createTextNode(name));$('issue-tags').append(label);}
      const p=data.parameters || {},lambda=p.lambda ?? p.lam ?? .0005,scale=p.downscale ?? .6,sigma=p.sigma_at_2048 ?? p.sigma ??12;
      const percent=new Intl.NumberFormat('zh-CN',{style:'percent',maximumFractionDigits:2});
      const restoration=percent.format(p.restoration_gain ?? 1);
      const protection=(p.structure_protection ?? 1)===0?'关闭':percent.format(p.structure_protection ?? 1);
      $('parameter-summary').textContent=`L0 ${lambda}　ESRGAN预缩图 ${Math.round(scale*100)}%　sigma ${sigma}@短边2048　渐变恢复 ${restoration}　线条保护 ${protection}　混合强度 ${p.strength ?? 1}`;
      $('parameter-json').textContent=JSON.stringify(p,null,2);
    }
    function checks(){
      const c=current(),root=$('check-list');root.replaceChildren();const rows=checkRows(c.checks || {});
      for(const row of rows){const line=document.createElement('div');line.className='check-row '+row.status;const label=document.createElement('span'),status=document.createElement('strong'),detail=document.createElement('small');label.textContent=row.name;status.textContent={pass:'通过',fail:'失败',na:'不适用',unknown:'未提供',info:'记录'}[row.status];detail.textContent=row.detail;line.append(label,status,detail);root.append(line);}
      const consistent=c.checks && c.checks.same_masks===true && c.checks.same_weight===true;
      $('shared-status').className='shared-status '+(consistent?'pass':'warning');$('shared-status').textContent=consistent?'已核对：两算法分区与浮点权重完全一致':'尚未确认两算法选区一致，请检查记录';
      $('method-links').replaceChildren();for(const m of ['l0','esrgan-l0']){const method=c.methods && c.methods[m] || {};safeLink($('method-links'),NAMES[m]+' 报告',method.report);if(method.sr)safeLink($('method-links'),'ESRGAN单步缓存预览',method.sr);}
      $('parameter-links').replaceChildren();for(const m of ['l0','esrgan-l0'])safeLink($('parameter-links'),NAMES[m]+' 原生精度与模型报告',c.methods && c.methods[m] && c.methods[m].report);
      $('parameter-json').textContent=JSON.stringify({batch:data.parameters,case_id:c.id,method_parameters:Object.fromEntries(['l0','esrgan-l0'].map(m=>[m,c.methods && c.methods[m] && c.methods[m].parameters || {}]))},null,2);
      const trace=$('trace-list');trace.replaceChildren();for(const entry of c.trace || []){const box=document.createElement('article');box.className='trace-entry';const title=document.createElement('strong'),detail=document.createElement('p'),links=document.createElement('div');title.textContent=entry.title || entry.stage || '阶段';detail.textContent=(entry.status?'['+entry.status+'] ':'')+(entry.detail || '');for(const link of entry.links || [])safeLink(links,link.label || link.title || link.path,link.path);box.append(title,detail,links);trace.append(box);}
      const sources=$('source-legend');sources.replaceChildren();for(const row of c.shared && c.shared.source_legend || []){const span=document.createElement('span'),dot=document.createElement('i');dot.style.background=Array.isArray(row.color)?`rgb(${row.color.join(',')})`:row.color;span.append(dot,document.createTextNode(row.code+' '+(row.name || '')));sources.append(span);}
    }
    function choose(index){state.index=(index+data.cases.length)%data.cases.length;state.fit=true;state.hover=null;imageCache.clear();tintCache.clear();const c=current();state.centerX=c.width/2;state.centerY=c.height/2;$('case-title').textContent=c.title || c.id;$('case-description').textContent=c.description || c.id;$('case-position').textContent=(state.index+1)+' / '+data.cases.length;$('dimensions').textContent=c.width+' × '+c.height+' px';$('source-sha').textContent=c.source_sha256;document.querySelectorAll('.sample-button').forEach(b=>b.setAttribute('aria-current',String(b.dataset.id===c.id)));const r=cleanReview(opinions[c.id]);$('segmentation-review').value=r.segmentation;$('preference-review').value=r.preference;$('review-note').value=r.note;document.querySelectorAll('#issue-tags input').forEach(x=>x.checked=r.tags.includes(x.value));checks();progress();stageControls();load();}
    function viewContext(){return {stage:state.stage,region:state.region,evidence_view:state.evidenceView,candidate:state.candidate,difference_gain:Number(state.diffGain),peek_original:state.peekOriginal,zoom:state.zoom,center:[state.centerX,state.centerY],visible_columns:activeColumns()};}
    function updateReview(){opinions[current().id]={segmentation:$('segmentation-review').value,preference:$('preference-review').value,tags:Array.from(document.querySelectorAll('#issue-tags input:checked'),x=>x.value),note:$('review-note').value,context:viewContext()};save();}
    async function load(){const request=++state.request,c=current();$('load-status').textContent='载入原尺寸资源…';$('page-error').hidden=true;for(const p of Object.values(panes)){p.base=null;p.layer=null;p.cutout=null;p.selection=null;}draw();
      try{const base=await loadImage(c.original);if(request!==state.request)return;await Promise.all(activeColumns().map(async id=>{const selection=selectPanel(c,id,state,data),pane=panes[id];let layer=null,cutout=null;if(selection.available && selection.path){layer=selection.kind==='mask'?await tinted(selection.path,selection.color):await loadImage(selection.path);if(request!==state.request)return;if(selection.kind==='mask' && state.mode==='cutout'){cutout=document.createElement('canvas');cutout.width=c.width;cutout.height=c.height;const ctx=cutout.getContext('2d');ctx.drawImage(layer,0,0,c.width,c.height);ctx.globalCompositeOperation='source-in';ctx.drawImage(base,0,0,c.width,c.height);}}
        if(request!==state.request)return;pane.base=base;pane.layer=layer;pane.cutout=cutout;pane.selection=selection;$(id+'-title').textContent=selection.title || NAMES[id];$(id+'-note').textContent=selection.note || '';$(id+'-message').textContent=selection.note || '该阶段没有保存资源。';$(id+'-message').hidden=selection.available || state.mode==='original';}));if(request!==state.request)return;if(state.fit)fit();else draw();$('load-status').textContent='当前为保存的实际阶段产物';}catch(e){if(request!==state.request)return;$('page-error').textContent=e.message+'。请保持整个验收目录的相对位置。';$('page-error').hidden=false;$('load-status').textContent='资源加载失败';}}
    function fit(){const c=current(),rects=activeColumns().map(id=>panes[id].canvas.getBoundingClientRect());state.zoom=Math.max(.01,Math.min(...rects.map(r=>Math.min((r.width-24)/c.width,(r.height-24)/c.height))));state.centerX=c.width/2;state.centerY=c.height/2;state.fit=true;draw();}
    function draw(){const c=current();$('zoom-value').textContent=Math.round(state.zoom*100)+'%';for(const id of activeColumns()){const p=panes[id],r=p.canvas.getBoundingClientRect(),dpr=Math.min(window.devicePixelRatio || 1,2),w=Math.max(1,Math.round(r.width*dpr)),h=Math.max(1,Math.round(r.height*dpr));if(p.canvas.width!==w || p.canvas.height!==h){p.canvas.width=w;p.canvas.height=h;}const ctx=p.ctx;ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,r.width,r.height);if(!p.base || !p.selection)continue;const sel=p.selection;ctx.save();ctx.imageSmoothingEnabled=state.zoom<1;ctx.translate(r.width/2,r.height/2);ctx.scale(state.zoom,state.zoom);ctx.translate(-state.centerX,-state.centerY);
        if(sel.kind==='image'){if(p.layer)ctx.drawImage(p.layer,0,0,c.width,c.height);else ctx.drawImage(p.base,0,0,c.width,c.height);}else{if(['overlay','original'].includes(state.mode))ctx.drawImage(p.base,0,0,c.width,c.height);if(state.mode!=='original' && sel.available){if(state.mode==='cutout'){if(p.cutout)ctx.drawImage(p.cutout,0,0);}else{ctx.globalAlpha=state.mode==='overlay'?state.opacity:1;if(p.layer)ctx.drawImage(p.layer,0,0,c.width,c.height);ctx.globalAlpha=1;}}}ctx.restore();if(state.hover){const x=r.width/2+(state.hover.x-state.centerX)*state.zoom,y=r.height/2+(state.hover.y-state.centerY)*state.zoom;ctx.save();ctx.strokeStyle='#ffffffcc';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(x-7,y);ctx.lineTo(x+7,y);ctx.moveTo(x,y-7);ctx.lineTo(x,y+7);ctx.stroke();ctx.restore();}}
    }
    function zoom(factor,canvas,event){const r=canvas.getBoundingClientRect(),x=event?event.clientX-r.left-r.width/2:0,y=event?event.clientY-r.top-r.height/2:0,px=state.centerX+x/state.zoom,py=state.centerY+y/state.zoom;state.zoom=Math.max(.01,Math.min(8,state.zoom*factor));state.centerX=px-x/state.zoom;state.centerY=py-y/state.zoom;state.fit=false;draw();}
    for(const id of COLUMNS){const canvas=panes[id].canvas;let drag=null;canvas.addEventListener('wheel',e=>{e.preventDefault();zoom(Math.exp(-e.deltaY*.0015),canvas,e);},{passive:false});canvas.addEventListener('pointerdown',e=>{if(e.button!==0)return;drag={x:e.clientX,y:e.clientY};canvas.setPointerCapture(e.pointerId);canvas.classList.add('dragging');});canvas.addEventListener('pointermove',e=>{if(drag){state.centerX-=(e.clientX-drag.x)/state.zoom;state.centerY-=(e.clientY-drag.y)/state.zoom;drag={x:e.clientX,y:e.clientY};state.fit=false;}const r=canvas.getBoundingClientRect();state.hover={x:state.centerX+(e.clientX-r.left-r.width/2)/state.zoom,y:state.centerY+(e.clientY-r.top-r.height/2)/state.zoom};$('pixel-position').textContent='原图坐标 '+Math.floor(state.hover.x)+', '+Math.floor(state.hover.y);draw();});const stop=()=>{drag=null;canvas.classList.remove('dragging');};canvas.addEventListener('pointerup',stop);canvas.addEventListener('pointercancel',stop);canvas.addEventListener('pointerleave',()=>{if(!drag){state.hover=null;$('pixel-position').textContent='各列同步查看同一位置';draw();}});canvas.addEventListener('dblclick',fit);$('show-'+id).addEventListener('change',e=>{if(!e.target.checked && activeColumns().length===1){e.target.checked=true;return;}state.visible[id]=e.target.checked;stageControls();if(state.fit)fit();load();});}
    $('previous-case').addEventListener('click',()=>choose(state.index-1));$('next-case').addEventListener('click',()=>choose(state.index+1));$('fit-view').addEventListener('click',fit);$('native-view').addEventListener('click',()=>{state.zoom=1;state.fit=false;draw();});
    $('peek-original').addEventListener('click',()=>{state.peekOriginal=!state.peekOriginal;stageControls();load();});
    for(const [id,key] of [['region-view','region'],['evidence-view','evidenceView'],['candidate-view','candidate'],['difference-gain','diffGain'],['mask-mode','mode']])$(id).addEventListener('change',e=>{state[key]=e.target.value;stageControls();load();});$('opacity').addEventListener('input',e=>{state.opacity=Number(e.target.value)/100;stageControls();draw();});
    $('segmentation-review').addEventListener('change',updateReview);$('preference-review').addEventListener('change',updateReview);$('review-note').addEventListener('input',updateReview);
    $('export-review').addEventListener('click',()=>{const out=reviewExport(data,opinions,viewContext()),blob=new Blob([JSON.stringify(out,null,2)],{type:'application/json;charset=utf-8'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='batch-acceptance-'+data.run_id+'.json';document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);$('save-status').textContent='已导出验收意见；处理结果和策略未修改';});
    document.addEventListener('keydown',e=>{if(e.target.closest('input,textarea,select,[contenteditable]'))return;if(e.key==='ArrowLeft'){e.preventDefault();choose(state.index-1);}if(e.key==='ArrowRight'){e.preventDefault();choose(state.index+1);}if(e.key==='0')fit();if(e.key==='1'){state.zoom=1;state.fit=false;draw();}if(e.key==='+' || e.key==='=')zoom(1.25,panes[activeColumns()[0]].canvas);if(e.key==='-')zoom(.8,panes[activeColumns()[0]].canvas);});
    const observer=new ResizeObserver(()=>{if(state.fit)fit();else draw();});for(const p of Object.values(panes))observer.observe(p.canvas);
    build();choose(0);
  }
})();
