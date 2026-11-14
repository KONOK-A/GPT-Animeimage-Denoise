(function () {
  'use strict';

  const MODELS=['see','m2f','fusion'];
  const NAMES={see:'See-through',m2f:'Mask2Former v4',fusion:'当前 fusion'};
  const DIFF_COLORS={m2f_only:'#3572d0',see_only:'#ed9738',fusion_added:'#2ba878',fusion_removed:'#e768a8'};
  const CHOICES=[['pending','待判断'],['see','See-through 更好'],['m2f','Mask2Former 更好'],['fusion','当前 fusion 更好'],['complementary','适合互补']];
  const DETAILS=[['all','服饰分组'],['11','上衣 topwear'],['13','下装 bottomwear'],['1','头饰 headwear'],['10','颈饰 neckwear'],['6','耳饰 earwear'],['4','眼镜 eyewear'],['15','鞋 footwear'],['12','手／臂粗类（含裸臂）'],['14','腿粗类（含裸腿）']];

  function colorOf(label,fallback) {
    const value=label && label.color;
    return Array.isArray(value)?`rgb(${value.join(',')})`:(value || fallback || '#7cb6e8');
  }
  function defaultState() {
    return {index:0,part:'outfit',view:'white',projection:'score',threshold:'1',detail:'all',raw:false,
      rawIds:{see:[],m2f:[],fusion:[]},difference:false,showUnknown:false,mode:'overlay',opacity:.55,
      visible:{see:true,m2f:true,fusion:true},zoom:1,centerX:0,centerY:0,hover:null,fit:true,request:0};
  }
  function rawPanelState(state,open) {
    const hasSelection=MODELS.some(model=>(state.rawIds[model] || []).length>0);
    return {rawPanel:open,raw:open && hasSelection?true:state.raw};
  }
  function normalizeData(data) {
    if(!data || !Array.isArray(data.cases) || !data.cases.length)throw new Error('没有找到对比数据。请保持 manifest.js、assets 与页面的相对位置。');
    const ids=new Set();
    for(const c of data.cases) {
      if(!c.id || ids.has(c.id) || !c.original || !(c.width>0) || !(c.height>0) || !c.fusion)throw new Error('样本缺少有效身份、尺寸或已保存的 fusion 结果。');
      ids.add(c.id);
    }
    return {...data,modules:data.modules || [],see_labels:data.see_labels || [],m2f_labels:data.m2f_labels || [],meta:data.meta || {}};
  }
  function prediction(c,model,s) {
    if(model==='fusion')return c.fusion;
    const view=c.views && c.views[s.view];
    if(!view || !view[model])return null;
    return model==='see'?view.see.thresholds && view.see.thresholds[String(s.threshold)]:view.m2f.projections && view.m2f.projections[s.projection];
  }
  function partExplanation(s,data) {
    const selected=data.modules.find(m=>m.id===s.part);
    const scope=selected && selected.note || '六模块映射范围不等价：See皮肤仅为颈部，服饰不强并有歧义的手脚粗类；原始通道可独立查看。';
    if(s.difference)return scope+' 当前显示相对所选单视图的标签差异，不是准确率。fusion还包含双输入与SAM影响，不全部归因于See；待确认255不计为已移出。';
    if(s.raw)return '独立勾选原始通道，See允许重叠。fusion只有六业务模块；意见不会自动修改融合。';
    if(s.part==='outfit' && s.detail!=='all')return 'See显示真实细类；其他两栏只提供服饰或配饰整体参照，不是等价细分类。';
    if(s.part==='objects')return 'See显示当前单输入原始objects，fusion显示已保存的最终物体排除；M2F没有独立物体类。';
    if(s.part==='unknown')return '待确认255单独展示，不当作已确认移除，也不是错误率。';
    if(s.part==='conflict')return '分歧记录不等于全部待确认；部分区域已由当前fusion作出归属。';
    return scope;
  }
  function differenceSupported(c,s,data) {
    if(s.raw || (s.part==='outfit' && s.detail!=='all') || !['all',...data.modules.map(m=>m.id)].includes(s.part))return false;
    return MODELS.every(model=>{const p=prediction(c,model,s);return p && p.groups && p.groups[s.part];}) && Boolean(c.fusion.unknown);
  }
  function selectAssets(c,model,s,data) {
    const p=prediction(c,model,s);
    const labels=model==='see'?data.see_labels:model==='m2f'?data.m2f_labels:data.modules;
    const module=data.modules.find(m=>m.id===s.part);
    const baseDetail=model==='fusion'?'固定双输入 + SAM':model==='see'?`${s.view==='white'?'白底':'原图'}输入 / See > ${s.threshold}`:`${s.view==='white'?'白底':'原图'} / ${s.projection==='score'?'分数还原判类':'基线最近邻'}`;
    const make=(assets,extra={})=>({available:true,assets,detail:baseDetail,note:'',...extra});
    if(!p)return {available:false,assets:[],reason:'missing-view',detail:baseDetail,message:'缺少该输入来源下的结果。'};
    if(s.difference) {
      if(!differenceSupported(c,s,data))return {available:false,assets:[],reason:'difference-unavailable',detail:'此选择不提供等价分组差异'};
      const m=prediction(c,'m2f',s).groups[s.part],v=prediction(c,'see',s).groups[s.part],f=c.fusion.groups[s.part];
      const colors={...DIFF_COLORS,...(data.meta && data.meta.diff_colors || {})};
      if(model==='m2f')return make([{path:m,subtract:v,color:colors.m2f_only}],{detail:'M2F 相对 See 的独有选区',note:'选区差异，不是误差或准确率。'});
      if(model==='see')return make([{path:v,subtract:m,color:colors.see_only}],{detail:'See 相对 M2F 的独有选区',note:'模型类别范围不同，请结合原图判断。'});
      return make([{path:f,subtract:m,color:colors.fusion_added},{path:m,subtract:f,exclude:[c.fusion.unknown],color:colors.fusion_removed}],
        {detail:'固定 fusion 相对当前 M2F 的标签差异',note:'绿色为新增，粉色为已确认移出；待确认255已排除。差异包含双输入与SAM影响，不全部归因于See。'});
    }
    if(s.raw) {
      const ids=s.rawIds[model] || [];
      return make(ids.map(id=>model==='fusion'?{path:p.groups && p.groups[id],color:colorOf(labels.find(m=>m.id===id))}:
        {path:p.masks && p.masks[id],color:colorOf(labels.find(m=>Number(m.id)===Number(id)))}).filter(a=>a.path),
        {detail:model==='fusion'?'固定结果 / 业务模块':baseDetail+' / 原始通道',note:ids.length?'See通道可重叠；fusion只显示实际业务模块，不伪造细分类。':'此模型尚未勾选通道，请在原始通道中选择。'});
    }
    if(['unknown','conflict'].includes(s.part)) {
      if(model!=='fusion')return {available:false,assets:[],reason:'fusion-diagnostic-only',detail:'仅当前 fusion 提供',message:'这是 fusion 的诊断层，原始模型没有同名输出。'};
      return make([{path:p[s.part],color:s.part==='unknown'?'#e9b941':'#e87943'}],{note:s.part==='unknown'?'黄色是待确认255，默认不处理；不是已经确认移除。':'冲突记录出现过的模型分歧，其中部分已按规则作出归属。'});
    }
    if(s.part==='objects') {
      if(model==='m2f')return {available:false,assets:[],reason:'no-object-class',detail:'无独立物体类 / 原图参照',message:'Mask2Former没有独立物体类别；叠加或原图模式下仅作原图参照。'};
      return make([{path:model==='see'?p.masks[18]:p.objects,color:'#ee684d'}],{detail:model==='see'?baseDetail+' / 原始 objects':'固定结果 / 最终物体排除',note:model==='see'?'原始物体通道可与人物通道重叠。':'这里是经过融合和SAM后的最终外物层。'});
    }
    if(s.part==='outfit' && s.detail!=='all') {
      const id=Number(s.detail),label=data.see_labels.find(m=>Number(m.id)===id);
      const ambiguous=id===12 || id===14;
      if(model==='see')return make([{path:p.masks[id],color:colorOf(label)}],
        {detail:baseDetail+' / '+(label && label.name || id),note:ambiguous?'手脚粗类可能含裸露四肢，不自动归入裸肤或服饰。':'See原始细类；其他两栏只能作为更宽范围的整体参照。'});
      if(model==='m2f' && [1,4,6,10].includes(id))return make([{path:p.masks[11],color:colorOf(module)}],
        {detail:baseDetail+' / 配饰整体参照',note:'M2F只有配饰整体类，与See当前细类不等价。'});
      return make([{path:p.groups.outfit,color:colorOf(module)}],
        {detail:baseDetail+' / 服饰整体参照',note:'没有对应的独立细类；当前显示服饰整体，与See当前细类不等价。'});
    }
    if(s.part==='all')return make([{path:p.groups_colored || p.colored,color:null}],{note:'六模块映射总览；See歧义肢体、尾巴、翅膀与物体未强行并入。'});
    return make([{path:p.groups && p.groups[s.part],color:colorOf(module)}].filter(a=>a.path),{note:module && module.note || ''});
  }
  function reviewExport(data,opinions,current) {
    return {schema_version:1,kind:'segmentation-and-fusion-preferences',created_at:new Date().toISOString(),
      purpose:'用户观察，不是像素真值，不自动修改fusion策略或模型。',display_context:JSON.parse(JSON.stringify(current)),
      cases:data.cases.map(c=>({id:c.id,title:c.title || c.id,source_sha256:c.source_sha256,size:[c.width,c.height],
        fusion_report_fingerprint:c.fusion.report_fingerprint,
        modules:JSON.parse(JSON.stringify(opinions[c.id] && opinions[c.id].modules || {})),note:opinions[c.id] && opinions[c.id].note || ''}))};
  }

  function importReview(data,opinions,imported) {
    if(!imported || !Array.isArray(imported.cases))throw new Error('判断JSON缺少cases列表。');
    const known=new Map(data.cases.map(c=>[c.id,c])),result={...opinions};
    const stats={matched_cases:0,skipped_cases:0,copied_choices:0,kept_choices:0,copied_notes:0};
    const allowed=new Set(CHOICES.map(c=>c[0]));
    for(const item of imported.cases) {
      const c=item && known.get(item.id);
      if(!c || item.source_sha256!==c.source_sha256){stats.skipped_cases++;continue;}
      stats.matched_cases++;
      const old=result[c.id] && typeof result[c.id]==='object'?result[c.id]:{};
      const record={...old,modules:{...(old.modules || {})},note:typeof old.note==='string'?old.note:''};
      for(const m of data.modules) {
        const candidate=item.modules && item.modules[m.id];
        if(!candidate || !allowed.has(candidate.choice))continue;
        const current=record.modules[m.id];
        if(current && current.choice && current.choice!=='pending'){stats.kept_choices++;continue;}
        record.modules[m.id]=JSON.parse(JSON.stringify(candidate));stats.copied_choices++;
      }
      if(!record.note && typeof item.note==='string' && item.note){record.note=item.note;stats.copied_notes++;}
      result[c.id]=record;
    }
    return {opinions:result,stats};
  }

  const API={defaultState,rawPanelState,normalizeData,prediction,partExplanation,differenceSupported,selectAssets,reviewExport,importReview,DETAILS};
  if(typeof module!=='undefined' && module.exports)module.exports=API;
  if(typeof window==='undefined' || typeof document==='undefined')return;
  window.SEG_FULL_COMPARISON_API=API;
  if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',start);else start();

  function start() {
    const $=id=>document.getElementById(id);let data;
    try {data=normalizeData(window.SEG_FULL_COMPARISON_DATA);}
    catch(error){$('page-error').textContent=error.message;$('page-error').hidden=false;return;}
    const state=defaultState(),panes={},imageCache=new Map(),layerCache=new Map();
    const storageKey='segmentation-full-comparison-v1:'+data.cases.map(c=>c.source_sha256).join(':');
    let opinions={};
    try{const saved=JSON.parse(localStorage.getItem(storageKey)||'{}');if(saved && typeof saved==='object' && !Array.isArray(saved))opinions=saved;}
    catch(error){$('save-status').textContent='浏览器未允许保存，请导出判断JSON留存';}
    const current=()=>data.cases[state.index];
    const context=()=>({view:state.view,see_threshold:Number(state.threshold),projection:state.projection,part:state.part,detail:state.detail,raw:state.raw,
      raw_ids:Object.fromEntries(MODELS.map(model=>[model,[...state.rawIds[model]]])),difference:state.difference});
    for(const model of MODELS){const canvas=$(model+'-canvas');panes[model]={canvas,context:canvas.getContext('2d'),layers:[],base:null,cutout:null};}

    function boundedSet(cache,key,value,max) {cache.set(key,value);while(cache.size>max)cache.delete(cache.keys().next().value);return value;}
    function image(path) {
      if(imageCache.has(path)){const value=imageCache.get(path);imageCache.delete(path);imageCache.set(path,value);return value;}
      const promise=new Promise((resolve,reject)=>{const img=new Image();img.onload=()=>resolve(img);img.onerror=()=>{imageCache.delete(path);reject(new Error('无法加载 '+path));};img.src=path;});
      return boundedSet(imageCache,path,promise,12);
    }
    async function layer(asset) {
      if(!asset.color && !asset.subtract && !asset.exclude)return image(asset.path);
      const key=JSON.stringify(asset);if(layerCache.has(key))return layerCache.get(key);
      const source=await image(asset.path),canvas=document.createElement('canvas');canvas.width=source.naturalWidth;canvas.height=source.naturalHeight;
      const ctx=canvas.getContext('2d');ctx.drawImage(source,0,0);
      for(const excluded of [asset.subtract,...(asset.exclude || [])].filter(Boolean)) {
        const other=await image(excluded);ctx.globalCompositeOperation='destination-out';ctx.drawImage(other,0,0,canvas.width,canvas.height);
      }
      if(asset.color){ctx.globalCompositeOperation='source-in';ctx.fillStyle=asset.color;ctx.fillRect(0,0,canvas.width,canvas.height);}
      return boundedSet(layerCache,key,canvas,5);
    }
    function save() {
      try{localStorage.setItem(storageKey,JSON.stringify(opinions));$('save-status').textContent='判断已保存在当前浏览器';}
      catch(error){$('save-status').textContent='浏览器未允许保存，请导出判断JSON留存';}
      progress();
    }
    function progress() {
      for(const button of document.querySelectorAll('.sample-button')) {
        const values=(opinions[button.dataset.id] || {}).modules || {};
        const count=data.modules.filter(m=>values[m.id] && values[m.id].choice!=='pending').length;
        button.querySelector('.sample-reviewed').textContent=count?'已判断 '+count+' / 6':'';
      }
      const values=(opinions[current().id] || {}).modules || {};
      $('review-progress').textContent=data.modules.filter(m=>values[m.id] && values[m.id].choice!=='pending').length+' / '+data.modules.length;
    }
    function build() {
      $('sample-count').textContent=data.cases.length+' 张';
      data.cases.forEach((c,index)=>{
        const button=document.createElement('button');button.type='button';button.className='sample-button';button.dataset.id=c.id;
        const thumb=document.createElement('img');thumb.src=c.thumbnail || c.original;thumb.alt='';thumb.loading='lazy';
        const text=document.createElement('span');for(const [cls,value] of [['sample-name',c.title || c.id],['sample-id',c.id],['sample-reviewed','']]){const item=document.createElement('span');item.className=cls;item.textContent=value;text.append(item);}
        button.append(thumb,text);button.addEventListener('click',()=>choose(index));$('sample-list').append(button);
      });
      for(const m of [{id:'all',name:'分组总览'},...data.modules,{id:'objects',name:'物体'},{id:'unknown',name:'待确认'},{id:'conflict',name:'分歧记录'}]) {
        const b=document.createElement('button');b.type='button';b.className='part-tab';b.dataset.part=m.id;b.textContent=m.name;
        b.addEventListener('click',()=>{state.part=m.id;state.raw=false;state.rawPanel=false;state.detail='all';constrain();controls();load();});$('module-tabs').append(b);
      }
      for(const [value,name] of DETAILS){const option=document.createElement('option');option.value=value;option.textContent=name;$('garment-detail').append(option);}
      for(const model of MODELS) {
        const labels=model==='see'?data.see_labels:model==='m2f'?data.m2f_labels:data.modules;
        for(const entry of labels) {
          const label=document.createElement('label');label.className='channel-choice';const input=document.createElement('input');input.type='checkbox';input.value=entry.id;
          const swatch=document.createElement('i');swatch.style.background=colorOf(entry);const text=document.createElement('span');text.textContent=entry.name;
          input.addEventListener('change',()=>{state.raw=true;state.rawIds[model]=Array.from($(model+'-channel-list').querySelectorAll('input:checked'),x=>model==='fusion'?x.value:Number(x.value));constrain();controls();load();});
          label.append(input,swatch,text);$(model+'-channel-list').append(label);
        }
      }
      for(const m of data.modules) {
        const label=document.createElement('label');label.className='review-part';const text=document.createElement('span');text.textContent=m.name;
        const select=document.createElement('select');select.dataset.part=m.id;select.setAttribute('aria-label',m.name+'偏好');
        for(const [value,name] of CHOICES){const option=document.createElement('option');option.value=value;option.textContent=name;select.append(option);}
        select.addEventListener('change',()=>{const id=current().id;opinions[id] ||= {modules:{},note:''};opinions[id].modules ||= {};opinions[id].modules[m.id]={choice:select.value,evidence_context:context()};save();});
        label.append(text,select);$('review-choices').append(label);
      }
    }
    function constrain() {if(!differenceSupported(current(),state,data))state.difference=false;}
    function controls() {
      document.querySelectorAll('.part-tab').forEach(b=>b.setAttribute('aria-pressed',String(!state.raw && b.dataset.part===state.part)));
      $('garment-detail').disabled=state.raw || state.part!=='outfit';$('garment-detail').value=state.detail;
      $('difference-view').disabled=!differenceSupported(current(),state,data);$('difference-view').checked=state.difference;
      $('show-unknown').checked=state.showUnknown;
      $('show-unknown').disabled=['cutout','original'].includes(state.mode);
      $('overlay-opacity').disabled=state.mode!=='overlay';$('opacity-value').value=Math.round(state.opacity*100)+'%';
      $('part-note').textContent=partExplanation(state,data);
      $('difference-legend').hidden=!state.difference;
      $('raw-panel').hidden=!state.rawPanel;$('raw-toggle').setAttribute('aria-expanded',String(Boolean(state.rawPanel)));
      $('input-context').textContent=`原始模型：${state.view==='white'?'白底':'原图'}输入；当前fusion固定，不随这些查看参数重算`;
      const input=current().views && current().views[state.view] && current().views[state.view].input;
      if(input)$('analysis-input').href=input;else $('analysis-input').removeAttribute('href');
      for(const model of MODELS){$(model+'-pane').hidden=!state.visible[model];$('show-'+model).checked=state.visible[model];}
      $('comparison-grid').style.setProperty('--panes',MODELS.filter(m=>state.visible[m]).length);
    }
    function choose(index) {
      state.index=(index+data.cases.length)%data.cases.length;state.fit=true;state.hover=null;imageCache.clear();layerCache.clear();
      const c=current();state.centerX=c.width/2;state.centerY=c.height/2;
      $('case-title').textContent=c.title || c.id;$('case-description').textContent=c.description || '';$('case-position').textContent=(state.index+1)+' / '+data.cases.length;
      $('image-dimensions').textContent=c.width+' × '+c.height+' px';
      document.querySelectorAll('.sample-button').forEach(b=>b.setAttribute('aria-current',String(b.dataset.id===c.id)));
      const record=opinions[c.id] || {};document.querySelectorAll('.review-part select').forEach(s=>s.value=(record.modules && record.modules[s.dataset.part] || {}).choice || 'pending');
      $('review-notes').value=record.note || '';constrain();controls();progress();load();
    }
    async function load() {
      const request=++state.request,c=current();$('load-status').textContent='正在载入选区…';$('page-error').hidden=true;
      for(const pane of Object.values(panes)){pane.base=null;pane.layers=[];pane.cutout=null;}draw();
      try {
        const base=await image(c.original);
        if(request!==state.request)return;
        await Promise.all(MODELS.map(async model=>{
          if(!state.visible[model])return;
          const selected=selectAssets(c,model,state,data);let assets=selected.assets;
          if(model==='fusion' && state.showUnknown && !['unknown','conflict'].includes(state.part) && ['overlay','mask'].includes(state.mode))assets=[...assets,{path:c.fusion.unknown,color:'#efbd3f'}];
          let layers=[];
          if(state.mode!=='original') {
            if(state.raw && assets.length>1) {
              const combined=document.createElement('canvas');combined.width=c.width;combined.height=c.height;
              const ctx=combined.getContext('2d');ctx.globalCompositeOperation='screen';
              for(const asset of assets){const rendered=await layer(asset);if(request!==state.request)return;ctx.drawImage(rendered,0,0,c.width,c.height);}layers=[combined];
            } else layers=await Promise.all(assets.map(layer));
          }
          if(request!==state.request)return;
          const pane=panes[model];pane.base=base;pane.layers=layers;pane.available=selected.available;
          if(state.mode==='cutout' && selected.available) {
            const cut=document.createElement('canvas');cut.width=c.width;cut.height=c.height;const ctx=cut.getContext('2d');
            for(const rendered of layers)ctx.drawImage(rendered,0,0);ctx.globalCompositeOperation='source-in';ctx.drawImage(base,0,0);pane.cutout=cut;
          }
          $(model+'-detail').textContent=selected.detail || '';
          $(model+'-note').textContent=selected.note || '';
          $(model+'-message').hidden=selected.available || state.mode==='original' || (selected.reason==='no-object-class' && state.mode==='overlay');
          $(model+'-message').textContent=selected.message || '此选择没有对应的独立分类。';
        }));
        if(request!==state.request)return;
        if(state.fit)fit();else draw();$('load-status').textContent=state.raw?'原始通道独立显示':'选择与数据已载入';
      } catch(error) {
        if(request!==state.request)return;$('load-status').textContent='资源加载失败';$('page-error').textContent=error.message+'。请保持页面与assets的相对位置。';$('page-error').hidden=false;
      }
    }
    function fit() {
      const c=current(),rects=MODELS.filter(m=>state.visible[m]).map(m=>panes[m].canvas.getBoundingClientRect());
      state.zoom=Math.max(.01,Math.min(...rects.map(r=>Math.min((r.width-24)/c.width,(r.height-24)/c.height))));
      state.centerX=c.width/2;state.centerY=c.height/2;state.fit=true;draw();
    }
    function draw() {
      const c=current();$('zoom-value').textContent=Math.round(state.zoom*100)+'%';
      for(const model of MODELS) {
        if(!state.visible[model])continue;
        const pane=panes[model],{canvas,context:ctx}=pane,r=canvas.getBoundingClientRect(),dpr=Math.min(window.devicePixelRatio || 1,2);
        const w=Math.max(1,Math.round(r.width*dpr)),h=Math.max(1,Math.round(r.height*dpr));if(canvas.width!==w || canvas.height!==h){canvas.width=w;canvas.height=h;}
        ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,r.width,r.height);if(!pane.base)continue;
        ctx.save();ctx.imageSmoothingEnabled=state.zoom<1;ctx.translate(r.width/2,r.height/2);ctx.scale(state.zoom,state.zoom);ctx.translate(-state.centerX,-state.centerY);
        if(state.mode==='overlay' || state.mode==='original')ctx.drawImage(pane.base,0,0,c.width,c.height);
        if(pane.available && state.mode!=='original') {
          if(state.mode==='cutout'){if(pane.cutout)ctx.drawImage(pane.cutout,0,0);}
          else {ctx.globalAlpha=state.mode==='overlay'?state.opacity:1;for(const rendered of pane.layers)ctx.drawImage(rendered,0,0,c.width,c.height);ctx.globalAlpha=1;}
        }
        ctx.restore();
        if(state.hover){const x=r.width/2+(state.hover.x-state.centerX)*state.zoom,y=r.height/2+(state.hover.y-state.centerY)*state.zoom;ctx.save();ctx.strokeStyle='#ffffffc7';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(x-7,y);ctx.lineTo(x+7,y);ctx.moveTo(x,y-7);ctx.lineTo(x,y+7);ctx.stroke();ctx.restore();}
      }
    }
    function zoom(factor,canvas,event) {
      const r=canvas.getBoundingClientRect(),x=event?event.clientX-r.left-r.width/2:0,y=event?event.clientY-r.top-r.height/2:0;
      const ix=state.centerX+x/state.zoom,iy=state.centerY+y/state.zoom;state.zoom=Math.max(.01,Math.min(8,state.zoom*factor));
      state.centerX=ix-x/state.zoom;state.centerY=iy-y/state.zoom;state.fit=false;draw();
    }
    for(const model of MODELS) {
      const canvas=panes[model].canvas;let drag=null;
      canvas.addEventListener('wheel',e=>{e.preventDefault();zoom(Math.exp(-e.deltaY*.0015),canvas,e);},{passive:false});
      canvas.addEventListener('pointerdown',e=>{if(e.button!==0)return;drag={x:e.clientX,y:e.clientY};canvas.setPointerCapture(e.pointerId);canvas.classList.add('dragging');});
      canvas.addEventListener('pointermove',e=>{
        if(drag){state.centerX-=(e.clientX-drag.x)/state.zoom;state.centerY-=(e.clientY-drag.y)/state.zoom;drag={x:e.clientX,y:e.clientY};state.fit=false;}
        const r=canvas.getBoundingClientRect();state.hover={x:state.centerX+(e.clientX-r.left-r.width/2)/state.zoom,y:state.centerY+(e.clientY-r.top-r.height/2)/state.zoom};
        $('pixel-position').textContent='原图坐标 '+Math.floor(state.hover.x)+', '+Math.floor(state.hover.y);draw();
      });
      const stop=()=>{drag=null;canvas.classList.remove('dragging');};canvas.addEventListener('pointerup',stop);canvas.addEventListener('pointercancel',stop);
      canvas.addEventListener('pointerleave',()=>{if(!drag){state.hover=null;$('pixel-position').textContent='三个视图使用同一图像中心';draw();}});canvas.addEventListener('dblclick',fit);
      $('show-'+model).addEventListener('change',e=>{
        if(!e.target.checked && MODELS.filter(m=>state.visible[m]).length===1){e.target.checked=true;return;}
        state.visible[model]=e.target.checked;controls();if(state.fit)fit();load();
      });
    }
    $('previous-case').addEventListener('click',()=>choose(state.index-1));$('next-case').addEventListener('click',()=>choose(state.index+1));
    $('fit-view').addEventListener('click',fit);$('native-view').addEventListener('click',()=>{state.zoom=1;state.fit=false;draw();});
    for(const [id,key] of [['input-view','view'],['see-threshold','threshold'],['m2f-projection','projection'],['display-mode','mode']])$(id).addEventListener('change',e=>{state[key]=e.target.value;constrain();controls();load();});
    $('garment-detail').addEventListener('change',e=>{state.detail=e.target.value;state.raw=false;state.rawPanel=false;constrain();controls();load();});
    $('overlay-opacity').addEventListener('input',e=>{state.opacity=Number(e.target.value)/100;controls();draw();});
    $('difference-view').addEventListener('change',e=>{state.difference=e.target.checked;controls();load();});
    $('show-unknown').addEventListener('change',e=>{state.showUnknown=e.target.checked;controls();load();});
    $('raw-toggle').addEventListener('click',()=>{const next=rawPanelState(state,!state.rawPanel),reload=next.raw!==state.raw;Object.assign(state,next);constrain();controls();if(reload)load();});
    $('return-modules').addEventListener('click',()=>{state.raw=false;state.rawPanel=false;constrain();controls();load();});
    $('review-toggle').addEventListener('click',()=>{const open=$('review-content').hidden;$('review-content').hidden=!open;$('review-toggle').setAttribute('aria-expanded',String(open));$('review-toggle').textContent=open?'收起记录':'展开记录';});
    $('review-notes').addEventListener('input',e=>{opinions[current().id] ||= {modules:{},note:''};opinions[current().id].note=e.target.value;save();});
    $('export-review').addEventListener('click',()=>{const value=reviewExport(data,opinions,context()),blob=new Blob([JSON.stringify(value,null,2)],{type:'application/json;charset=utf-8'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='segmentation-full-review-'+new Date().toISOString().slice(0,10)+'.json';document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);$('save-status').textContent='已导出判断JSON；模型与fusion未修改';});
    $('import-review').addEventListener('change',async e=>{
      const file=e.target.files && e.target.files[0];if(!file)return;
      try {
        const merged=importReview(data,opinions,JSON.parse(await file.text()));opinions=merged.opinions;save();
        const record=opinions[current().id] || {};
        document.querySelectorAll('.review-part select').forEach(s=>s.value=(record.modules && record.modules[s.dataset.part] || {}).choice || 'pending');$('review-notes').value=record.note || '';
        $('save-status').textContent=`导入${merged.stats.copied_choices}项判断；保留当前${merged.stats.kept_choices}项；身份不匹配跳过${merged.stats.skipped_cases}图`;
      } catch(error){$('save-status').textContent='导入失败：'+error.message;}
      e.target.value='';
    });
    document.addEventListener('keydown',e=>{
      if(e.target.closest('input,select,textarea,[contenteditable]'))return;
      if(e.key==='ArrowLeft'){e.preventDefault();choose(state.index-1);}if(e.key==='ArrowRight'){e.preventDefault();choose(state.index+1);}
      if(e.key==='0')fit();if(e.key==='1'){state.zoom=1;state.fit=false;draw();}
      const active=MODELS.find(m=>state.visible[m]);if(e.key==='+' || e.key==='=')zoom(1.25,panes[active].canvas);if(e.key==='-')zoom(.8,panes[active].canvas);
    });
    const observer=new ResizeObserver(()=>{if(state.fit)fit();else draw();});for(const pane of Object.values(panes))observer.observe(pane.canvas);
    build();choose(0);
  }
})();
