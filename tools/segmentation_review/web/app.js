(function () {
  'use strict';

  const COLORS = {skin:'#f3b58f',face:'#5b9be7',hair:'#ed6c91',eyes:'#5cd4d2',mouth:'#ee91d5',outfit:'#a696eb'};
  const CHOICES = [['pending','待判断'],['see','See-through 更好'],['m2f','Mask2Former 更好'],['complementary','适合互补']];

  function colorOf(label, fallback) {
    const value=label && label.color;
    return Array.isArray(value) ? `rgb(${value.join(',')})` : (value || fallback || '#59b9d2');
  }

  function normalizeData(data) {
    if (!data || !Array.isArray(data.cases) || !data.cases.length) throw new Error('没有找到样本数据。请把 manifest.js 和 assets 文件夹与页面放在一起。');
    const ids=new Set();
    data.cases.forEach(c=>{
      if (!c.id || ids.has(c.id) || !(c.width>0) || !(c.height>0) || !c.original) throw new Error('样本清单缺少有效的图片身份或尺寸。');
      ids.add(c.id);
    });
    return {...data, modules:data.modules || [], see_labels:data.see_labels || [], m2f_labels:data.m2f_labels || []};
  }

  function maskAssets(c, model, view, threshold, part, rawIds, modules, labels) {
    const prediction=c.views && c.views[view] && c.views[view][model];
    if (!prediction) return {available:false, assets:[]};
    const selected=model==='see' ? prediction.thresholds && prediction.thresholds[String(threshold)] : prediction;
    if (!selected) return {available:false, assets:[]};
    if (rawIds !== null) {
      const assets=rawIds.map(id=>({path:selected.masks && selected.masks[id],color:colorOf((labels || []).find(x=>Number(x.id)===Number(id)))})).filter(x=>x.path);
      return {available:true,assets};
    }
    if (part==='objects') {
      if(model==='m2f') return {available:false,assets:[],reason:'no-object-class',message:'无独立物体类。Mask2Former 没有对应的 objects 类；叠加或原图模式下，该侧仅作原图参照。'};
      const objectPath=selected.masks && selected.masks[18];
      return {available:Boolean(objectPath),assets:objectPath?[{path:objectPath,color:colorOf((labels || []).find(x=>Number(x.id)===18),'#eab85c')}]:[]};
    }
    if (part==='all' && selected.groups_colored) return {available:true,assets:[{path:selected.groups_colored,color:null}]};
    if (selected.groups && selected.groups[part]) return {available:true,assets:[{path:selected.groups[part],color:colorOf(modules.find(m=>m.id===part),COLORS[part] || '#68cbd2')}]};
    const selectedModules=part==='all' ? modules : modules.filter(m=>m.id===part);
    const assets=[];
    for (const module of selectedModules) {
      const groupPath=selected.groups && selected.groups[module.id];
      if (groupPath) assets.push({path:groupPath,color:colorOf(module,COLORS[module.id])});
      else for (const id of module[model] || []) if (selected.masks && selected.masks[id]) assets.push({path:selected.masks[id],color:colorOf(module,COLORS[module.id])});
    }
    return {available:true,assets};
  }

  function reviewExport(data, opinions, current) {
    return {schema_version:1,kind:'segmentation-model-preferences',created_at:new Date().toISOString(),
      purpose:'用户观察；不是像素真值，不自动修改 fusion 配置',
      display_context:current,
      cases:data.cases.map(c=>({id:c.id,title:c.title || c.id,source_sha256:c.source_sha256,
        size:[c.width,c.height],modules:opinions[c.id] && opinions[c.id].modules || {},note:opinions[c.id] && opinions[c.id].note || ''}))};
  }

  const API={normalizeData,maskAssets,reviewExport};
  if (typeof module!=='undefined' && module.exports) module.exports=API;
  if (typeof window==='undefined' || typeof document==='undefined') return;
  window.SEG_REVIEW_API=API;
  if (document.readyState==='loading') document.addEventListener('DOMContentLoaded',start); else start();

  function start() {
    const $=id=>document.getElementById(id);
    let data;
    try { data=normalizeData(window.SEG_REVIEW_DATA); } catch(error) { $('page-error').textContent=error.message;$('page-error').hidden=false;$('case-title').textContent='样本尚未就绪';return; }
    const state={index:0,part:'all',view:'white',threshold:'0',mode:'overlay',opacity:.55,zoom:1,centerX:0,centerY:0,
      raw:false,rawSee:[],rawM2f:[],hover:null,fit:true,request:0};
    const storageKey='segmentation-review-v1:'+data.cases.map(c=>c.source_sha256 || c.id).join(':');
    let opinions={};let storageAvailable=true;
    try { opinions=JSON.parse(localStorage.getItem(storageKey) || '{}');if (!opinions || typeof opinions!=='object') opinions={}; } catch(error) { storageAvailable=false; }
    const cache=new Map();const tintCache=new Map();const MAX_CACHED=12;
    const panes={};
    for (const model of ['see','m2f']) {
      const canvas=$(model+'-canvas');
      panes[model]={canvas,context:canvas.getContext('2d'),layers:[],base:null,cutout:null,available:true};
    }
    const current=()=>data.cases[state.index];
    const moduleName=id=>(data.modules.find(m=>m.id===id) || {}).name || id;

    function save() {
      try { localStorage.setItem(storageKey,JSON.stringify(opinions));$('save-status').textContent='判断已保存在当前浏览器'; }
      catch(error) { storageAvailable=false;$('save-status').textContent='浏览器未允许本地保存，请导出 JSON 留存'; }
      updateProgress();
    }

    function image(path) {
      if (cache.has(path)) {const value=cache.get(path);cache.delete(path);cache.set(path,value);return value;}
      const promise=new Promise((resolve,reject)=>{const img=new Image();img.onload=()=>resolve(img);img.onerror=()=>{cache.delete(path);reject(new Error('无法加载图片：'+path));};img.src=path;});
      cache.set(path,promise);
      while(cache.size>MAX_CACHED) cache.delete(cache.keys().next().value);
      return promise;
    }

    async function coloredImage(asset) {
      const img=await image(asset.path);
      if (!asset.color) return img;
      const key=asset.path+'|'+asset.color;
      if (tintCache.has(key)) return tintCache.get(key);
      const canvas=document.createElement('canvas');canvas.width=img.naturalWidth;canvas.height=img.naturalHeight;
      const context=canvas.getContext('2d');context.drawImage(img,0,0);context.globalCompositeOperation='source-in';context.fillStyle=asset.color;context.fillRect(0,0,canvas.width,canvas.height);
      tintCache.set(key,canvas);while(tintCache.size>4)tintCache.delete(tintCache.keys().next().value);
      return canvas;
    }

    function updateProgress() {
      const record=opinions[current().id] || {};const values=record.modules || {};
      const count=data.modules.filter(m=>values[m.id] && values[m.id].choice!=='pending').length;
      $('review-progress').textContent=count+' / '+data.modules.length;
      document.querySelectorAll('.sample-button').forEach(button=>{
        const record=opinions[button.dataset.id];const values=record && record.modules || {};
        const count=data.modules.filter(m=>values[m.id] && values[m.id].choice!=='pending').length;
        button.querySelector('.sample-reviewed').textContent=count ? '已判断 '+count+' / '+data.modules.length : '';
      });
    }

    function buildSamples() {
      const list=$('sample-list');
      data.cases.forEach((c,index)=>{
        const button=document.createElement('button');button.type='button';button.className='sample-button';button.dataset.id=c.id;
        const thumbnail=document.createElement('img');thumbnail.src=c.thumbnail || c.original;thumbnail.alt='';thumbnail.loading='lazy';
        const text=document.createElement('span');
        const title=document.createElement('span');title.className='sample-name';title.textContent=c.title || c.id;
        const id=document.createElement('span');id.className='sample-id';id.textContent=c.id;
        const reviewed=document.createElement('span');reviewed.className='sample-reviewed';
        text.append(title,id,reviewed);button.append(thumbnail,text);button.addEventListener('click',()=>chooseCase(index));list.append(button);
      });
      $('sample-count').textContent=data.cases.length+' 张';
    }

    function buildModules() {
      const root=$('module-tabs');
      for (const item of [{id:'all',name:'分组总览'},...data.modules,{id:'objects',name:'物体（See）'}]) {
        const button=document.createElement('button');button.type='button';button.className='part-tab';button.dataset.part=item.id;button.textContent=item.name;
        button.addEventListener('click',()=>{state.part=item.id;state.raw=false;updateControls();loadLayers();});root.append(button);
      }
      const reviews=$('review-choices');
      for (const item of data.modules) {
        const label=document.createElement('label');label.className='review-part';label.dataset.part=item.id;
        const name=document.createElement('span');name.textContent=item.name;
        const select=document.createElement('select');select.dataset.part=item.id;select.setAttribute('aria-label',item.name+'分割偏好');
        for (const [value,text] of CHOICES) {const option=document.createElement('option');option.value=value;option.textContent=text;select.append(option);}
        select.addEventListener('change',()=>{const id=current().id;opinions[id] ||= {modules:{},note:''};opinions[id].modules ||= {};opinions[id].modules[item.id]={choice:select.value,evidence_context:{view:state.view,see_threshold:Number(state.threshold)}};save();});
        label.append(name,select);reviews.append(label);
      }
    }

    function buildRawChannels(model) {
      const labels=model==='see' ? data.see_labels : data.m2f_labels;
      const list=$(model+'-channel-list');
      for (const channel of labels) {
        const label=document.createElement('label');label.className='channel-choice';
        const input=document.createElement('input');input.type='checkbox';input.value=channel.id;input.dataset.model=model;
        const swatch=document.createElement('i');swatch.className='channel-swatch';swatch.style.background=colorOf(channel);
        const text=document.createElement('span');text.textContent=channel.name;
        input.addEventListener('change',()=>{state.raw=true;state[model==='see'?'rawSee':'rawM2f']=Array.from(list.querySelectorAll('input:checked'),x=>Number(x.value));updateControls();loadLayers();});
        label.append(input,swatch,text);list.append(label);
      }
    }

    function updateControls() {
      document.querySelectorAll('.part-tab').forEach(button=>button.setAttribute('aria-pressed',String(!state.raw && button.dataset.part===state.part)));
      document.querySelectorAll('.review-part').forEach(label=>label.classList.toggle('active',label.dataset.part===state.part && !state.raw));
      const selected=data.modules.find(m=>m.id===state.part);
      const objects=!state.raw && state.part==='objects';
      $('part-note').textContent=state.raw ? '原始类别独立显示。两模型类别定义不同，请结合原图判断；同时勾选的重叠通道会混色，不代表互斥标签。' :
        (objects ? '显示 See-through 的 objects 原始独立通道，可与人物通道重叠；不扣人物、不填孔，也不作为第七个人物模块。Mask2Former 无独立物体类，右侧叠加或原图模式仅作原图参照。' :
        selected && selected.note || '这里只显示六模块映射，不等于完整人物蒙版。See-through 没有独立的全身裸肤类；手脚粗类、尾巴和翅膀请在原始通道中查看。');
      $('opacity-value').value=Math.round(state.opacity*100)+'%';
      $('overlay-opacity').disabled=state.mode!=='overlay';
      $('see-pane-detail').textContent=(state.raw?'原始通道':objects?'objects 原始通道':state.part==='all'?'六模块映射':moduleName(state.part))+' / 阈值 '+state.threshold;
      $('m2f-pane-detail').textContent=objects?'无独立物体类 · 原图参照':(state.raw?'原始类别':'768 × 768 / FP32')+' / '+(state.view==='white'?'白底':'原图');
      $('input-context').textContent=(state.view==='white'?'白底':'原图')+'输入预测，叠加底图为原图';
      const analysisInput=current().views && current().views[state.view] && current().views[state.view].input;
      if(analysisInput)$('analysis-input-link').href=analysisInput;else $('analysis-input-link').removeAttribute('href');
    }

    function chooseCase(index) {
      state.index=(index+data.cases.length)%data.cases.length;state.fit=true;state.hover=null;
      cache.clear();tintCache.clear();
      const c=current();state.centerX=c.width/2;state.centerY=c.height/2;
      $('case-title').textContent=c.title || c.id;$('case-description').textContent=c.description || '';
      $('case-position').textContent=(state.index+1)+' / '+data.cases.length;
      $('image-dimensions').textContent=c.width+' × '+c.height+' px';
      document.querySelectorAll('.sample-button').forEach(button=>button.setAttribute('aria-current',String(button.dataset.id===c.id)));
      const record=opinions[c.id] || {};
      document.querySelectorAll('.review-part select').forEach(select=>select.value=(record.modules && record.modules[select.dataset.part] || {}).choice || 'pending');
      $('review-notes').value=record.note || '';updateProgress();updateControls();loadLayers();
    }

    async function loadLayers() {
      const request=++state.request;const c=current();$('load-status').textContent='正在载入蒙版…';
      $('page-error').hidden=true;
      for (const pane of Object.values(panes)) {pane.layers=[];pane.base=null;pane.cutout=null;}
      drawAll();
      try {
        const base=await image(c.original);
        await Promise.all(['see','m2f'].map(async model=>{
          const selection=maskAssets(c,model,state.view,state.threshold,state.part,state.raw?(model==='see'?state.rawSee:state.rawM2f):null,data.modules,model==='see'?data.see_labels:data.m2f_labels);
          let layers=[];
          if(state.mode!=='original' && state.raw && selection.assets.length>1) {
            const merged=document.createElement('canvas');merged.width=c.width;merged.height=c.height;
            const mergedContext=merged.getContext('2d');mergedContext.globalCompositeOperation='screen';
            for(const asset of selection.assets) {
              const layer=await coloredImage(asset);if(request!==state.request)return;
              mergedContext.drawImage(layer,0,0,c.width,c.height);
            }
            layers=[merged];
          } else if(state.mode!=='original')layers=await Promise.all(selection.assets.map(coloredImage));
          if(request!==state.request)return;
          const pane=panes[model];pane.base=base;pane.layers=layers;pane.available=selection.available;
          if(state.mode==='cutout' && selection.available) {
            const crop=document.createElement('canvas');crop.width=c.width;crop.height=c.height;
            const cropContext=crop.getContext('2d');for(const layer of layers)cropContext.drawImage(layer,0,0,c.width,c.height);
            cropContext.globalCompositeOperation='source-in';cropContext.drawImage(base,0,0,c.width,c.height);pane.cutout=crop;
          }
          $(model+'-message').hidden=selection.reason==='no-object-class' ? state.mode==='overlay' || state.mode==='original' : selection.available || state.mode==='original';
          $(model+'-message').textContent=selection.message || '此样本没有'+(state.view==='white'?'白底':'原图')+'输入下的'+(model==='see'?'See-through':'Mask2Former')+'结果。';
        }));
        if(request!==state.request)return;
        if(state.fit)fitView();else drawAll();
        $('load-status').textContent=state.raw && !state.rawSee.length && !state.rawM2f.length ? '勾选通道开始对照' : '原始分割 · 未做融合修补';
      } catch(error) {
        if(request!==state.request)return;
        $('load-status').textContent='资源加载失败';$('page-error').textContent=error.message+'。请保持 assets 文件夹与页面的相对位置。';$('page-error').hidden=false;
      }
    }

    function fitView() {
      const c=current();const sizes=Object.values(panes).map(p=>p.canvas.getBoundingClientRect());
      state.zoom=Math.max(.01,Math.min(...sizes.map(r=>Math.min((r.width-28)/c.width,(r.height-28)/c.height))));
      state.centerX=c.width/2;state.centerY=c.height/2;state.fit=true;drawAll();
    }

    function drawAll() {
      const c=current();$('zoom-value').textContent=Math.round(state.zoom*100)+'%';
      for (const pane of Object.values(panes)) {
        const {canvas,context}=pane;const rect=canvas.getBoundingClientRect();const dpr=Math.min(window.devicePixelRatio || 1,2);
        const width=Math.max(1,Math.round(rect.width*dpr)),height=Math.max(1,Math.round(rect.height*dpr));
        if(canvas.width!==width || canvas.height!==height){canvas.width=width;canvas.height=height;}
        context.setTransform(dpr,0,0,dpr,0,0);context.clearRect(0,0,rect.width,rect.height);
        if(!pane.base)continue;
        context.save();context.translate(rect.width/2,rect.height/2);context.scale(state.zoom,state.zoom);context.translate(-state.centerX,-state.centerY);
        if(state.mode==='overlay' || state.mode==='original')context.drawImage(pane.base,0,0,c.width,c.height);
        if(pane.available && state.mode!=='original') {
          if(state.mode==='cutout') {
            if(pane.cutout)context.drawImage(pane.cutout,0,0);
          } else {context.globalAlpha=state.mode==='overlay'?state.opacity:1;for(const layer of pane.layers)context.drawImage(layer,0,0,c.width,c.height);context.globalAlpha=1;}
        }
        context.restore();
        if(state.hover){const x=rect.width/2+(state.hover.x-state.centerX)*state.zoom,y=rect.height/2+(state.hover.y-state.centerY)*state.zoom;context.save();context.lineWidth=1;context.strokeStyle='#ffffffc7';context.beginPath();context.moveTo(x-7,y);context.lineTo(x+7,y);context.moveTo(x,y-7);context.lineTo(x,y+7);context.stroke();context.restore();}
      }
    }

    function zoomAt(factor,canvas,event) {
      const rect=canvas.getBoundingClientRect();const x=event?event.clientX-rect.left-rect.width/2:0,y=event?event.clientY-rect.top-rect.height/2:0;
      const imageX=state.centerX+x/state.zoom,imageY=state.centerY+y/state.zoom;
      state.zoom=Math.max(.01,Math.min(8,state.zoom*factor));state.centerX=imageX-x/state.zoom;state.centerY=imageY-y/state.zoom;state.fit=false;drawAll();
    }

    for (const pane of Object.values(panes)) {
      let dragging=null;const canvas=pane.canvas;
      canvas.addEventListener('wheel',event=>{event.preventDefault();zoomAt(Math.exp(-event.deltaY*.0015),canvas,event);},{passive:false});
      canvas.addEventListener('pointerdown',event=>{if(event.button!==0)return;dragging={x:event.clientX,y:event.clientY};canvas.setPointerCapture(event.pointerId);canvas.classList.add('dragging');});
      canvas.addEventListener('pointermove',event=>{
        if(dragging){state.centerX-=(event.clientX-dragging.x)/state.zoom;state.centerY-=(event.clientY-dragging.y)/state.zoom;dragging={x:event.clientX,y:event.clientY};state.fit=false;}
        const rect=canvas.getBoundingClientRect();state.hover={x:state.centerX+(event.clientX-rect.left-rect.width/2)/state.zoom,y:state.centerY+(event.clientY-rect.top-rect.height/2)/state.zoom};
        $('pixel-position').textContent='原图坐标 '+Math.floor(state.hover.x)+', '+Math.floor(state.hover.y);drawAll();
      });
      const stop=()=>{dragging=null;canvas.classList.remove('dragging');};canvas.addEventListener('pointerup',stop);canvas.addEventListener('pointercancel',stop);
      canvas.addEventListener('pointerleave',()=>{if(!dragging){state.hover=null;$('pixel-position').textContent='两侧同步缩放与平移';drawAll();}});
      canvas.addEventListener('dblclick',fitView);
    }
    $('previous-case').addEventListener('click',()=>chooseCase(state.index-1));$('next-case').addEventListener('click',()=>chooseCase(state.index+1));
    $('fit-view').addEventListener('click',fitView);$('native-view').addEventListener('click',()=>{state.zoom=1;state.fit=false;drawAll();});
    $('input-view').addEventListener('change',event=>{state.view=event.target.value;updateControls();loadLayers();});
    $('display-mode').addEventListener('change',event=>{state.mode=event.target.value;updateControls();loadLayers();});
    $('see-threshold').addEventListener('change',event=>{state.threshold=event.target.value;updateControls();loadLayers();});
    $('overlay-opacity').addEventListener('input',event=>{state.opacity=Number(event.target.value)/100;updateControls();drawAll();});
    $('raw-toggle').addEventListener('click',()=>{const open=$('raw-channel-panel').hidden;$('raw-channel-panel').hidden=!open;$('raw-toggle').setAttribute('aria-expanded',String(open));});
    $('return-modules').addEventListener('click',()=>{state.raw=false;$('raw-channel-panel').hidden=true;$('raw-toggle').setAttribute('aria-expanded','false');updateControls();loadLayers();});
    $('review-toggle').addEventListener('click',()=>{const open=$('review-content').hidden;$('review-content').hidden=!open;$('review-toggle').setAttribute('aria-expanded',String(open));$('review-toggle').textContent=open?'收起记录':'展开记录';});
    $('review-notes').addEventListener('input',event=>{const id=current().id;opinions[id] ||= {modules:{},note:''};opinions[id].note=event.target.value;save();});
    $('export-review').addEventListener('click',()=>{const value=reviewExport(data,opinions,{view:state.view,see_threshold:Number(state.threshold),part:state.part});const blob=new Blob([JSON.stringify(value,null,2)],{type:'application/json;charset=utf-8'});const url=URL.createObjectURL(blob);const link=document.createElement('a');link.href=url;link.download='segmentation-preferences-'+new Date().toISOString().slice(0,10)+'.json';document.body.append(link);link.click();link.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);$('save-status').textContent='已导出 JSON；分割结果未修改';});
    document.addEventListener('keydown',event=>{if(event.target.closest('input,select,textarea,[contenteditable]'))return;if(event.key==='ArrowLeft'){event.preventDefault();chooseCase(state.index-1);}if(event.key==='ArrowRight'){event.preventDefault();chooseCase(state.index+1);}if(event.key==='0')fitView();if(event.key==='1'){state.zoom=1;state.fit=false;drawAll();}if(event.key==='+' || event.key==='=')zoomAt(1.25,panes.see.canvas);if(event.key==='-')zoomAt(.8,panes.see.canvas);});
    const observer=new ResizeObserver(()=>{if(state.fit)fitView();else drawAll();});for(const pane of Object.values(panes))observer.observe(pane.canvas);
    buildSamples();buildModules();buildRawChannels('see');buildRawChannels('m2f');chooseCase(0);
    if(!storageAvailable)$('save-status').textContent='浏览器未允许本地保存，请导出 JSON 留存';
  }
})();
