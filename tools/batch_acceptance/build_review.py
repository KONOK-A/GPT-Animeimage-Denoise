"""Export a traceable, fixed-selection denoising acceptance viewer.

All candidates come from exact declared numerical caches. PNGs produced here
are display artifacts, never input to a denoiser or to the numerical replay.
"""
import argparse
import colorsys
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import shutil
import sys

PROJECT=Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:sys.path.insert(0,str(PROJECT))
if str(PROJECT/'src') not in sys.path:sys.path.insert(0,str(PROJECT/'src'))

import numpy as np
from PIL import Image

from gpt_animeimage_denoise.filters import blend,gradient_preserving_candidate,validate_unit_interval
from gpt_animeimage_denoise.fusion_evaluation import _review_candidate
from gpt_animeimage_denoise.fusion_pipeline import OUTPUT_FILES,_payload_hash,_unpack,_validate_evidence
from gpt_animeimage_denoise.pipeline import _assert_inputs_safe,_case_folder,_digest,_read_source,_save_rgb,_sha
from tools.segmentation_review import build_assets as display
from tools.segmentation_review.build_full_comparison import fusion_groups,index_evidence,project_m2f

METHODS=('l0','esrgan-l0')
PARTS=tuple(entry['id'] for entry in display.MODULES)
MASK_FILES=('person_labels.png','raw_labels.png','objects.png','unknown.png','conflict.png',
            'consensus.png','protected.png','source_map.png',
            *('module_'+part+'.png' for part in PARTS))


def _cache_path(root,stage,identity):return Path(root)/(stage+'-'+_digest(identity)+'.npz')


def load_exact_candidate(cache_root,report,shape):
    if not isinstance(report.get('candidate_cache_identity'),dict) or not report.get('candidate_cache_stage'):
        raise ValueError('Acceptance requires a declared exact float candidate identity')
    return _review_candidate(cache_root,report,shape)


def load_exact_sr(cache_root,report,shape):
    identity=report.get('sr_cache_identity')
    if (not isinstance(identity,dict) or identity.get('source_sha256')!=report['source_sha256']
            or identity.get('device')!=report['parameters']['device']):
        raise ValueError('ESRGAN preview requires its exact source-bound SR cache identity')
    expected_config={key:value for key,value in report['parameters']['denoiser_config'].items() if key!='method'}
    if (identity.get('denoiser_config')!=expected_config or identity.get('precision')!='uint8-native-sr-cache'
            or identity.get('size')!=list(shape)):
        raise ValueError('SR cache recipe, precision or grid differs from the selected method')
    path=_cache_path(cache_root,'sr',identity)
    if not path.is_file():raise ValueError('Exact uint8 SR cache is unavailable; no reconstruction is allowed')
    with np.load(path,allow_pickle=False) as saved:
        metadata=json.loads(str(saved['metadata'].item()))
        arrays={name:saved[name] for name in saved.files if name!='metadata'}
        if metadata['identity']!=identity or metadata['payload_hash']!=_payload_hash(metadata['tree'],arrays):
            raise ValueError('SR cache identity or payload hash differs')
        value=_unpack(metadata['tree'],arrays)
    if not isinstance(value,np.ndarray) or value.dtype!=np.uint8 or value.shape!=shape:
        raise ValueError('SR cache must retain original-grid uint8 RGB')
    return value


def _finish_controls(parameters):
    return {key:validate_unit_interval(parameters.get(key,1.),key)
            for key in ('structure_protection','restoration_gain')}


def replay_final(rgb,candidate,weight,sigma,expected,*,restoration_gain=1.):
    restored=gradient_preserving_candidate(rgb,candidate,sigma,restoration_gain=restoration_gain)
    if not np.array_equal(blend(rgb,restored,weight),expected):
        raise ValueError('Exact float candidate and saved weight do not reproduce the saved final PNG')
    return restored


def load_method(folder,source_sha256,rgb,alpha,icc,method):
    folder=Path(folder).resolve();path=folder/'report.json'
    report=json.loads(path.read_text(encoding='utf-8'));parameters=report.get('parameters',{})
    if (report.get('status')!='complete' or report.get('profile')!='fusion'
            or report.get('source_sha256')!=source_sha256 or report.get('size')!=[rgb.shape[1],rgb.shape[0]]
            or parameters.get('denoiser')!=method):
        raise ValueError('Method report has a different source, method, grid or incomplete status')
    recorded=set(report.get('output_hashes',{}))
    expected=set(OUTPUT_FILES)
    if recorded not in (expected,expected-{'suspected_objects.png'}):
        raise ValueError('Native output inventory differs')
    inputs=[path,*[folder/name for name in OUTPUT_FILES if name in recorded]]
    for name,digest in report['output_hashes'].items():
        if not (folder/name).is_file() or _sha(folder/name)!=digest:
            raise ValueError('Native output hash differs: '+name)
    with Image.open(folder/'result.png') as image:
        final=np.array(image.convert('RGB'))
        if image.size!=(rgb.shape[1],rgb.shape[0]) or image.info.get('icc_profile')!=icc:
            raise ValueError('Output size or ICC was not preserved')
        if alpha is not None:
            if 'A' not in image.getbands() or not np.array_equal(np.array(image.getchannel('A')),alpha):
                raise ValueError('Original alpha was not preserved')
        elif 'A' in image.getbands():raise ValueError('A new alpha channel was introduced')
    weight=np.load(folder/'weight.npy',allow_pickle=False)
    if (weight.dtype!=np.float32 or weight.shape!=rgb.shape[:2] or not np.isfinite(weight).all()
            or np.any((weight<0)|(weight>1))):raise ValueError('Invalid original-grid float weight')
    if not np.array_equal(final[weight==0],rgb[weight==0]):raise ValueError('W=0 pixels changed')
    if alpha is not None and weight[alpha==0].any():raise ValueError('Transparent pixels received processing weight')
    labels=np.array(Image.open(folder/'person_labels.png'));groups=fusion_groups(labels)
    objects=np.array(Image.open(folder/'objects.png'))>0;unknown=np.array(Image.open(folder/'unknown.png'))>0
    protected=np.array(Image.open(folder/'protected.png'))>0
    if not np.array_equal(unknown,labels==255) or (labels[objects]!=0).any() or weight[objects|unknown].any():
        raise ValueError('Objects, unknown labels or their protection differ')
    if protected.shape!=weight.shape or weight[protected].any():raise ValueError('Protected pixels received processing weight')
    if not report.get('edits') and weight[~np.isin(labels,[1,2,3])].any():
        raise ValueError('Default processing expanded beyond skin, face and hair')
    for part in PARTS:
        mask=np.array(Image.open(folder/('module_'+part+'.png')))>0
        if not np.array_equal(mask,groups[part]) or (mask&objects).any():
            raise ValueError('Actual person modules are inconsistent: '+part)
    ident=report.get('candidate_cache_identity');stage=report.get('candidate_cache_stage')
    if not isinstance(ident,dict) or not stage:raise ValueError('Exact float candidate provenance missing')
    inputs.append(_cache_path(folder.parent/'.cache',stage,ident))
    if method=='esrgan-l0':
        if not isinstance(report.get('sr_cache_identity'),dict):raise ValueError('Exact ESRGAN stage identity missing')
        inputs.append(_cache_path(folder.parent/'.cache','sr',report['sr_cache_identity']))
    if any(not p.is_file() for p in inputs):raise ValueError('A consumed native artifact or numerical cache is missing')
    return {'folder':folder,'report':report,'final':final,'weight':weight,'labels':labels,
            'objects':objects,'unknown':unknown,'input_paths':inputs}


def validate_pair(pair):
    if set(pair)!=set(METHODS):raise ValueError('Both requested candidate methods are required')
    a,b=(pair[method] for method in METHODS);pa=a['report']['parameters'];pb=b['report']['parameters']
    for key in ('profile','lambda','sigma_at_2048','strength','device','models','implementation','policy','l0_implementation'):
        if pa.get(key)!=pb.get(key):raise ValueError('Unfair comparison; parameter differs: '+key)
    controls_b=_finish_controls(pb)
    for key,value in _finish_controls(pa).items():
        if value!=controls_b[key]:raise ValueError('Unfair comparison; parameter differs: '+key)
    if not isinstance(pa.get('lambda'),(int,float)) or not math.isfinite(pa['lambda']) or pa['lambda']<=0:
        raise ValueError('Both methods need the same finite positive L0 lambda')
    if a['report'].get('effective_sigma')!=b['report'].get('effective_sigma'):
        raise ValueError('Effective residual sigma differs')
    if a['report'].get('provenance')!=b['report'].get('provenance'):
        raise ValueError('The methods did not use identical model evidence provenance')
    if not np.array_equal(a['weight'],b['weight']):raise ValueError('The final float weights differ')
    for name in MASK_FILES:
        # Compare actual bytes as well as recorded hashes: an in-memory report
        # must not hide a changed source file between loading and exporting.
        first_hash=_sha(a['folder']/name);second_hash=_sha(b['folder']/name)
        if (first_hash!=a['report']['output_hashes'][name] or second_hash!=b['report']['output_hashes'][name]
                or first_hash!=second_hash):
            raise ValueError('The two methods used different segmentation or protection: '+name)
    return {'same_masks':True,'same_weight':True,'same_segmentation_policy':True,
            'protected_unchanged':True,'same_grid':True,'modules_objects_consistent':True}


def _targets(output,case_id,has_evidence):
    folder=_case_folder(Path(output)/'assets',case_id)
    names=['original.png','thumb.jpg','groups-colored.png','person-labels.png','source-map.png',
           'source-colored.png','weight.png','weight-gray.png','weight.npy','allowed.png','active.png','zero-weight.png',
           'protected.png','objects.png','unknown.png','conflict.png','pre-sam-colored.png','post-sam-colored.png']
    names += [part+'.png' for part in (*PARTS,'all')]
    names += ['difference-'+str(gain)+'.png' for gain in (1,4,8)]
    for method in METHODS:names += [method+'-'+stage+ext for stage,ext in [('final','.png'),('candidate','.png'),('restored','.png'),('report','.json'),('sr','.png'),('changed','.png')]]
    if has_evidence:
        names += ['white.png','input-original.png','alpha.png']
        for view in ('white','original'):
            for model in ('m2f','see'):
                names += [model+'-'+view+'-'+part+'.png' for part in (*PARTS,'all','colored','objects')]
    return folder,[folder/name for name in names]


def _alpha(path,weight,color=(74,154,230)):
    rgba=np.empty((*weight.shape,4),np.uint8);rgba[:,:,:3]=color
    rgba[:,:,3]=np.ceil(np.clip(weight,0,1)*255).astype(np.uint8)
    Image.fromarray(rgba).save(path,optimize=True)


def _colored_labels(path,labels,objects=None):
    groups=fusion_groups(labels)
    masks=[groups[part] for part in PARTS]+[labels==255]
    colors=[display.COLORS[part] for part in PARTS]+['#e7b849']
    if objects is not None:masks.append(objects);colors.append('#e9694f')
    display._colored(path,masks,colors)


def _source_colored(path,source,codes):
    masks=[];colors=[];legend=[]
    names={int(value):key for key,value in codes.items()}
    for value in np.unique(source):
        value=int(value)
        if value==0:continue
        rgb=tuple(round(c*255) for c in colorsys.hsv_to_rgb((value*.61803398875)%1,.60,.94))
        color='#'+''.join(f'{v:02x}' for v in rgb)
        masks.append(source==value);colors.append(color);legend.append({'code':value,'name':names.get(value,str(value)),'color':color})
    if masks:display._colored(path,masks,colors)
    else:display._mask(path,np.zeros(source.shape,bool))
    return legend


def _trace_inputs(run_root,case_id):
    if run_root is None:return []
    root=Path(run_root)/'trace'/case_id
    return [p.resolve() for method in METHODS for step in ('pre-sam','post-sam','weight')
            for p in (root/method/step).glob('*') if p.is_file()]


def export_case(item,output_root,pair,*,evidence=None,evidence_path=None,run_root=None,input_paths=()):
    output=Path(output_root).resolve();source=Path(item['path']).resolve()
    rgb,alpha,icc=_read_source(source)
    if _sha(source)!=item['sha256']:raise ValueError('Source SHA differs from frozen input manifest')
    checks=validate_pair(pair);shared=pair['l0'];weight=shared['weight'];labels=shared['labels']
    if labels.shape!=rgb.shape[:2]:raise ValueError('Shared person grid differs from input')
    folder,targets=_targets(output,item['id'],evidence is not None)
    inputs=[source,*map(Path,input_paths),*[p for value in pair.values() for p in value['input_paths']],
            *_trace_inputs(run_root,item['id'])]
    if evidence_path is not None:inputs.append(Path(evidence_path))
    _assert_inputs_safe(targets,inputs);snapshots={str(p):_sha(p) for p in inputs}
    # Reconstruct both exact numerical stages before publishing any assets.
    # Only small RGB candidates are retained here, not raw19 evidence copies.
    replay={}
    for method in METHODS:
        value=pair[method];report=value['report'];cache=value['folder'].parent/'.cache'
        candidate=load_exact_candidate(cache,report,rgb.shape)
        gain=report['parameters'].get('restoration_gain',1.)
        restored=replay_final(rgb,candidate,weight,report['effective_sigma'],value['final'],restoration_gain=gain)
        replay[method]=(candidate,restored)
    sr=load_exact_sr(pair['esrgan-l0']['folder'].parent/'.cache',pair['esrgan-l0']['report'],rgb.shape)
    folder.mkdir(parents=True,exist_ok=True)
    rel=lambda p:p.relative_to(output).as_posix()
    _save_rgb(folder/'original.png',rgb,alpha,icc)
    thumb=Image.fromarray(rgb);thumb.thumbnail((240,240),Image.Resampling.LANCZOS);thumb.save(folder/'thumb.jpg',quality=88)
    groups=fusion_groups(labels);group_paths={}
    for part,mask in groups.items():
        p=folder/(part+'.png');display._mask(p,mask);group_paths[part]=rel(p)
    _colored_labels(folder/'groups-colored.png',labels,shared['objects'])
    protected=np.array(Image.open(shared['folder']/'protected.png'))>0
    allowed=np.isin(labels,[1,2,3])&~protected&~shared['objects']&~shared['unknown']
    if run_root is not None:
        actual=Path(run_root)/'trace'/item['id']/'l0/weight/allowed.png'
        if actual.is_file():
            with Image.open(actual) as image:allowed=np.array(image)>0
            if allowed.shape!=weight.shape or weight[~allowed].any() or (allowed&protected).any():
                raise ValueError('Saved build_weight allowed argument is inconsistent with final weights')
    for name,mask in [('allowed',allowed),('active',weight>0),('zero-weight',weight==0),('protected',protected),('objects',shared['objects']),
                      ('unknown',shared['unknown']),('conflict',np.array(Image.open(shared['folder']/'conflict.png'))>0)]:
        display._mask(folder/(name+'.png'),mask)
    _alpha(folder/'weight.png',weight)
    shutil.copyfile(shared['folder']/'weight.png',folder/'weight-gray.png')
    shutil.copyfile(shared['folder']/'weight.npy',folder/'weight.npy')
    shutil.copyfile(shared['folder']/'person_labels.png',folder/'person-labels.png')
    shutil.copyfile(shared['folder']/'source_map.png',folder/'source-map.png')
    source_map=np.array(Image.open(shared['folder']/'source_map.png'))
    legend=_source_colored(folder/'source-colored.png',source_map,shared['report']['parameters'].get('policy',{}).get('source_codes',{}))
    shared_assets={name:rel(folder/file) for name,file in [('colored','groups-colored.png'),('labels','person-labels.png'),
        ('source_map','source-map.png'),('source_colored','source-colored.png'),('weight','weight.png'),
        ('weight_gray','weight-gray.png'),('weight_float','weight.npy'),('allowed','allowed.png'),('active','active.png'),
        ('zero_weight','zero-weight.png'),('protected','protected.png'),('objects','objects.png'),('unknown','unknown.png'),('conflict','conflict.png')]}
    shared_assets.update(groups=group_paths,source_legend=legend,
        weight_note='RGBA仅显示实际浮点W的预览；最终作用范围以weight.npy为准，不是噪点识别图。')
    methods={}
    for method in METHODS:
        value=pair[method];candidate,restored=replay[method]
        final_path=folder/(method+'-final.png');shutil.copyfile(value['folder']/'result.png',final_path)
        for name,array in [('candidate',candidate),('restored',restored)]:
            _save_rgb(folder/(method+'-'+name+'.png'),np.rint(array*255).astype(np.uint8),alpha,icc)
        report_path=folder/(method+'-report.json');shutil.copyfile(value['folder']/'report.json',report_path)
        change=np.any(value['final']!=rgb,axis=2);display._mask(folder/(method+'-changed.png'),change)
        methods[method]={'final':rel(final_path),'candidate':rel(folder/(method+'-candidate.png')),
            'restored':rel(folder/(method+'-restored.png')),'report':rel(report_path),
            'report_fingerprint':value['report']['fingerprint'],'parameters':deepcopy(value['report']['parameters']),
            'changed':rel(folder/(method+'-changed.png')),
            'candidate_note':'浮点候选的8位显示预览：未Gaussian恢复、未蒙版权重混合，不是最终图。',
            'restored_note':'Gaussian恢复后候选的8位显示预览：尚未按W与原图混合。',
            'exact_float_replay_verified':True}
    _save_rgb(folder/'esrgan-l0-sr.png',sr,alpha,icc)
    methods['esrgan-l0']['sr']=rel(folder/'esrgan-l0-sr.png')
    methods['esrgan-l0']['sr_note']='60%预缩→ESRGAN超分→恢复原尺寸；未L0、未Gaussian、未蒙版混合。'
    differences={}
    delta=np.abs(pair['l0']['final'].astype(np.int16)-pair['esrgan-l0']['final'].astype(np.int16))
    for gain in (1,4,8):
        p=folder/('difference-'+str(gain)+'.png');Image.fromarray(np.clip(delta*gain,0,255).astype(np.uint8)).save(p,optimize=True)
        differences[str(gain)]=rel(p)
    checks.update(source_unchanged=True,float_replay_verified=True,
        alpha={'passed':True,'status':'verified' if alpha is not None else 'not_present',
               'detail':'原alpha逐像素保留' if alpha is not None else '原图未含alpha通道'},
        icc={'passed':True,'status':'verified' if icc is not None else 'not_present',
             'detail':'原ICC字节保留' if icc is not None else '原图未含ICC配置'})
    entry={'id':item['id'],'title':item.get('title',Path(item['path']).name),'source_sha256':item['sha256'],
        'width':rgb.shape[1],'height':rgb.shape[0],'original':rel(folder/'original.png'),
        'thumbnail':rel(folder/'thumb.jpg'),'methods':methods,'shared':shared_assets,
        'differences':differences,'difference_note':'两张真实最终RGB的绝对像素差异，显示倍率1/4/8；非原始颜色或质量评分。',
        'checks':checks,'trace':[]}
    if evidence is not None:_export_evidence(entry,folder,output,evidence,rgb,alpha,icc)
    _trace(entry,folder,output,run_root,pair)
    if any(_sha(p)!=digest for p,digest in snapshots.items()):raise RuntimeError('A source or native artifact changed during export')
    return entry


def _export_evidence(entry,folder,output,evidence,rgb,alpha,icc):
    _validate_evidence(evidence,rgb.shape[:2],True)
    if evidence.get('see_original_logits') is None:raise ValueError('Acceptance evidence lacks original See view')
    rel=lambda p:p.relative_to(output).as_posix()
    analysis=rgb
    if alpha is not None:
        a=alpha[...,None].astype(np.float32)/255;analysis=np.rint(rgb*a+255*(1-a)).astype(np.uint8)
    sky=evidence['alpha'];white=np.rint(analysis.astype(np.float32)*sky[...,None]+255*(1-sky[...,None])).clip(0,255).astype(np.uint8)
    _save_rgb(folder/'white.png',white,None,icc);_save_rgb(folder/'input-original.png',analysis,None,icc)
    Image.fromarray(np.rint(sky*255).astype(np.uint8)).save(folder/'alpha.png',optimize=True)
    entry['shared'].update(white=rel(folder/'white.png'),input_original=rel(folder/'input-original.png'),alpha=rel(folder/'alpha.png'),evidence={})
    threshold=entry['methods']['l0']['parameters']['policy'].get('see',1.)
    for view in ('white','original'):
        labels=project_m2f(evidence['m2f_'+view],rgb.shape[:2],'score')
        m2f=display.m2f_groups(labels)
        logits=evidence['see_logits' if view=='white' else 'see_original_logits']
        see=display.see_groups(logits,threshold)
        # The independent eye contour has its own >0 display threshold, as in
        # the production contour mechanism; this is not the fused eye result.
        see['eyes']=logits[3]>0;see['all']=np.logical_or.reduce([see[p] for p in PARTS])
        evidence_view={}
        for model,groups in [('m2f',m2f),('see',see)]:
            prefix=model+'-'+view;paths={}
            for part,mask in groups.items():
                p=folder/(prefix+'-'+part+'.png');display._mask(p,mask);paths[part]=rel(p)
            p=folder/(prefix+'-colored.png');display._colored(p,[groups[p] for p in PARTS],[display.COLORS[p] for p in PARTS])
            evidence_view[model]={'groups':paths,'groups_colored':rel(p),
                'note':'原始分数投影；未进行双模型修正或扣除外物' if model=='m2f' else
                       f'独立See通道映射，强证据>{threshold:g}、眼轮廓>0；可重叠，皮肤仅neck，不强并手脚粗类'}
        p=folder/('see-'+view+'-objects.png');display._mask(p,logits[18]>threshold)
        evidence_view['see']['objects']=rel(p)
        entry['shared']['evidence'][view]=evidence_view


def _trace(entry,folder,output,run_root,pair):
    local=lambda p:Path(os.path.relpath(p,output)).as_posix()
    link=lambda label,p:{'label':label,'path':local(p)}
    shared=entry['shared'];methods=entry['methods']
    entries=[{'stage':'input','title':'输入与身份','status':'complete','detail':'EXIF显示方向归一；原图保持不变。',
              'links':[link('原图显示副本',output/entry['original'])]},
        {'stage':'evidence','title':'原始模型证据','status':'complete','detail':'同一份模型证据供两方法使用，单输入证据不冒充融合。','links':[]},
        {'stage':'regions','title':'融合分区与SAM','status':'complete','detail':'两方法保存的实际标签、外物、来源与保护图完全相同。',
         'links':[link('融合诊断记录',pair['l0']['folder']/'regions.json'),link('真实人物标签',output/shared['labels'])]},
        {'stage':'weight','title':'最终作用权重','status':'complete','detail':'weight.npy为实际float32最终W；PNG仅预览，W=0逐像素保留。',
         'links':[link('浮点最终权重',output/shared['weight_float']),link('灰度权重预览',output/shared['weight_gray'])]},
        {'stage':'candidate','title':'降噪数值候选','status':'complete','detail':'精确读取报告声明的浮点缓存；显示PNG不参与生产重放。','links':[]},
        {'stage':'final','title':'最终出图','status':'complete','detail':'原图+Gaussian恢复候选+同一W在线性RGB中混合；两方法唯一变量为候选算法。','links':[]},
        {'stage':'checks','title':'工程检查与验收','status':'complete','detail':'检查通过不等于视觉质量已由用户验收；请记录偏好与问题。','links':[]}]
    by_stage={v['stage']:v for v in entries}
    if shared.get('white'):by_stage['evidence']['links'].append(link('共同白底分析图',output/shared['white']))
    for view,models in shared.get('evidence',{}).items():
        for model,data in models.items():by_stage['evidence']['links'].append(link(model+' '+view+' 原始部位',output/data['groups_colored']))
    for method in METHODS:
        value=pair[method];r=value['report']
        by_stage['candidate']['links'] += [link(method+' 浮点候选缓存',_cache_path(value['folder'].parent/'.cache',r['candidate_cache_stage'],r['candidate_cache_identity'])),
                                          link(method+' 候选显示图',output/methods[method]['candidate'])]
        by_stage['final']['links'] += [link(method+' 最终图',output/methods[method]['final']),link(method+' 完整报告',output/methods[method]['report'])]
    by_stage['candidate']['links'].append(link('ESRGAN单步（尚未L0）',output/methods['esrgan-l0']['sr']))
    if run_root is not None:
        run=Path(run_root)
        for relative in ('input-manifest.json','run-manifest.json','validation.json','log/run-config.json'):
            p=run/relative
            if p.is_file():by_stage['checks']['links'].append(link(relative,p))
        for method in METHODS:
            for step in ('pre-sam','post-sam','weight'):
                path=run/'trace'/entry['id']/method/step
                if not path.is_dir():continue
                stage='weight' if step=='weight' else 'regions'
                for p in sorted(path.iterdir()):
                    if p.is_file():by_stage[stage]['links'].append(link(method+' '+step+' '+p.name,p))
                if step in ('pre-sam','post-sam') and (path/'person_labels.png').is_file():
                    label_array=np.array(Image.open(path/'person_labels.png'))
                    obj=np.array(Image.open(path/'objects.png'))>0 if (path/'objects.png').is_file() else None
                    if method=='l0':
                        target=folder/(step+'-colored.png');_colored_labels(target,label_array,obj)
                        entry['shared'][step.replace('-','_')+'_colored']=local(target)
    entry['trace']=entries


def write_manifest(output_root,cases,*,run_id,parameters=None,inputs=()):
    output=Path(output_root).resolve();targets=[output/'manifest.json',output/'manifest.js',output/'export-validation.json']
    _assert_inputs_safe(targets,inputs)
    data={'schema_version':1,'kind':'fixed-mask-denoise-batch-acceptance','run_id':run_id,
        'parameters':parameters or {},'summary':{'cases':len(cases),'methods':list(METHODS),
            'same_selection_for_both_methods':True,'engineering_checks_passed':True,'user_quality_accepted':False},
        'modules':display.MODULES,'cases':cases,
        'notes':['候选PNG仅供查看；最终图通过原始float候选与同一W重放验证。',
                 '差异放大不是原始颜色或质量分数；验收意见不自动修改模型或策略。']}
    output.mkdir(parents=True,exist_ok=True);payload=json.dumps(data,ensure_ascii=False,indent=2)
    targets[0].write_text(payload,encoding='utf-8');targets[1].write_text('window.BATCH_ACCEPTANCE_DATA = '+payload+';\n',encoding='utf-8')
    targets[2].write_text(json.dumps({'cases':len(cases),'same_masks_and_weight':True,'exact_float_replay':True,
        'source_preserved':True,'user_quality_accepted':False,'browser_runtime_verified':False},indent=2),encoding='utf-8')
    return data


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--out',type=Path);args=parser.parse_args(argv)
    run=args.run.resolve();output=(args.out or run).resolve();native=run/'native'
    if output==native or output.is_relative_to(native):raise ValueError('Review output must not be inside native results')
    manifest_path=run/'input-manifest.json';document=json.loads(manifest_path.read_text(encoding='utf-8'))
    items=document['items'];index=index_evidence(native/'.cache');jobs=[];inputs=[manifest_path]
    config_path=run/'log/run-config.json';parameters={}
    if config_path.is_file():
        config=json.loads(config_path.read_text(encoding='utf-8'));parameters=config.get('parameters',{})
        inputs.append(config_path)
    if len({item['id'] for item in items})!=len(items):raise ValueError('Duplicate input IDs')
    for item in items:
        source=Path(item['path']).resolve();rgb,alpha,icc=_read_source(source)
        if _sha(source)!=item['sha256']:raise ValueError('Input source no longer matches frozen manifest')
        pair={method:load_method(native/(item['id']+'--'+method),item['sha256'],rgb,alpha,icc,method) for method in METHODS}
        validate_pair(pair);p=pair['l0']['report']['parameters']
        for key in ('lambda','sigma_at_2048','strength'):
            expected=parameters.get(key,parameters.get('l0') if key=='lambda' else None)
            if expected is not None and p[key]!=expected:raise ValueError('Report differs from run config: '+key)
        if not config_path.is_file() and not jobs:
            parameters.update(_finish_controls(p))
        if _finish_controls(p)!=_finish_controls(parameters):
            raise ValueError('Report line protection or restoration gain differs from the run')
        matches=index.get((item['sha256'],p['models']),[])
        if len(matches)!=1:raise ValueError('Need one exact dual-input evidence cache: '+item['id'])
        jobs.append((item,matches[0]));inputs += [source,matches[0],*[q for v in pair.values() for q in v['input_paths']],*_trace_inputs(run,item['id'])]
        del pair,rgb
    targets=[output/'manifest.json',output/'manifest.js',output/'export-validation.json']
    for item,_ in jobs:targets += _targets(output,item['id'],True)[1]
    _assert_inputs_safe(targets,inputs)
    entries=[]
    for item,cache in jobs:
        print('EXPORT_ACCEPTANCE',item['id'],flush=True)
        rgb,alpha,icc=_read_source(item['path'])
        pair={method:load_method(native/(item['id']+'--'+method),item['sha256'],rgb,alpha,icc,method) for method in METHODS}
        ev=display.load_evidence_cache(cache,expected_source_sha256=item['sha256'],expected_shape=rgb.shape[:2])
        _validate_evidence(ev,rgb.shape[:2],True)
        report=pair['l0']['report'];identity=ev['_review_cache_identity']
        if (identity['models']!=report['parameters']['models'] or identity.get('device')!=report['parameters']['device']
                or ev['provenance'].get('model_hashes')!=report['provenance'].get('model_hashes')):
            raise ValueError('Displayed evidence does not match run model provenance')
        entries.append(export_case(item,output,pair,evidence=ev,evidence_path=cache,run_root=run,input_paths=[manifest_path]))
        del ev,pair,rgb
    write_manifest(output,entries,run_id=document.get('run_id',run.name),parameters=parameters,inputs=inputs)
    print('EXPORTED_BATCH_ACCEPTANCE',len(entries),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
