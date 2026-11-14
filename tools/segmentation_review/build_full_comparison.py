"""Export raw model views and a verified, fixed fusion result for local HTML.

This module performs no inference, denoising or fusion decisions. Difference
views are composed by the viewer from transparent masks, without pixel reads.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np
from PIL import Image

PROJECT=Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:sys.path.insert(0,str(PROJECT))
if str(PROJECT/'src') not in sys.path:sys.path.insert(0,str(PROJECT/'src'))

from tools.segmentation_review import build_assets as base
from gpt_animeimage_denoise.fusion_pipeline import _code_hash,_validate_evidence
from gpt_animeimage_denoise.pipeline import _assert_inputs_safe,_case_folder,_digest,_read_source,_sha

PARTS=tuple(entry['id'] for entry in base.MODULES)
FUSION_FILES=('person_labels.png','objects.png','unknown.png','conflict.png','source_map.png',
              *('module_'+part+'.png' for part in PARTS),'regions.json')
DIFF_COLORS={'m2f_only':'#3572d0','see_only':'#ed9738',
             'fusion_added':'#2ba878','fusion_removed':'#e768a8'}


def project_m2f(scores,shape,projection='score'):
    """Return raw12 labels; interpolate log scores before argmax for score mode."""
    scores=np.asarray(scores)
    if (scores.dtype!=np.float32 or scores.ndim!=3 or scores.shape[0]!=12
            or not scores.size or not np.isfinite(scores).all()):
        raise ValueError('M2F needs finite native float32 raw12 log scores')
    if len(shape)!=2 or min(shape)<1:raise ValueError('Invalid original image grid')
    h,w=map(int,shape)
    if projection=='nearest':
        native=scores.argmax(axis=0).astype(np.uint8)
        return np.asarray(Image.fromarray(native).resize((w,h),Image.Resampling.NEAREST)).copy()
    if projection!='score':raise ValueError('Projection must be score or nearest')
    # Streaming is equivalent to argmax(stack(resized scores)), but avoids a
    # 12-channel original-grid temporary on the largest images.
    best=np.full((h,w),-np.inf,np.float32);labels=np.zeros((h,w),np.uint8)
    for index,plane in enumerate(scores):
        up=plane if plane.shape==(h,w) else cv2.resize(plane,(w,h),interpolation=cv2.INTER_LINEAR)
        take=up>best;best[take]=up[take];labels[take]=index
    return labels


def fusion_groups(labels):
    labels=np.asarray(labels)
    if (labels.ndim!=2 or not labels.size or labels.dtype!=np.uint8
            or not np.isin(labels,[0,1,2,3,4,5,6,255]).all()):
        raise ValueError('Fusion labels must be original-grid uint8 0..6 or255')
    result={part:labels==number for number,part in enumerate(PARTS,1)}
    result['all']=(labels>0)&(labels<7)
    return result


def load_fusion_result(folder,source_sha256,shape):
    """Bind all consumed masks to the actual completed native report."""
    folder=Path(folder).resolve();report_path=folder/'report.json'
    report=json.loads(report_path.read_text(encoding='utf-8'))
    if (report.get('status')!='complete' or report.get('profile')!='fusion'
            or report.get('source_sha256')!=source_sha256
            or report.get('size')!=[shape[1],shape[0]]):
        raise ValueError('Fusion report is incomplete, wrong source/profile or wrong grid')
    inputs=[report_path,*[folder/name for name in FUSION_FILES]]
    for name in FUSION_FILES:
        if report.get('output_hashes',{}).get(name)!=_sha(folder/name):
            raise ValueError('Native fusion output hash differs: '+name)
    values={}
    for name,key in [('person_labels','labels'),('objects','objects'),('unknown','unknown'),
                     ('conflict','conflict'),('source_map','source_map')]:
        with Image.open(folder/(name+'.png')) as image:array=np.asarray(image).copy()
        if array.dtype!=np.uint8 or array.shape!=tuple(shape):
            raise ValueError('Native fusion mask must be uint8 on original grid: '+name)
        values[key]=array if key in ('labels','source_map') else array>0
    groups=fusion_groups(values['labels'])
    if not np.array_equal(values['unknown'],values['labels']==255):
        raise ValueError('Fusion unknown is inconsistent with final labels')
    if np.any(values['labels'][values['objects']]!=0):
        raise ValueError('Fusion object pixels overlap person labels')
    for part in PARTS:
        with Image.open(folder/('module_'+part+'.png')) as image:mask=np.asarray(image)>0
        if mask.shape!=tuple(shape) or not np.array_equal(mask,groups[part]):
            raise ValueError('Fusion module is not derived from final labels: '+part)
    values.update(report=report,regions=json.loads((folder/'regions.json').read_text(encoding='utf-8')),
                  input_paths=inputs,snapshots={str(path):_sha(path) for path in inputs})
    return values


def _extra_targets(output,case_id):
    folder=_case_folder(Path(output)/'assets',case_id)
    names=[]
    for view in ('white','original'):
        prefix='m2f-'+view+'-score'
        names += [prefix+'-labels.png',prefix+'-colored.png',prefix+'-groups-colored.png']
        names += [prefix+'-raw-'+str(i).zfill(2)+'.png' for i in range(12)]
        names += [prefix+'-'+part+'.png' for part in base.GROUP_IDS]
    names += ['fusion-'+part+'.png' for part in base.GROUP_IDS]
    names += ['fusion-labels.png','fusion-objects.png','fusion-unknown.png',
              'fusion-conflict.png','fusion-source-map.png','fusion-groups-colored.png']
    return [folder/name for name in names]


def _write_projection(folder,relative,scores,shape):
    labels=project_m2f(scores,shape,'score');prefix=folder.name
    label_path=folder.with_name(prefix+'-labels.png');Image.fromarray(labels).save(label_path,optimize=True)
    masks=[];paths=[]
    for index in range(12):
        mask=labels==index;masks.append(mask);path=folder.with_name(prefix+'-raw-'+str(index).zfill(2)+'.png')
        base._mask(path,mask);paths.append(relative(path))
    groups=base.m2f_groups(labels);group_paths={}
    for part,mask in groups.items():
        path=folder.with_name(prefix+'-'+part+'.png');base._mask(path,mask);group_paths[part]=relative(path)
    colored=folder.with_name(prefix+'-colored.png');base._colored(colored,masks[1:],base.M2F_COLORS[1:])
    grouped=folder.with_name(prefix+'-groups-colored.png')
    base._colored(grouped,[groups[part] for part in PARTS],[base.COLORS[part] for part in PARTS])
    return {'labels':relative(label_path),'masks':paths,'groups':group_paths,
            'colored':relative(colored),'groups_colored':relative(grouped),
            'postprocessing':'bilinear native log scores to original grid, then argmax; no semantic fusion'}


def export_full_case(case,output_root,evidence,fusion,*,evidence_path=None,input_paths=()):
    output=Path(output_root).resolve();source=Path(case['path']).resolve()
    if fusion['report']['source_sha256']!=case['sha256']:
        raise ValueError('Fusion belongs to another source')
    if evidence.get('see_original_logits') is None:
        raise ValueError('Full comparison requires original-input See evidence')
    shape=fusion['labels'].shape
    if tuple(evidence['alpha'].shape)!=shape:raise ValueError('Model/fusion pixel grids differ')
    inputs=[source,*fusion['input_paths'],*map(Path,input_paths)]
    if evidence_path is not None:inputs.append(Path(evidence_path))
    targets=[*base._paths_for_case(output,case['id'],True)[1],*_extra_targets(output,case['id'])]
    _assert_inputs_safe(targets,inputs)
    entry=base.export_case(case,output,evidence,see_original=evidence['see_original_logits'],
                           evidence_path=evidence_path,input_paths=inputs)
    folder=output/'assets'/case['id']
    def relative(path):return path.relative_to(output).as_posix()
    for view in ('white','original'):
        nearest=entry['views'][view]['m2f']
        nearest['postprocessing']='native argmax then nearest to original grid; reference display baseline'
        score=_write_projection(folder/('m2f-'+view+'-score'),relative,evidence['m2f_'+view],shape)
        entry['views'][view]['m2f']={'projections':{'score':score,'nearest':nearest}}
    groups=fusion_groups(fusion['labels']);paths={}
    for part,mask in groups.items():
        path=folder/('fusion-'+part+'.png');base._mask(path,mask);paths[part]=relative(path)
    final_files={}
    for key in ('labels','objects','unknown','conflict','source_map'):
        path=folder/('fusion-'+key.replace('_','-')+'.png')
        if key in ('labels','source_map'):Image.fromarray(fusion[key]).save(path,optimize=True)
        else:base._mask(path,fusion[key])
        final_files[key]=relative(path)
    colored=folder/'fusion-groups-colored.png'
    base._colored(colored,[groups[part] for part in PARTS],[base.COLORS[part] for part in PARTS])
    report=fusion['report']
    entry['fusion']={**final_files,'groups':paths,'colored':relative(colored),'groups_colored':relative(colored),
        'report_fingerprint':report['fingerprint'],'policy':report.get('parameters',{}).get('policy',{}),
        'semantic_repairs':fusion['regions'].get('semantic_repairs',{}),
        'fixed':True,'input_views':['white','original'],
        'note':'固定真实fusion结果；浏览器中的单视图、投影和See阈值不重新运行fusion。'}
    entry['m2f_postprocessing']={'score':'score-first, same raw projection used by fusion',
                               'nearest':'argmax-then-nearest reference baseline'}
    if any(_sha(path)!=digest for path,digest in fusion['snapshots'].items()):
        raise RuntimeError('Native fusion outputs changed during export')
    if _sha(source)!=case['sha256']:raise RuntimeError('Source changed during export')
    return entry


def write_manifest(output_root,cases,*,input_paths=()):
    output=Path(output_root).resolve();targets=[output/'manifest.json',output/'manifest.js']
    inputs=[*map(Path,input_paths),*[Path(c['source']) for c in cases]]
    _assert_inputs_safe(targets,inputs)
    document={'schema_version':1,'kind':'model-and-fixed-fusion-comparison','cases':cases,
        'modules':deepcopy(base.MODULES),
        'm2f_labels':[{'id':i,'name':name,'color':base.M2F_COLORS[i]} for i,name in enumerate(base.M2F_NAMES)],
        'see_labels':[{'id':i,'name':name,'color':base.SEE_COLORS[i]} for i,name in enumerate(base.SEE_NAMES)],
        'thresholds':[0,1,2],
        'defaults':{'part':'outfit','view':'white','projection':'score','see_threshold':1},
        'meta':{'diff_colors':DIFF_COLORS,
            'projection_info':{'score':'先双线性还原12类log分数，再判类；对齐fusion的原始投影方式',
                               'nearest':'先在768网格判类，再最近邻放大；旧reference显示基线'},
            'fusion':'固定已完成的双输入、锚定规则与SAM结果；服饰/外物强证据阈值1，眼睛另有>0锚定轮廓规则',
            'differences':'仅表示相对当前单视图的范围分歧，不是真值误差，也不全是See的因果贡献；未知区不能算作已确认移除。',
            'purpose':'只审阅原始模型和固定fusion；未训练、重新推理或更改融合策略。'}}
    output.mkdir(parents=True,exist_ok=True)
    payload=json.dumps(document,ensure_ascii=False,indent=2,allow_nan=False)
    targets[0].write_text(payload,encoding='utf-8')
    targets[1].write_text('window.SEG_FULL_COMPARISON_DATA = '+payload+';\n',encoding='utf-8')
    return document


def index_evidence(cache_root):
    expected_code=_code_hash(['fusion_models.py','fusion_pipeline.py','pipeline.py'])
    result={}
    for path in sorted(Path(cache_root).glob('evidence-*.npz')):
        with np.load(path,allow_pickle=False) as saved:metadata=json.loads(str(saved['metadata'].item()))
        identity=metadata['identity']
        if (identity.get('code')!=expected_code or identity.get('dual_view') is not True
                or identity.get('see_views')!=['white','original']):continue
        if path.name!='evidence-'+_digest(identity)+'.npz':
            raise ValueError('Evidence filename does not bind its declared identity')
        key=(identity.get('source_sha256'),identity.get('models'))
        result.setdefault(key,[]).append(path)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--titles',type=Path)
    args=parser.parse_args(argv);native=args.native.resolve();output=args.out.resolve()
    if output==native or output.is_relative_to(native):
        raise ValueError('Comparison output must be outside read-only native results')
    manifest_path=native/'input-manifest.json'
    items=json.loads(manifest_path.read_text(encoding='utf-8'))['items']
    if not items or any(x.get('status')!='eligible' or x.get('split')!='tune' for x in items):
        raise ValueError('This research comparison accepts eligible development cases only')
    if len({x['id'] for x in items})!=len(items):raise ValueError('Duplicate case IDs')
    if (output/'manifest.json').exists():
        existing=json.loads((output/'manifest.json').read_text(encoding='utf-8'))
        if existing.get('kind')!='model-and-fixed-fusion-comparison':
            raise FileExistsError('Keep previous comparison intact; choose another output directory')
    descriptions={}
    if args.titles:
        previous=json.loads(args.titles.read_text(encoding='utf-8'))
        descriptions={c['id']:(c.get('title',c['id']),c.get('description','')) for c in previous['cases']}
    cache_index=index_evidence(native/'.cache');jobs=[];all_inputs=[manifest_path]
    if args.titles:all_inputs.append(args.titles.resolve())
    for item in items:
        rgb,_,_=_read_source(item['path']);shape=rgb.shape[:2]
        if _sha(item['path'])!=item['sha256']:raise ValueError('Development source hash differs')
        fusion=load_fusion_result(native/item['id'],item['sha256'],shape)
        key=(item['sha256'],fusion['report']['parameters']['models'])
        matches=cache_index.get(key,[])
        if len(matches)!=1:raise ValueError('Need exactly one matching full evidence cache: '+item['id'])
        title,description=descriptions.get(item['id'],(item['id'],''))
        case={**item,'title':title,'description':description,'evidence_cache':str(matches[0])}
        jobs.append((case,matches[0],fusion['input_paths']))
        all_inputs += [Path(item['path']).resolve(),matches[0],*fusion['input_paths']]
    targets=[output/'manifest.json',output/'manifest.js']
    for case,_,_ in jobs:targets += [*base._paths_for_case(output,case['id'],True)[1],*_extra_targets(output,case['id'])]
    _assert_inputs_safe(targets,all_inputs)
    entries=[]
    for case,cache,native_inputs in jobs:
        print('EXPORT_FULL',case['id'],flush=True)
        rgb,_,_=_read_source(case['path']);shape=rgb.shape[:2]
        fusion=load_fusion_result(native/case['id'],case['sha256'],shape)
        evidence=base.load_evidence_cache(cache,expected_source_sha256=case['sha256'],expected_shape=shape)
        _validate_evidence(evidence,shape,True)
        identity=evidence['_review_cache_identity'];report=fusion['report']
        if (identity['models']!=report['parameters']['models']
                or evidence['provenance'].get('model_hashes')!=report.get('provenance',{}).get('model_hashes')):
            raise ValueError('Evidence and fixed fusion model provenance differ')
        entries.append(export_full_case(case,output,evidence,fusion,evidence_path=cache,
            input_paths=[manifest_path,*([args.titles.resolve()] if args.titles else [])]))
        del evidence,fusion,rgb
    write_manifest(output,entries,input_paths=all_inputs)
    print('EXPORTED_FULL_COMPARISON',len(entries),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
